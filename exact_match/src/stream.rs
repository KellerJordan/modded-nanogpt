//! StreamIndex: training rows, computed on the clock ahead of training from the documents the loader will deliver,
//! and strictly causal: a step's rows come only from the steps before it.
//!
//! The stream holds, step by step and rank by rank, each rank's documents (its inputs plus the last target) and a
//! STOP. A position of this rank's run at step s is resolved against the CAP most recent occurrences of its MIN-gram
//! in steps before s: its length is the deepest of MIN and LEVELS that one of them matches, its row from the next
//! tokens of the occurrences that match at least that length. Each phase indexes the stream up to its last step (one
//! 8-byte entry per position, 30-bit tag | 34-bit position, scattered into hash partitions and sorted by tag, so a
//! tag's entries list its occurrences in stream order) and resolves this rank's rows of its steps.
use crate::{open_shard, prefault, row, stream, Map, Output, BOS, LEVELS, MIN, STOP};
use numpy::{ndarray::Array2, IntoPyArray, PyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::os::fd::AsRawFd;
use std::sync::{Condvar, Mutex, OnceLock};

const CAP: usize = 96; // occurrences a position is resolved against, the most recent
const PHASES: [usize; 4] = [8, 32, 128, 256]; // the last steps of the phases before the final one
const MAX: usize = LEVELS[1]; // a match is not extended past the last level
const THREADS: usize = 4;
const HEADER: usize = 512; // a shard file's header, in tokens
const TOKENS_PER_SHARD: usize = 101_000_000;
const STRIDE: usize = (HEADER + TOKENS_PER_SHARD).next_multiple_of(2048); // tokens between mapped shards
const P: u64 = 0x9e3779b185ebca87;
const POSITION_BITS: u32 = 34;
const POSITION_MASK: u64 = (1 << POSITION_BITS) - 1;
const TAG_MASK: u64 = (1 << 30) - 1;
const BITS: usize = 12;
const PARTITIONS: usize = 1 << BITS;
const BURST: usize = 32;
const SMALL: usize = 1 << 16; // entries of the cache-resident sort buffer

/// Calls emit(key, p) for each position p of thread t's share of tokens[..len] (start < p <= end) that has a next
/// token and MIN tokens of its document before it, keyed on those.
fn keys(tokens: &[u16], len: usize, t: usize, emit: &mut dyn FnMut(u64, usize)) {
    let (start, end) = (len * t / THREADS, len * (t + 1) / THREADS);
    let power = P.wrapping_pow(MIN as u32);
    let (mut hash, mut available) = (0u64, 0);
    for i in start.saturating_sub(MIN - 1)..end {
        if tokens[i] == STOP || tokens[i] == BOS {
            (hash, available) = (0, 0);
        }
        if tokens[i] == STOP {
            continue;
        }
        hash = hash.wrapping_mul(P).wrapping_add(tokens[i] as u64 + 1);
        if available >= MIN {
            hash = hash.wrapping_sub((tokens[i - MIN] as u64 + 1).wrapping_mul(power));
        }
        available += 1;
        if available >= MIN && i >= start && i + 1 < len && tokens[i + 1] != STOP {
            let h = (hash ^ hash >> 30).wrapping_mul(0xbf58476d1ce4e5b9);
            let h = (h ^ h >> 27).wrapping_mul(0x94d049bb133111eb);
            emit(h ^ h >> 31, i + 1);
        }
    }
}

fn padded(n: usize) -> usize {
    n.next_multiple_of(BURST)
}

/// One stable counting-sort pass of src into dst on the 11 bits of each entry starting at `shift`.
fn radix_pass(src: &[u64], dst: &mut [u64], shift: u32) {
    let mut at = [0usize; 2048];
    for &x in src {
        at[((x >> shift) & 2047) as usize] += 1;
    }
    let mut sum = 0;
    for a in &mut at {
        (*a, sum) = (sum, sum + *a);
    }
    for &x in src {
        let d = ((x >> shift) & 2047) as usize;
        dst[at[d]] = x;
        at[d] += 1;
    }
}

/// The index build's memory, allocated and faulted in before the clock.
struct Build {
    entries: Map<u64>,
    bursts: Vec<Map<u64>>,
    scratch: Vec<Map<u64>>,
    small: Vec<Map<u64>>,
}

impl Build {
    /// Index tokens[..len] and call each(entries) with every partition's entries, sorted, on the sorting thread.
    /// Every thread streams its share of the entries into its segment of each of the hash partitions, padded to a
    /// whole burst. A partition is gathered from its segments into 256 buckets by the tag's top 8 bits, then each
    /// bucket is sorted by the remaining 22 bits through a cache-resident buffer into its final place. Every pass is
    /// stable and positions are unique, so sorting whole entries (small buckets, and partitions too big for the
    /// scratch) orders them the same.
    fn index(&mut self, tokens: &[u16], len: usize, each: impl Fn(&[u64]) + Sync) {
        let counts: Vec<Vec<usize>> = (0..THREADS)
            .into_par_iter()
            .map(|t| {
                let mut counts = vec![0; PARTITIONS];
                keys(tokens, len, t, &mut |h, _| counts[(h >> (64 - BITS)) as usize] += 1);
                counts
            })
            .collect();
        let (mut offsets, mut bases) = (vec![0; PARTITIONS + 1], vec![vec![0; PARTITIONS]; THREADS]);
        for p in 0..PARTITIONS {
            offsets[p + 1] = offsets[p];
            for t in 0..THREADS {
                bases[t][p] = offsets[p + 1];
                offsets[p + 1] += padded(counts[t][p]);
            }
        }
        let entries = Output(self.entries.ptr);
        self.bursts.par_iter_mut().zip(bases).enumerate().for_each(|(t, (burst, mut at))| {
            let mut used = vec![0; PARTITIONS];
            keys(tokens, len, t, &mut |h, pos| {
                let p = (h >> (64 - BITS)) as usize;
                burst[p * BURST + used[p]] = (h >> (64 - BITS - 30) & TAG_MASK) << POSITION_BITS | pos as u64;
                used[p] += 1;
                if used[p] == BURST {
                    unsafe { stream(entries.0.add(at[p]), &burst[p * BURST..(p + 1) * BURST]) };
                    at[p] += BURST;
                    used[p] = 0;
                }
            });
            for p in 0..PARTITIONS {
                unsafe { entries.write(at[p], &burst[p * BURST..p * BURST + used[p]]) };
            }
            crate::fence();
        });

        let parts: Vec<usize> = (0..PARTITIONS).collect();
        let chunks = parts.par_chunks(PARTITIONS.div_ceil(THREADS));
        chunks.zip(&mut self.scratch).zip(&mut self.small).for_each(|((parts, scratch), small)| {
            for &p in parts {
                let part = unsafe { entries.slice(offsets[p], offsets[p + 1] - offsets[p]) };
                let n = counts.iter().map(|c| c[p]).sum();
                let segments = || {
                    counts.iter().scan(0, |at, counts| {
                        let (start, len) = (*at, counts[p]);
                        *at += padded(len);
                        Some(start..start + len)
                    })
                };
                if n > scratch.len() {
                    let mut at = 0;
                    for segment in segments() {
                        let len = segment.len();
                        part.copy_within(segment, at);
                        at += len;
                    }
                    part[..n].sort_unstable();
                } else {
                    let tmp = &mut scratch[..n];
                    let mut starts = [0usize; 257];
                    for segment in segments() {
                        for &x in &part[segment] {
                            starts[(x >> 56) as usize + 1] += 1;
                        }
                    }
                    for b in 0..256 {
                        starts[b + 1] += starts[b];
                    }
                    let mut at = starts;
                    for segment in segments() {
                        for &x in &part[segment] {
                            let b = (x >> 56) as usize;
                            tmp[at[b]] = x;
                            at[b] += 1;
                        }
                    }
                    for b in 0..256 {
                        let (lo, hi) = (starts[b], starts[b + 1]);
                        if hi - lo <= 256 {
                            part[lo..hi].copy_from_slice(&tmp[lo..hi]);
                            part[lo..hi].sort_unstable();
                        } else if hi - lo <= small.len() {
                            radix_pass(&tmp[lo..hi], &mut small[..hi - lo], POSITION_BITS);
                            radix_pass(&small[..hi - lo], &mut part[lo..hi], POSITION_BITS + 11);
                        } else {
                            radix_pass(&tmp[lo..hi], &mut part[lo..hi], POSITION_BITS);
                            radix_pass(&part[lo..hi], &mut tmp[lo..hi], POSITION_BITS + 11);
                            part[lo..hi].copy_from_slice(&tmp[lo..hi]);
                        }
                    }
                }
                each(&part[..n]);
            }
        });
    }
}

#[pyclass(frozen)]
pub struct StreamIndex {
    rank: usize,
    world: usize,
    schedule: Vec<(usize, usize)>,
    bases: Vec<usize>,
    rows: Vec<usize>,
    corpus: Map<u16>,
    ranges: Vec<(usize, usize)>,
    plan: OnceLock<(Vec<(usize, usize)>, Vec<usize>)>,
    stream: Map<u16>,
    context: Map<u16>,
    resolved: Map<[i32; 3]>,
    build: Mutex<Build>,
    pool: rayon::ThreadPool,
    done: Mutex<Result<usize, String>>,
    ready: Condvar,
}

#[pymethods]
impl StreamIndex {
    /// Before the clock: `files` are the training shards in loader order, `schedule` has one (tokens per rank,
    /// longest document) pair per step. Allocates and faults in everything the phases write.
    #[new]
    fn new(py: Python<'_>, files: Vec<String>, schedule: Vec<(usize, usize)>, rank: usize, world: usize) -> PyResult<Self> {
        // Run k occupies bases[k]..bases[k + 1]: its n + 1 tokens, then a STOP. This rank's rows of step s are
        // rows[s]..rows[s + 1].
        let (mut bases, mut rows) = (vec![0], vec![0]);
        for &(n, _) in &schedule {
            for _ in 0..world {
                bases.push(bases.last().unwrap() + n + 2);
            }
            rows.push(rows.last().unwrap() + n);
        }
        let (total, own) = (*bases.last().unwrap(), *rows.last().unwrap());
        let pool = rayon::ThreadPoolBuilder::new().num_threads(THREADS).build().unwrap();
        let (corpus, ranges, stream, context, resolved, build) = py.detach(|| {
            pool.install(|| -> PyResult<_> {
                // Read-only private mappings of the shard files, STRIDE tokens apart in one reserved range: token k
                // of file i sits at i * STRIDE + HEADER + k. The page cache is the corpus.
                let len = files.len() * STRIDE;
                let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
                let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len * 2, libc::PROT_READ, flags, -1, 0) };
                if ptr == libc::MAP_FAILED {
                    return Err(std::io::Error::last_os_error().into());
                }
                let corpus = Map { ptr: ptr.cast::<u16>(), len, bytes: len * 2 };
                let base = &corpus;
                let ranges = files
                    .par_iter()
                    .enumerate()
                    .map(|(i, path)| {
                        let (file, n) = open_shard(path)?;
                        let (at, bytes) = (unsafe { base.ptr.add(i * STRIDE) }.cast(), (HEADER + n) * 2);
                        let flags = libc::MAP_PRIVATE | libc::MAP_FIXED;
                        if HEADER + n > STRIDE || unsafe { libc::mmap(at, bytes, libc::PROT_READ, flags, file.as_raw_fd(), 0) } == libc::MAP_FAILED {
                            return Err(PyValueError::new_err(format!("cannot map shard {path}")));
                        }
                        // The loader reads about total / TOKENS_PER_SHARD shards.
                        if i < total / TOKENS_PER_SHARD + 2 {
                            unsafe { libc::madvise(at, bytes, libc::MADV_POPULATE_READ) };
                        }
                        Ok((i * STRIDE + HEADER, i * STRIDE + HEADER + n))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                let maps = |len| (0..THREADS).map(|_| Map::new(len)).collect::<std::io::Result<Vec<_>>>();
                let mut build = Build {
                    entries: Map::new(total + THREADS * PARTITIONS * (BURST - 1))?,
                    bursts: maps(PARTITIONS * BURST)?,
                    scratch: maps(2 * total.div_ceil(PARTITIONS) + 4096)?,
                    small: maps(SMALL)?,
                };
                let (mut stream, mut context, mut resolved) = (Map::new(total)?, Map::new(total)?, Map::new(own)?);
                prefault(&mut stream);
                prefault(&mut context);
                prefault(&mut resolved);
                prefault(&mut build.entries);
                for m in build.bursts.iter_mut().chain(&mut build.scratch).chain(&mut build.small) {
                    prefault(m);
                }
                Ok((corpus, ranges, stream, context, resolved, build))
            })
        })?;
        Ok(Self {
            rank,
            world,
            schedule,
            bases,
            rows,
            corpus,
            ranges,
            plan: OnceLock::new(),
            stream,
            context,
            resolved,
            build: Mutex::new(build),
            pool,
            done: Mutex::new(Ok(0)),
            ready: Condvar::new(),
        })
    }

    /// On the clock: plan the loader's documents for every step, then index and resolve the steps phase by phase.
    fn build(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| -> Result<(), String> {
            let planned = self.pool.install(|| self.plan());
            if let Err(e) = &planned {
                *self.done.lock().unwrap() = Err(e.clone());
                self.ready.notify_all();
            }
            planned?;
            let (steps, mut first) = (self.schedule.len(), 0);
            for last in PHASES.into_iter().filter(|&e| e < steps).chain([steps]) {
                self.pool.install(|| self.phase(first, last));
                *self.done.lock().unwrap() = Ok(last);
                self.ready.notify_all();
                first = last;
            }
            Ok(())
        })
        .map_err(PyValueError::new_err)
    }

    /// This rank's rows of `step` (cell, two slots), once resolved. Fails unless the loader's documents for it (every
    /// rank's starts and ends within a shard of `size` tokens) are the planned ones.
    fn rows<'py>(
        &self,
        py: Python<'py>,
        step: usize,
        size: usize,
        starts: Vec<Vec<usize>>,
        ends: Vec<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let done = py.detach(|| self.ready.wait_while(self.done.lock().unwrap(), |d| matches!(d, Ok(last) if *last <= step)).unwrap().clone());
        done.map_err(PyValueError::new_err)?;
        let (docs, runs) = self.plan.get().unwrap();
        let k = step * self.world;
        let (lo, hi) = self.ranges[docs[runs[k]].0 / STRIDE];
        let planned = hi - lo == size
            && (0..self.world).all(|q| {
                let docs = &docs[runs[k + q]..runs[k + q + 1]];
                docs.len() == starts[q].len() && docs.iter().zip(&starts[q]).zip(&ends[q]).all(|((&doc, &s), &e)| doc == (lo + s, lo + e))
            });
        if !planned {
            return Err(PyValueError::new_err(format!("loader documents at step {step} differ from the plan")));
        }
        let rows = &self.resolved[self.rows[step]..self.rows[step + 1]];
        Ok(Array2::from_shape_vec((rows.len(), 3), rows.as_flattened().to_vec()).unwrap().into_pyarray(py))
    }
}

impl StreamIndex {
    /// The loader's document selection over every step: docs[runs[k]..runs[k + 1]] (corpus ranges) make up run
    /// k = step * world + rank. As in Shard.next_batch, a run takes documents of n + 1 tokens in all, each cut at the
    /// step's longest document, and a step the shard runs out of documents for is taken from the next shard.
    fn plan(&self) -> Result<(), String> {
        let bos = |shard: usize| -> Vec<usize> {
            let (lo, hi) = self.ranges[shard];
            (lo..hi).into_par_iter().filter(|&i| self.corpus[i] == BOS).map(|i| i - lo).collect()
        };
        let (mut docs, mut runs) = (Vec::new(), vec![0]);
        let (mut shard, mut starts, mut i) = (0, bos(0), 0);
        for &(n, max_len) in &self.schedule {
            let (d, r) = (docs.len(), runs.len());
            let (mut taken, mut length) = (0, 0);
            while taken < self.world {
                let Some(&start) = starts.get(i) else {
                    (shard, i, taken, length) = (shard + 1, 0, 0, 0);
                    if shard == self.ranges.len() {
                        return Err("the schedule needs more training shards".into());
                    }
                    starts = bos(shard);
                    docs.truncate(d);
                    runs.truncate(r);
                    continue;
                };
                let (lo, hi) = self.ranges[shard];
                i += 1;
                let end = starts.get(i).map_or(hi - lo, |&s| s).min(start + max_len).min(start + n - length + 1);
                docs.push((lo + start, lo + end));
                length += end - start;
                if length > n {
                    runs.push(docs.len());
                    (taken, length) = (taken + 1, 0);
                }
            }
        }
        self.plan.set((docs, runs)).map_err(|_| "already planned".to_string())
    }

    /// Write steps first..last into the stream, index every step before `last`, and resolve this rank's rows of steps
    /// first..last.
    fn phase(&self, first: usize, last: usize) {
        let (world, rank, bases, rows) = (self.world, self.rank, &self.bases, &self.rows);
        let (docs, runs) = self.plan.get().unwrap();
        // Copy each run's documents, then a STOP; context[g] counts the tokens from the start of token g - 1's
        // segment (its run or its document) through g - 1, at most MAX.
        let (stream, context) = (Output(self.stream.ptr), Output(self.context.ptr));
        (first * world..last * world).into_par_iter().for_each(|k| {
            let (mut at, mut segment) = (bases[k], bases[k]);
            for &(s, e) in &docs[runs[k]..runs[k + 1]] {
                for (j, &t) in (at..).zip(&self.corpus[s..e]) {
                    if t == BOS {
                        segment = j;
                    }
                    unsafe { context.set(j + 1, (j + 1 - segment).min(MAX) as u16) };
                }
                unsafe { stream.write(at, &self.corpus[s..e]) };
                at += e - s;
            }
            unsafe { stream.set(at, STOP) };
        });

        let (lo, hi) = (bases[first * world], bases[last * world]);
        let (stream, context) = (&self.stream[..hi], &self.context[..hi]);
        let resolved = Output(self.resolved.ptr);
        let position = |v: u64| (v & POSITION_MASK) as usize;
        self.build.lock().unwrap().index(stream, hi, |part| {
            let (mut found, mut tokens) = (Vec::with_capacity(CAP), Vec::with_capacity(CAP));
            for group in part.chunk_by(|a, b| a >> POSITION_BITS == b >> POSITION_BITS) {
                // group[..b]: the key's occurrences in steps before the current entry's step.
                let mut b = 0;
                for (k, &v) in group.iter().enumerate().skip(1) {
                    let g = position(v);
                    if g < lo || g >= hi {
                        continue;
                    }
                    let run = bases.partition_point(|&x| x < g) - 1;
                    if run % world != rank {
                        continue;
                    }
                    while b < k && position(group[b]) < bases[run - rank] {
                        b += 1;
                    }
                    // Match against each candidate as far back as both segments reach.
                    let query = &stream[g - context[g] as usize..g];
                    found.clear();
                    for c in group[..b].iter().rev().take(CAP).map(|&w| position(w)) {
                        let floor = c - context[c] as usize;
                        if c < floor + MIN || stream[c - MIN..c] != query[query.len() - MIN..] {
                            continue;
                        }
                        let maximum = MAX.min(query.len()).min(c - floor);
                        let mut length = MIN;
                        while length < maximum && stream[c - length - 1] == query[query.len() - length - 1] {
                            length += 1;
                        }
                        found.push((length, stream[c] as i32));
                    }
                    let Some(best) = found.iter().map(|&(l, _)| l).max() else { continue };
                    let level = LEVELS.iter().rev().copied().find(|&l| l <= best).unwrap_or(MIN);
                    tokens.clear();
                    tokens.extend(found.iter().filter(|&&(l, _)| l >= level).map(|&(_, t)| t));
                    unsafe { resolved.set(rows[run / world] + g - bases[run] - 1, row(level, &mut tokens)) };
                }
            }
        });
    }
}
