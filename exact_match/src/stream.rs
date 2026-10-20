//! StreamIndex: training rows, computed on the clock ahead of training from the documents the loader will deliver,
//! and strictly causal: a step's rows come only from the steps before it.
//!
//! The stream holds, step by step and rank by rank, each rank's documents (its inputs plus the last target) and a
//! STOP. A position of a run at step s is resolved against the CAP most recent occurrences of its MIN-gram in steps
//! before s: its length is the deepest of MIN and LEVELS that one of them matches, its row from the next tokens of
//! the occurrences that match at least that length. Each phase indexes the stream up to its last
//! step (one 8-byte entry per position, 30-bit tag | 34-bit position, scattered into hash partitions and radix-sorted
//! by tag, so a tag's entries list its occurrences in stream order) and resolves every rank's rows of its steps.
//!
//! Rank 0 builds every rank's rows once, on its own pinned pool, into shared memory that the other ranks map:
//! the plan (the loader's documents of every step), the rows, and a counter of the steps resolved so far, which
//! readers wait on with a futex.
//!
//! Rows are compact: each position's cell, and per run a list of the matched positions' candidates (lib.rs's row),
//! appended under an atomic count in the order resolved and sorted when read, up to the run's capacity.
//!
//! The build is written for AVX-512 (the record's hosts); without it (CPU tests) the same code runs on the baseline ISA.
use crate::{compact, create_empty, madvise, mmap, HEADER_BYTES, open_shard, panic_message, pinned_pool, prefault, row, stream, Map, Output, AVX512, BOS, LEVELS, MIN, STOP};
use numpy::PyArray1;
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::{Acquire, Relaxed, Release}};
use std::sync::Mutex;

const CAP: usize = 96; // occurrences a position is resolved against, the most recent
const MAX: usize = LEVELS[LEVELS.len() - 1]; // the deepest level, which a match is not extended past
const PHASES: [usize; 2] = [8, 32]; // the last steps of the phases before the final one: the first rows early, then
                                    // one full pass, so the stream is indexed (8 + 32 + 575) / 575 = 1.07 times
const HEADER: usize = HEADER_BYTES / 2; // a shard file's header, in tokens
// The largest shard the corpus mapping has room for (create fails on a larger one): FineWeb shards hold 100M tokens.
const MAX_SHARD_TOKENS: usize = 101_000_000;
const STRIDE: usize = (HEADER + MAX_SHARD_TOKENS).next_multiple_of(2048); // tokens between mapped shards
// The tokens a shard gives the loader, to estimate how many shards it reaches (the plan fails if that is too few).
const SHARD_TOKENS: usize = 100_000_000;
const P: u64 = 0x9e3779b185ebca87; // xxHash64's first prime: an odd multiplier with well-mixed bits
const POSITION_BITS: u32 = 34;
const POSITION_MASK: u64 = (1 << POSITION_BITS) - 1;
const TAG_MASK: u64 = (1 << 30) - 1;
const BITS: usize = 12;
const PARTITIONS: usize = 1 << BITS;
const BURST: usize = 32;
const PREFETCH: usize = 64; // entries ahead of the resolve whose stream lines are prefetched
const PAGE_BITS: u32 = 12; // the run lookup's pages: 4096 positions, shorter than any run
const STEP: usize = 1 << 12; // positions hashed at once
const FAILED: u32 = u32::MAX; // the shared counter if rank 0's build failed

/// The rolling n-gram hash's weights: the token k before a window's end counts (token + 1) * P^k.
const WEIGHTS: [u64; 16] = {
    let mut w = [1u64; 16];
    let mut k = 1;
    while k < 16 {
        w[k] = w[k - 1].wrapping_mul(P);
        k += 1;
    }
    w
};

/// out[r]: the key of the MIN-token window ending at tokens[i0 + r], r < n. The polynomial a rolling hash keeps,
/// computed per position from its own tokens (independent multiplies that vectorize), then mixed.
#[inline(always)]
unsafe fn hash_body(tokens: &[u16], i0: usize, n: usize, out: &mut [u64; STEP]) {
    assert!(i0 + 1 >= MIN && i0 + n <= tokens.len() && n <= STEP);
    let (tp, op) = (tokens.as_ptr(), out.as_mut_ptr());
    for r in 0..n {
        let mut hash = 0u64;
        for (k, &w) in WEIGHTS[..MIN].iter().enumerate() {
            hash = hash.wrapping_add((*tp.add(i0 + r - k) as u64 + 1).wrapping_mul(w));
        }
        // splitmix64's finalizer
        let h = (hash ^ hash >> 30).wrapping_mul(0xbf58476d1ce4e5b9);
        let h = (h ^ h >> 27).wrapping_mul(0x94d049bb133111eb);
        *op.add(r) = h ^ h >> 31;
    }
}

/// hash_body on AVX-512 if the host has it.
fn hash(tokens: &[u16], i0: usize, n: usize, out: &mut [u64; STEP]) {
    #[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
    unsafe fn avx512(tokens: &[u16], i0: usize, n: usize, out: &mut [u64; STEP]) {
        hash_body(tokens, i0, n, out)
    }
    if *AVX512 { unsafe { avx512(tokens, i0, n, out) } } else { unsafe { hash_body(tokens, i0, n, out) } }
}

/// Calls emit(key, p) for each position p of thread t's (of `threads`) share of tokens[..len] (start < p <= end) that
/// has a next token and MIN tokens of its document before it (context[p] >= MIN), keyed on those.
#[inline(always)]
fn keys(tokens: &[u16], context: &[u16], len: usize, t: usize, threads: usize, mut emit: impl FnMut(u64, usize)) {
    let (start, end) = (len * t / threads, len * (t + 1) / threads);
    let mut hashes = [0u64; STEP];
    let mut i0 = start.max(MIN - 1);
    while i0 < end {
        let n = STEP.min(end - i0);
        hash(tokens, i0, n, &mut hashes);
        for (r, &h) in hashes[..n].iter().enumerate() {
            let g = i0 + r + 1;
            if g < len && context[g] as usize >= MIN && tokens[g] != STOP {
                emit(h, g);
            }
        }
        i0 += n;
    }
}

/// The offsets (plus `at`) of the BOS tokens in t: 32-token compares to a mask, then its set bits.
fn bos_scan(t: &[u16], at: usize) -> Vec<usize> {
    #[inline(always)]
    fn masks(t: &[u16], at: usize, out: &mut Vec<usize>) {
        for (b, block) in t.chunks(32).enumerate() {
            let mut mask = block.iter().enumerate().fold(0u32, |m, (j, &x)| m | ((x == BOS) as u32) << j);
            while mask != 0 {
                out.push(at + b * 32 + mask.trailing_zeros() as usize);
                mask &= mask - 1;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn masks_avx512(t: &[u16], at: usize, out: &mut Vec<usize>) {
        masks(t, at, out)
    }
    let mut out = Vec::with_capacity(t.len() / 256);
    if *AVX512 { unsafe { masks_avx512(t, at, &mut out) } } else { masks(t, at, &mut out) };
    out
}

/// How many tokens before positions a and b agree, from a - 1 and b - 1 back, up to `max` (a multiple of 4; the
/// caller caps it at the deepest level): four tokens per compare.
#[inline(always)]
fn agree(s: &[u16], a: usize, b: usize, max: usize) -> usize {
    if a < max || b < max {
        return (0..max.min(a).min(b)).find(|&n| s[a - n - 1] != s[b - n - 1]).unwrap_or(max.min(a).min(b));
    }
    let (pa, pb) = (s[a - max..a].as_ptr(), s[b - max..b].as_ptr());
    for n in (0..max).step_by(4) {
        let x = unsafe { pa.add(max - n - 4).cast::<u64>().read_unaligned() ^ pb.add(max - n - 4).cast::<u64>().read_unaligned() };
        if x != 0 {
            return n + x.leading_zeros() as usize / 16;
        }
    }
    max
}

/// The run that position g is in.
fn run_of(bases: &[usize], g: usize) -> usize {
    bases.partition_point(|&x| x <= g) - 1
}

fn padded(n: usize) -> usize {
    n.next_multiple_of(BURST)
}

/// The index build's memory, allocated and faulted in before the clock.
struct Build {
    threads: usize,
    entries: Map<u64>,
    bursts: Vec<Map<u64>>,
    scratch: Vec<Map<u64>>,
}

impl Build {
    /// Index tokens[..len] (context: as in the stream) and call each(entries) with every partition's entries, sorted,
    /// on the sorting thread. Every thread streams its share of the entries into its segment of each of the hash
    /// partitions, padded to a whole burst. A partition is radix-sorted by tag from its segments through a
    /// cache-resident scratch. Segments list positions in order and every pass is stable, so the entries end up
    /// sorted by tag and then position, as sorting whole entries (positions are unique; small and oversized
    /// partitions) would.
    fn index(&mut self, tokens: &[u16], context: &[u16], len: usize, each: impl Fn(&[u64]) + Sync) {
        let threads = self.threads;
        let counts: Vec<Vec<usize>> = (0..threads)
            .into_par_iter()
            .map(|t| {
                let mut counts = vec![0; PARTITIONS];
                keys(tokens, context, len, t, threads, |h, _| counts[(h >> (64 - BITS)) as usize] += 1);
                counts
            })
            .collect();
        let (mut offsets, mut bases) = (vec![0; PARTITIONS + 1], vec![vec![0; PARTITIONS]; threads]);
        for p in 0..PARTITIONS {
            offsets[p + 1] = offsets[p];
            for t in 0..threads {
                bases[t][p] = offsets[p + 1];
                offsets[p + 1] += padded(counts[t][p]);
            }
        }
        let entries = Output(self.entries.ptr);
        self.bursts.par_iter_mut().zip(bases).enumerate().for_each(|(t, (burst, mut at))| {
            let mut used = vec![0; PARTITIONS];
            keys(tokens, context, len, t, threads, |h, pos| {
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

        // Partitions are handed out largest first: hot keys make the resolve uneven.
        let mut parts: Vec<usize> = (0..PARTITIONS).collect();
        parts.sort_by_key(|&p| std::cmp::Reverse(offsets[p + 1] - offsets[p]));
        let next = AtomicUsize::new(0);
        self.scratch.par_iter_mut().for_each(|scratch| {
            let mut digits = vec![0u32; 2 << 15];
            while let Some(&p) = parts.get(next.fetch_add(1, Relaxed)) {
                let part = unsafe { entries.slice(offsets[p], offsets[p + 1] - offsets[p]) };
                let n = counts.iter().map(|c| c[p]).sum();
                let segments = || {
                    counts.iter().scan(0, |at, counts| {
                        let (start, len) = (*at, counts[p]);
                        *at += padded(len);
                        Some(start..start + len)
                    })
                };
                let sorted: &[u64] = if n > scratch.len() {
                    let mut at = 0;
                    for segment in segments() {
                        let len = segment.len();
                        part.copy_within(segment, at);
                        at += len;
                    }
                    part[..n].sort_unstable();
                    &part[..n]
                } else if n < 1 << 11 {
                    let tmp = &mut scratch[..n];
                    let mut at = 0;
                    for segment in segments() {
                        tmp[at..at + segment.len()].copy_from_slice(&part[segment.clone()]);
                        at += segment.len();
                    }
                    tmp.sort_unstable();
                    tmp
                } else {
                    // LSD radix sort by the tag, `width` bits per pass (all digits counted in one read), from the
                    // segments into the scratch, then back and forth. Stable, so equal tags keep position order.
                    let width = if n >= 1 << 15 { 15 } else { 10 };
                    let (passes, mask) = (30u32.div_ceil(width), (1usize << width) - 1);
                    let tally = &mut digits[..(passes as usize) << width];
                    tally.fill(0);
                    let cp = tally.as_mut_ptr();
                    let digit = |x: u64, d: u32| (x >> (POSITION_BITS + d * width)) as usize & mask;
                    for segment in segments() {
                        for &x in &part[segment] {
                            for d in 0..passes {
                                unsafe { *cp.add(((d as usize) << width) + digit(x, d)) += 1 };
                            }
                        }
                    }
                    for c in tally.chunks_mut(1 << width) {
                        let mut sum = 0;
                        for c in c {
                            (*c, sum) = (sum, sum + *c);
                        }
                    }
                    let (tp, pp) = (scratch.as_mut_ptr(), part.as_mut_ptr());
                    for segment in segments() {
                        for &x in &part[segment] {
                            unsafe {
                                let c = cp.add(digit(x, 0));
                                *tp.add(*c as usize) = x;
                                *c += 1;
                            }
                        }
                    }
                    let (mut src, mut dst) = (tp, pp);
                    for d in 1..passes {
                        let c0 = unsafe { cp.add((d as usize) << width) };
                        for i in 0..n {
                            unsafe {
                                let x = *src.add(i);
                                let c = c0.add(digit(x, d));
                                *dst.add(*c as usize) = x;
                                *c += 1;
                            }
                        }
                        (src, dst) = (dst, src);
                    }
                    unsafe { std::slice::from_raw_parts(src, n) }
                };
                each(sorted);
            }
        });
    }
}

/// Byte offsets in the shared file of the plan's runs (one u64 per run and an end), its documents (corpus ranges,
/// at most one per stream token), the rows (`ints` in all) and the compact entries (a 64-byte count per run of
/// `counted`, then `entries` of 8 bytes), and the file's size; the counter and the document count come first.
fn layout(runs: usize, tokens: usize, ints: usize, counted: usize, entries: usize) -> [usize; 5] {
    const PAGE: usize = 1 << 21;
    let runs_at = 4096;
    let docs_at = (runs_at + 8 * (runs + 1)).next_multiple_of(PAGE);
    let rows_at = (docs_at + 16 * tokens).next_multiple_of(PAGE);
    let entries_at = (rows_at + 4 * ints).next_multiple_of(PAGE);
    [runs_at, docs_at, rows_at, entries_at, (entries_at + 64 * counted + 8 * entries).next_multiple_of(PAGE)]
}

/// Rank 0's build state.
struct Builder {
    pool: rayon::ThreadPool,
    bases: Vec<usize>,
    pages: Vec<u32>,
    owned: Vec<(usize, usize)>,
    corpus: Map<u16>,
    stream: Map<u16>,
    context: Map<u16>,
    build: Mutex<Build>,
}

#[pyclass(frozen)]
pub struct StreamIndex {
    rank: usize,
    world: usize,
    schedule: Vec<(usize, usize)>,
    rows: Vec<usize>, // run k's first row
    capacity: Vec<usize>, // run k's entries
    first: Vec<usize>, // run k's first entry
    ranges: Vec<(usize, usize)>, // each shard's tokens in corpus coordinates
    shared: Map<u8>,
    at: [usize; 5],
    builder: Option<Builder>,
}

#[pymethods]
impl StreamIndex {
    /// Rank 0, before the clock: create the shared file `path` (new or empty). `files` are the training shards in
    /// loader order, `schedule` has one (tokens per rank, longest document) pair per step. Allocates and faults in
    /// everything the phases write. The build runs on `threads` threads pinned round-robin to `cpus` (or unpinned).
    /// `entries`: each step's capacity of candidates per run.
    #[staticmethod]
    fn create(py: Python<'_>, path: String, files: Vec<String>, schedule: Vec<(usize, usize)>, world: usize, threads: usize, cpus: Vec<usize>, entries: Vec<usize>) -> PyResult<Self> {
        // The run at each page's start. Runs are longer than a page, so position g's run is its page's or the next.
        if schedule.iter().any(|&(n, _)| n + 2 <= 1 << PAGE_BITS) {
            return Err(PyValueError::new_err("a run is shorter than a page of the run lookup"));
        }
        let mut index = Self::open(&path, &files, schedule, 0, world, entries, true)?;
        // Run k occupies bases[k]..bases[k + 1]: its n + 1 tokens, then a STOP.
        let mut bases = vec![0];
        for &(n, _) in index.schedule.iter().flat_map(|s| std::iter::repeat_n(s, world)) {
            bases.push(bases.last().unwrap() + n + 2);
        }
        let total = *bases.last().unwrap();
        let mut pages = vec![0u32; total.div_ceil(1 << PAGE_BITS)];
        for (i, page) in pages.iter_mut().enumerate().skip(1) {
            *page = run_of(&bases, i << PAGE_BITS) as u32;
        }
        // Per run: its step's start (its candidates come before it) and the offset of a position's row from it.
        let owned = (0..bases.len() - 1).map(|k| (bases[k - k % world], index.rows[k].wrapping_sub(bases[k] + 1))).collect();
        let pool = pinned_pool(threads, cpus, "streamindex")?;
        let (faulted, files) = (index.faulted(), &files[..index.ranges.len()]);
        let shared = &mut index.shared;
        let (corpus, stream, context, build) = py.detach(|| {
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
                files.par_iter().enumerate().try_for_each(|(i, path)| {
                    let (file, n) = open_shard(path)?;
                    let (at, bytes) = (unsafe { base.ptr.add(i * STRIDE) }.cast(), (HEADER + n) * 2);
                    let flags = libc::MAP_PRIVATE | libc::MAP_FIXED;
                    if HEADER + n > STRIDE || unsafe { libc::mmap(at, bytes, libc::PROT_READ, flags, file.as_raw_fd(), 0) } == libc::MAP_FAILED {
                        return Err(PyValueError::new_err(format!("cannot map shard {path}")));
                    }
                    madvise(at, bytes, libc::MADV_POPULATE_READ)?;
                    PyResult::Ok(())
                })?;
                let maps = |len| (0..threads).map(|_| Map::new(len)).collect::<std::io::Result<Vec<_>>>();
                let mut build = Build {
                    threads,
                    entries: Map::new(total + threads * PARTITIONS * (BURST - 1))?,
                    bursts: maps(PARTITIONS * BURST)?,
                    scratch: maps(2 * total.div_ceil(PARTITIONS) + 4096)?,
                };
                let (mut stream, mut context) = (Map::new(total)?, Map::new(total)?);
                prefault(&mut stream);
                prefault(&mut context);
                prefault(&mut build.entries);
                for m in build.bursts.iter_mut().chain(&mut build.scratch) {
                    prefault(m);
                }
                for (a, b) in faulted {
                    prefault(&mut shared[a..b]);
                }
                Ok((corpus, stream, context, build))
            })
        })?;
        index.builder = Some(Builder { pool, bases, pages, owned, corpus, stream, context, build: Mutex::new(build) });
        Ok(index)
    }

    /// The other ranks, before the clock, once rank 0 has created `path` (the same `files`, `schedule`, `world` and
    /// `entries`).
    #[staticmethod]
    fn attach(path: String, files: Vec<String>, schedule: Vec<(usize, usize)>, rank: usize, world: usize, entries: Vec<usize>) -> PyResult<Self> {
        let index = Self::open(&path, &files, schedule, rank, world, entries, false)?;
        // Map what rank 0 faulted in, so that reading it takes no page faults.
        for (a, b) in index.faulted() {
            madvise(unsafe { index.shared.ptr.add(a) }.cast(), b - a, libc::MADV_POPULATE_READ)?;
        }
        Ok(index)
    }

    /// Rank 0, on the clock: plan the loader's documents for every step, then index and resolve the steps phase by
    /// phase, publishing each phase's last step.
    fn build(&self, py: Python<'_>) -> PyResult<()> {
        let b = self.builder.as_ref().ok_or_else(|| PyValueError::new_err("only the creating rank builds"))?;
        if self.ready().load(Acquire) != 0 {
            return Err(PyValueError::new_err("the training index is already built"));
        }
        py.detach(|| {
            // Any failure, a panic included, is published, so that no rank waits for steps that never come.
            let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
                let (docs, runs) = b.pool.install(|| self.plan(b))?;
                let (steps, mut first) = (self.schedule.len(), 0);
                for last in PHASES.iter().copied().filter(|&e| e < steps).chain([steps]) {
                    b.pool.install(|| self.phase(b, &docs, &runs, first, last));
                    self.publish(last as u32);
                    first = last;
                }
                Ok(())
            }))
            .unwrap_or_else(|panic| Err(format!("the training index build panicked: {}", panic_message(&*panic))));
            if built.is_err() {
                self.publish(FAILED);
            }
            built
        })
        .map_err(PyValueError::new_err)
    }

    /// This rank's rows of `step`, once resolved: compact (lib.rs's compact), the step's capacity of entries. Fails
    /// unless the loader's documents for it (every rank's starts and ends within a shard of `size` tokens) are the
    /// planned ones.
    fn rows<'py>(&self, py: Python<'py>, step: usize, size: usize, starts: Vec<Vec<usize>>, ends: Vec<Vec<usize>>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        py.detach(|| self.wait(step))?;
        if starts.len() != self.world || ends.len() != self.world {
            return Err(PyValueError::new_err(format!("step {step}: document lists of {} ranks, expected {}", starts.len(), self.world)));
        }
        let (runs, docs) = (self.runs(), self.docs());
        let k = step * self.world;
        let (lo, hi) = self.ranges[docs[runs[k] as usize][0] as usize / STRIDE];
        let planned = hi - lo == size
            && (0..self.world).all(|q| {
                let docs = &docs[runs[k + q] as usize..runs[k + q + 1] as usize];
                docs.len() == starts[q].len() && docs.iter().zip(&starts[q]).zip(&ends[q]).all(|((d, &s), &e)| d[0] as usize == lo + s && d[1] as usize == lo + e)
            });
        if !planned {
            return Err(PyValueError::new_err(format!("loader documents at step {step} differ from the plan")));
        }
        let (run, n) = (k + self.rank, self.schedule[step].0);
        let (used, capacity) = (self.counter(run).load(Relaxed) as usize, self.capacity[run]);
        // Which rows would lose their candidates depends on the resolve threads' timing: an overflow fails the run.
        if used > capacity {
            return Err(PyValueError::new_err(format!("rank {} step {step}: {used} hint entries, past the capacity of {capacity}", self.rank)));
        }
        let entries = unsafe { std::slice::from_raw_parts(self.entries_ptr().add(self.first[run]), used) };
        let cells = unsafe { std::slice::from_raw_parts(self.cells_ptr().add(self.rows[run]), n) };
        Ok(compact(py, cells, &mut entries.to_vec(), capacity).0)
    }
}

impl StreamIndex {
    /// Create (rank 0) or open the shared file and map it, and lay out every run's rows.
    #[allow(clippy::too_many_arguments)]
    fn open(path: &str, files: &[String], schedule: Vec<(usize, usize)>, rank: usize, world: usize, entries: Vec<usize>, create: bool) -> PyResult<Self> {
        let (mut rows, mut total) = (vec![], 0);
        for &(n, _) in schedule.iter().flat_map(|s| std::iter::repeat_n(s, world)) {
            rows.push(total);
            total += n;
        }
        let runs = schedule.len() * world;
        if total + 2 * runs > 1 << POSITION_BITS {
            return Err(PyValueError::new_err("the training stream is too long for the entries' position bits"));
        }
        if entries.len() != schedule.len() {
            return Err(PyValueError::new_err("entries: one capacity per step"));
        }
        let capacity: Vec<usize> = entries.iter().flat_map(|&c| std::iter::repeat_n(c, world)).collect();
        let first = capacity.iter().scan(0, |at, &c| Some(std::mem::replace(at, *at + c))).collect();
        let at = layout(runs, total + 2 * runs, total, runs, capacity.iter().sum());
        let file = if create {
            let file = create_empty(path)?;
            file.set_len(at[4] as u64)?;
            file
        } else {
            let file = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
            if file.metadata()?.len() != at[4] as u64 {
                return Err(PyValueError::new_err("the shared training index has another layout"));
            }
            file
        };
        let ptr = mmap(at[4], libc::MAP_SHARED, file.as_raw_fd())?;
        let shared = Map { ptr: ptr.cast::<u8>(), len: at[4], bytes: at[4] };
        // Only the shards the loader can reach: about one per SHARD_TOKENS stream tokens, and two more (the loader
        // discards the document tails at each shard's end).
        let reach = ((total + 2 * runs) / SHARD_TOKENS + 2).min(files.len());
        let ranges = files[..reach].iter().enumerate().map(|(i, f)| Ok((i * STRIDE + HEADER, i * STRIDE + HEADER + open_shard(f)?.1))).collect::<PyResult<_>>()?;
        Ok(Self { rank, world, schedule, rows, capacity, first, ranges, shared, at, builder: None })
    }

    /// The parts of the shared file rank 0 faults in before the clock: the plan's runs, its first documents, and the
    /// rows' cells and entries.
    fn faulted(&self) -> [(usize, usize); 3] {
        let [runs, docs, rows, _, end] = self.at;
        [(runs, docs), (docs, (docs + (1 << 26)).min(rows)), (rows, end)]
    }

    /// Run k's count of entries appended.
    fn counter(&self, run: usize) -> &AtomicU32 {
        unsafe { &*self.shared.ptr.add(self.at[3] + 64 * run).cast::<AtomicU32>() }
    }

    /// Every run's entry list (`first`). Raw, as the build writes some runs' entries while the workers read others'.
    fn entries_ptr(&self) -> *mut [i32; 2] {
        unsafe { self.shared.ptr.add(self.at[3] + 64 * self.schedule.len() * self.world) }.cast()
    }

    /// Every run's cells (`rows`). Raw, like `entries_ptr`.
    fn cells_ptr(&self) -> *mut i32 {
        unsafe { self.shared.ptr.add(self.at[2]) }.cast()
    }

    /// The shared counter: every step before it is resolved (FAILED: rank 0's build failed).
    fn ready(&self) -> &AtomicU32 {
        unsafe { &*self.shared.ptr.cast::<AtomicU32>() }
    }

    fn publish(&self, value: u32) {
        self.ready().store(value, Release);
        unsafe { libc::syscall(libc::SYS_futex, self.ready().as_ptr(), libc::FUTEX_WAKE, i32::MAX) };
    }

    /// Wait until `step` is resolved.
    fn wait(&self, step: usize) -> PyResult<()> {
        loop {
            match self.ready().load(Acquire) {
                FAILED => return Err(PyValueError::new_err("rank 0's training index build failed")),
                v if v as usize > step => return Ok(()),
                v => unsafe { libc::syscall(libc::SYS_futex, self.ready().as_ptr(), libc::FUTEX_WAIT, v, std::ptr::null::<libc::timespec>()) },
            };
        }
    }

    fn runs(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.shared.ptr.add(self.at[0]).cast(), self.schedule.len() * self.world + 1) }
    }

    fn docs(&self) -> &[[u64; 2]] {
        let n = unsafe { &*self.shared.ptr.add(8).cast::<AtomicU64>() }.load(Relaxed) as usize;
        unsafe { std::slice::from_raw_parts(self.shared.ptr.add(self.at[1]).cast(), n) }
    }

    /// The loader's document selection over every step: docs[runs[k]..runs[k + 1]] (corpus ranges) make up run
    /// k = step * world + rank. As in Shard.next_batch, a run takes documents of n + 1 tokens in all, each cut at the
    /// step's longest document, and a step the shard runs out of documents for is taken from the next shard. Also
    /// written to the shared file, for every rank's loader check.
    fn plan(&self, b: &Builder) -> Result<(Vec<(usize, usize)>, Vec<usize>), String> {
        let bos = |shard: usize| -> Vec<usize> {
            let (lo, hi) = self.ranges[shard];
            let pieces: Vec<Vec<usize>> = b.corpus[lo..hi].par_chunks(1 << 20).enumerate().map(|(c, t)| bos_scan(t, c << 20)).collect();
            pieces.concat()
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
        let (runs_out, docs_out) = (Output(unsafe { self.shared.ptr.add(self.at[0]) }.cast::<u64>()), Output(unsafe { self.shared.ptr.add(self.at[1]) }.cast::<[u64; 2]>()));
        for (k, &r) in runs.iter().enumerate() {
            unsafe { runs_out.set(k, r as u64) };
        }
        for (k, &(s, e)) in docs.iter().enumerate() {
            unsafe { docs_out.set(k, [s as u64, e as u64]) };
        }
        unsafe { &*self.shared.ptr.add(8).cast::<AtomicU64>() }.store(docs.len() as u64, Relaxed);
        Ok((docs, runs))
    }

    /// Write steps first..last into the stream, index every step before `last`, and resolve every rank's rows of
    /// steps first..last.
    fn phase(&self, b: &Builder, docs: &[(usize, usize)], runs: &[usize], first: usize, last: usize) {
        let (world, bases, pages, owned) = (self.world, &b.bases, &b.pages, &b.owned);
        // Copy each run's documents, then a STOP; context[g] counts the tokens from the start of token g - 1's
        // segment (its run or its document) through g - 1, at most the deepest level.
        let (stream, context) = (Output(b.stream.ptr), Output(b.context.ptr));
        (first * world..last * world).into_par_iter().for_each(|k| {
            let (mut at, mut segment) = (bases[k], bases[k]);
            for &(s, e) in &docs[runs[k]..runs[k + 1]] {
                for (j, &t) in (at..).zip(&b.corpus[s..e]) {
                    if t == BOS {
                        segment = j;
                    }
                    unsafe { context.set(j + 1, (j + 1 - segment).min(MAX) as u16) };
                }
                unsafe { stream.write(at, &b.corpus[s..e]) };
                at += e - s;
            }
            unsafe { stream.set(at, STOP) };
        });

        let (lo, hi) = (bases[first * world], bases[last * world]);
        let (stream, context) = (&b.stream[..hi], &b.context[..hi]);
        let (cells, entries_out) = (Output(self.cells_ptr()), Output(self.entries_ptr()));
        let position = |v: u64| (v & POSITION_MASK) as usize;
        b.build.lock().unwrap().index(stream, context, hi, |part| {
            let (mut found, mut tokens, mut entries) = (Vec::with_capacity(CAP), Vec::with_capacity(CAP), Vec::with_capacity(CAP));
            // Entries before `ahead` are prefetched: the context and stream lines of each one in a group of two or
            // more (a query, or a candidate), PREFETCH entries ahead of the group being resolved.
            let (n, tag, mut ahead) = (part.len(), |i: usize| part[i] >> POSITION_BITS, 0);
            for group in part.chunk_by(|a, b| a >> POSITION_BITS == b >> POSITION_BITS) {
                let end = (unsafe { group.as_ptr().offset_from(part.as_ptr()) } as usize + group.len() + PREFETCH).min(n);
                while ahead < end {
                    if (ahead > 0 && tag(ahead - 1) == tag(ahead)) || (ahead + 1 < n && tag(ahead + 1) == tag(ahead)) {
                        let g = position(part[ahead]);
                        unsafe {
                            _mm_prefetch(context.as_ptr().add(g).cast(), _MM_HINT_T0);
                            _mm_prefetch(stream.as_ptr().add(g - MAX.min(g)).cast(), _MM_HINT_T0);
                            _mm_prefetch(stream.as_ptr().add(g).cast(), _MM_HINT_T0);
                        }
                    }
                    ahead += 1;
                }
                if group.len() == 1 {
                    continue;
                }
                // group[..before]: the key's occurrences in steps before the current entry's step.
                let mut before = 0;
                for (k, &v) in group.iter().enumerate().skip(1) {
                    let g = position(v);
                    if g < lo || g >= hi {
                        continue;
                    }
                    let page = pages[g >> PAGE_BITS] as usize;
                    let run = page + (g >= bases[page + 1]) as usize;
                    let (start, offset) = owned[run];
                    while before < k && position(group[before]) < start {
                        before += 1;
                    }
                    if before == 0 {
                        continue;
                    }
                    // Match against each candidate as far back as both segments reach (MIN at least, else it is a
                    // tag collision).
                    let reach = context[g] as usize;
                    found.clear();
                    for c in group[..before].iter().rev().take(CAP).map(|&w| position(w)) {
                        let length = agree(stream, c, g, MAX.next_multiple_of(4)).min(reach).min(context[c] as usize);
                        if length < MIN {
                            continue;
                        }
                        found.push((length, stream[c] as i32));
                    }
                    if found.is_empty() {
                        continue;
                    }
                    // The cell, and the run's next entries if they fit (else the row keeps its cell only).
                    let r = offset.wrapping_add(g); // the position's row
                    entries.clear();
                    unsafe { cells.set(r, row(&found, &mut tokens, (r - self.rows[run]) as i32, &mut entries)) };
                    let used = self.counter(run).fetch_add(entries.len() as u32, Relaxed) as usize;
                    if used + entries.len() <= self.capacity[run] {
                        unsafe { entries_out.write(self.first[run] + used, &entries) };
                    }
                }
            }
        });
    }
}
