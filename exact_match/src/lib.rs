use numpy::{ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::{fs::File, io::Read, os::unix::fs::FileExt};

const BOS: u16 = 50256;
const STOP: u16 = u16::MAX;
const P: u64 = 0x9e3779b185ebca87;
const POSITION_BITS: u32 = 34;
const POSITION_MASK: u64 = (1 << POSITION_BITS) - 1;

pub fn prefault<T>(v: &mut [T]) {
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) };
    bytes.par_chunks_mut(1 << 24).for_each(|chunk| {
        for i in (0..chunk.len()).step_by(4096) {
            unsafe { std::ptr::write_volatile(chunk.as_mut_ptr().add(i), 0) };
        }
    });
}

fn open_shard(path: &str) -> PyResult<(File, usize)> {
    let mut file = File::open(&path)?;
    let mut header = [0u8; 1024];
    file.read_exact(&mut header)?;
    let word = |i| i32::from_le_bytes(header[i..i + 4].try_into().unwrap());
    let n = word(8);
    if word(0) != 20240520 || word(4) != 1 || n < 0 || file.metadata()?.len() != 1024 + n as u64 * 2
    {
        return Err(PyValueError::new_err(format!("invalid shard: {path}")));
    }
    Ok((file, n as usize))
}

fn windows(tokens: &[u16], n: usize, mut emit: impl FnMut(usize, u64)) {
    let (mut hash, mut available) = (0u64, 0usize);
    let power = P.wrapping_pow(n as u32);
    for (i, &token) in tokens.iter().enumerate() {
        if token == STOP {
            hash = 0;
            available = 0;
            continue;
        }
        if token == BOS {
            hash = 0;
            available = 0;
        }
        hash = hash.wrapping_mul(P).wrapping_add(token as u64 + 1);
        if available >= n {
            hash = hash.wrapping_sub((tokens[i - n] as u64 + 1).wrapping_mul(power));
        }
        available += 1;
        if available >= n {
            emit(i + 1, hash);
        }
    }
}

fn check_boundaries(boundaries: &[i32], n: usize) -> PyResult<()> {
    if boundaries.iter().any(|&v| v < 0 || v as usize > n)
        || boundaries.windows(2).any(|w| w[0] > w[1])
    {
        return Err(PyValueError::new_err(
            "boundaries must be sorted and within the input",
        ));
    }
    Ok(())
}

fn segments(x: &[u16], boundaries: &[i32]) -> PyResult<Vec<(usize, usize)>> {
    check_boundaries(boundaries, x.len())?;
    let mut starts = vec![0, x.len()];
    starts.extend(boundaries.iter().map(|&v| v as usize));
    starts.extend(
        x.iter()
            .enumerate()
            .filter_map(|(i, &v)| (v == BOS).then_some(i)),
    );
    starts.sort_unstable();
    starts.dedup();
    Ok(starts.windows(2).map(|w| (w[0], w[1])).collect())
}

fn top_tokens(tokens: &mut Vec<i32>) -> Vec<(i32, i32)> {
    tokens.sort_unstable();
    let mut counts: Vec<(i32, i32)> = Vec::new();
    for &token in tokens.iter() {
        if counts.last().is_some_and(|&(t, _)| t == token) {
            counts.last_mut().unwrap().1 += 1;
        } else {
            counts.push((token, 1));
        }
    }
    counts.sort_unstable_by_key(|&(token, count)| (std::cmp::Reverse(count), token));
    counts.truncate(2);
    counts
}

fn resolve(
    corpus: &[u16],
    query: &[u16],
    candidates: impl Iterator<Item = usize>,
    min: usize,
    max: usize,
    best_tokens: &mut Vec<i32>,
) -> (i32, [i32; 8]) {
    let mut best = 0;
    best_tokens.clear();
    for pos in candidates {
        if pos < min || corpus[pos - min..pos] != query[query.len() - min..] {
            continue;
        }
        let maximum = max.min(query.len()).min(pos);
        let mut length = min;
        while length + 8 <= maximum
            && corpus[pos - length - 8..pos - length]
                == query[query.len() - length - 8..query.len() - length]
        {
            length += 8;
        }
        while length < maximum && corpus[pos - length - 1] == query[query.len() - length - 1] {
            length += 1;
        }
        if length > best {
            best = length;
            best_tokens.clear();
        }
        if length == best {
            best_tokens.push(corpus[pos] as i32);
        }
    }
    let mut next = [-1; 8];
    for (i, (token, count)) in top_tokens(best_tokens).into_iter().enumerate() {
        next[i] = token;
        next[4 + i] = count;
    }
    (best as i32, next)
}

#[pyclass]
struct Cache {
    min: usize,
    max: usize,
    corpus: Vec<u16>,
    online: Option<Online>,
    offline: Option<Index>,
    shards: Vec<String>,
    ready: bool,
}

impl Cache {
    fn lookup(&self, x: &[u16], bounds: &[(usize, usize)]) -> (Vec<i32>, Array2<i32>) {
        let width = 8;
        let results: Vec<_> = bounds
            .par_iter()
            .map(|&(start, end)| {
                let mut length = vec![0; end - start];
                let mut next = vec![[-1; 8]; end - start];
                let mut best = Vec::with_capacity(4);
                windows(&x[start..end], self.min, |offset, hash| {
                    let query = &x[start..start + offset];
                    let result = resolve(
                        &self.corpus,
                        query,
                        self.offline.as_ref().unwrap().candidates(hash),
                        self.min,
                        self.max,
                        &mut best,
                    );
                    length[offset - 1] = result.0;
                    next[offset - 1] = result.1;
                });
                (start, length, next)
            })
            .collect();
        let mut length = vec![0; x.len()];
        let mut next = Array2::from_elem((x.len(), width), -1);
        for (start, lengths, ids) in results {
            length[start..start + lengths.len()].copy_from_slice(&lengths);
            for (i, row) in ids.iter().enumerate() {
                next.as_slice_mut().unwrap()[(start + i) * width..(start + i + 1) * width]
                    .copy_from_slice(&row[..width]);
            }
        }
        (length, next)
    }
}

#[pymethods]
impl Cache {
    #[new]
    #[pyo3(signature = (min_context=8, max_context=512))]
    fn new(min_context: usize, max_context: usize) -> PyResult<Self> {
        if min_context == 0 || min_context > max_context || max_context > 512 {
            return Err(PyValueError::new_err(
                "require 1 <= min_context <= max_context <= 512",
            ));
        }
        Ok(Self {
            min: min_context,
            max: max_context,
            corpus: vec![],
            online: Some(
                Online::new(min_context, max_context).map_err(PyValueError::new_err)?,
            ),
            offline: None,
            shards: Vec::new(),
            ready: true,
        })
    }

    fn insert(
        &mut self,
        py: Python<'_>,
        inputs: PyReadonlyArray1<i32>,
        targets: PyReadonlyArray1<i32>,
        boundaries: PyReadonlyArray1<i32>,
    ) -> PyResult<()> {
        if self.offline.is_some() {
            return Err(PyValueError::new_err("offline cache is frozen"));
        }
        let x = inputs.as_slice()?.to_vec();
        let y = targets.as_slice()?.to_vec();
        if x.iter().chain(&y).any(|&v| !(0..=BOS as i32).contains(&v)) {
            return Err(PyValueError::new_err("invalid GPT-2 token"));
        }
        let boundaries = boundaries.as_slice()?.to_vec();
        check_boundaries(&boundaries, x.len())?;
        py.detach(|| self.online.as_mut().unwrap().insert(&x, &y, &boundaries))
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    fn reserve(&mut self, py: Python<'_>, tokens: usize) -> PyResult<()> {
        let online = self
            .online
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("offline cache"))?;
        py.detach(|| online.reserve(tokens))
            .map_err(PyValueError::new_err)
    }

    fn release(&mut self, py: Python<'_>) {
        self.ready = false;
        let storage = (std::mem::take(&mut self.corpus), self.offline.take(), self.online.take());
        py.detach(|| drop(storage));
    }

    #[staticmethod]
    #[pyo3(signature = (files, min_context=8, max_context=512, token_capacity=None))]
    fn allocate_offline(
        py: Python<'_>,
        files: Vec<String>,
        min_context: usize,
        max_context: usize,
        token_capacity: Option<usize>,
    ) -> PyResult<Self> {
        let mut cache = Self::new(min_context, max_context)?;
        cache.online = None;
        cache.ready = false;
        if files.is_empty() {
            return Err(PyValueError::new_err("no training shards"));
        }
        py.detach(|| -> PyResult<()> {
            let tokens = match token_capacity {
                Some(tokens) => tokens,
                None => {
                    let mut tokens = 0usize;
                    for path in &files {
                        tokens += open_shard(path)?.1;
                    }
                    tokens
                }
            };
            let total = tokens
                .checked_add(files.len())
                .and_then(|n| n.checked_add(1))
                .filter(|&n| n as u64 <= POSITION_MASK)
                .ok_or_else(|| PyValueError::new_err("corpus exceeds 34-bit positions"))?;
            cache.corpus = vec![0; total];
            prefault(&mut cache.corpus);
            cache.offline = Some(Index::allocate(total));
            cache.shards = files;
            Ok(())
        })?;
        Ok(cache)
    }

    fn build(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.ready || self.offline.is_none() {
            return Err(PyValueError::new_err(
                "build requires an unbuilt offline cache",
            ));
        }
        py.detach(|| -> PyResult<()> {
            let shards = self
                .shards
                .iter()
                .map(|path| open_shard(path))
                .collect::<PyResult<Vec<_>>>()?;
            let total = 1 + shards.iter().map(|(_, n)| n + 1).sum::<usize>();
            if total > self.corpus.len() {
                return Err(PyValueError::new_err(
                    "preallocated offline token capacity exceeded",
                ));
            }

            self.corpus.truncate(total);
            let mut remaining = self.corpus.as_mut_slice();

            let mut chunks = Vec::new();
            for (file, n) in &shards {
                let (chunk, rest) = remaining.split_at_mut(*n + 1);
                chunk[0] = STOP;
                chunks.push((file, &mut chunk[1..]));
                remaining = rest;
            }
            remaining[0] = STOP;
            chunks
                .into_par_iter()
                .try_for_each(|(file, tokens)| -> PyResult<()> {
                    const CHUNK: usize = 2 << 20;
                    tokens.par_chunks_mut(CHUNK).enumerate().try_for_each(
                        |(i, chunk)| -> PyResult<()> {

                            let bytes = unsafe {
                                std::slice::from_raw_parts_mut(
                                    chunk.as_mut_ptr().cast::<u8>(),
                                    chunk.len() * 2,
                                )
                            };
                            file.read_exact_at(bytes, 1024 + (i * CHUNK * 2) as u64)?;
                            for v in chunk {
                                *v = u16::from_le(*v);
                                if *v > BOS {
                                    return Err(PyValueError::new_err("invalid GPT-2 token"));
                                }
                            }
                            Ok(())
                        },
                    )
                })?;
            self.offline.as_mut().unwrap().build(&self.corpus, self.min);
            self.ready = true;
            Ok(())
        })?;
        Ok(())
    }

    fn query<'py>(
        &self,
        py: Python<'py>,
        inputs: PyReadonlyArray1<i32>,
        boundaries: PyReadonlyArray1<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray2<i32>>)> {
        if !self.ready {
            return Err(PyValueError::new_err("offline cache has not been built"));
        }
        let x = inputs.as_slice()?.to_vec();
        let boundaries = boundaries.as_slice()?.to_vec();
        check_boundaries(&boundaries, x.len())?;
        if x.iter().any(|&v| !(0..=BOS as i32).contains(&v)) {
            return Err(PyValueError::new_err("invalid GPT-2 token"));
        }
        let (length, next) = py.detach(|| {
            if let Some(cache) = &self.online {
                let (hits, candidates) = cache.query(&x, &boundaries);
                let mut length = vec![0; x.len()];
                let mut next = Array2::from_elem((x.len(), 4), -1);
                for hit in hits {
                    length[hit.index] = hit.length as i32;
                }
                for (row, ids) in candidates {
                    next.as_slice_mut().unwrap()[row * 4..row * 4 + ids.len()]
                        .copy_from_slice(&ids);
                }
                (length, next)
            } else {
                let x: Vec<u16> = x.into_iter().map(|v| v as u16).collect();
                let bounds = segments(&x, &boundaries).unwrap();
                self.lookup(&x, &bounds)
            }
        });
        Ok((length.into_pyarray(py), next.into_pyarray(py)))
    }
}

#[pymodule]
fn exact_match(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Cache>()
}

const BITS: usize = 12;
const PARTITIONS: usize = 1 << BITS;
const BURST: usize = 32;
const TAG_MASK: u64 = (1 << 30) - 1;

fn key(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

#[derive(Clone, Copy)]
struct Output(*mut u64);
unsafe impl Sync for Output {}
unsafe impl Send for Output {}
impl Output {
    unsafe fn write(self, offset: usize, values: &[u64]) {
        std::ptr::copy_nonoverlapping(values.as_ptr(), self.0.add(offset), values.len());
    }
}

pub struct Index {
    entries: Vec<u64>,
    offsets: Vec<usize>,
    lengths: Vec<usize>,
    counts: Vec<Vec<usize>>,
    bases: Vec<Vec<usize>>,
    buffers: Vec<Vec<u64>>,
    used: Vec<Vec<usize>>,
    cursors: Vec<Vec<usize>>,
    scratch: Vec<Vec<u64>>,
}

impl Index {
    pub fn allocate(tokens: usize) -> Self {
        let threads = rayon::current_num_threads();

        let mut entries = vec![0; tokens + threads * PARTITIONS * (BURST - 1)];
        let mut scratch = vec![vec![0; 2 * tokens.div_ceil(PARTITIONS) + 4096]; threads];
        prefault(&mut entries);
        scratch.iter_mut().for_each(|v| prefault(v));
        Self {
            entries,
            offsets: vec![0; PARTITIONS + 1],
            lengths: vec![0; PARTITIONS],
            counts: vec![vec![0; PARTITIONS]; threads],
            bases: vec![vec![0; PARTITIONS]; threads],
            buffers: vec![vec![0; PARTITIONS * BURST]; threads],
            used: vec![vec![0; PARTITIONS]; threads],
            cursors: vec![vec![0; PARTITIONS]; threads],

            scratch,
        }
    }

    pub fn build(&mut self, corpus: &[u16], min: usize) {
        let threads = self.counts.len();
        let stream = |t: usize, emit: &mut dyn FnMut(u64, usize)| {
            let start = corpus.len() * t / threads;
            let end = corpus.len() * (t + 1) / threads;
            let begin = start.saturating_sub(min - 1);
            windows(&corpus[begin..end], min, |p, h| {
                let pos = begin + p;
                if pos > start && pos < corpus.len() && corpus[pos] != STOP {
                    emit(key(h), pos);
                }
            });
        };
        let Self {
            entries,
            offsets,
            lengths,
            counts,
            bases,
            buffers,
            used,
            cursors,
            scratch,
        } = self;
        counts.par_iter_mut().enumerate().for_each(|(t, counts)| {
            stream(t, &mut |h, _| counts[(h >> (64 - BITS)) as usize] += 1);
        });
        for p in 0..PARTITIONS {
            let mut end = offsets[p];
            for t in 0..threads {
                bases[t][p] = end;
                end += (counts[t][p] + BURST - 1) & !(BURST - 1);
                lengths[p] += counts[t][p];
            }
            offsets[p + 1] = end;
        }
        assert!(offsets[PARTITIONS] <= entries.len());
        let output = Output(entries.as_mut_ptr());
        buffers
            .par_iter_mut()
            .zip(used.par_iter_mut())
            .zip(cursors.par_iter_mut())
            .enumerate()
            .for_each(|(t, ((buffer, used), at))| {
                at.copy_from_slice(&bases[t]);
                stream(t, &mut |h, pos| {
                    let p = (h >> (64 - BITS)) as usize;
                    buffer[p * BURST + used[p]] =
                        (((h >> (64 - BITS - 30)) & TAG_MASK) << POSITION_BITS) | pos as u64;
                    used[p] += 1;
                    if used[p] == BURST {
                        unsafe {
                            output.write(at[p], &buffer[p * BURST..(p + 1) * BURST]);
                        }
                        at[p] += BURST;
                        used[p] = 0;
                    }
                });
                for p in 0..PARTITIONS {
                    unsafe {
                        output.write(at[p], &buffer[p * BURST..p * BURST + used[p]]);
                    }
                }
            });

        let sizes: Vec<_> = offsets.windows(2).map(|w| w[1] - w[0]).collect();
        let mut remaining = &mut entries[..offsets[PARTITIONS]];
        let mut parts = Vec::with_capacity(PARTITIONS);
        for size in sizes {
            let (part, rest) = remaining.split_at_mut(size);
            parts.push(part);
            remaining = rest;
        }
        let group = PARTITIONS.div_ceil(threads);
        parts
            .par_chunks_mut(group)
            .zip(scratch.par_iter_mut())
            .enumerate()
            .for_each(|(g, (parts, scratch))| {
                for (i, part) in parts.iter_mut().enumerate() {
                    let p = g * group + i;
                    let mut at = 0;
                    for t in 0..threads {
                        let start = bases[t][p] - offsets[p];
                        let n = counts[t][p];
                        part.copy_within(start..start + n, at);
                        at += n;
                    }
                    let part = &mut part[..lengths[p]];
                    if part.len() > scratch.len() {
                        part.sort_unstable_by_key(|x| x >> POSITION_BITS);
                        continue;
                    }
                    let tmp = &mut scratch[..part.len()];
                    for shift in [POSITION_BITS, POSITION_BITS + 10, POSITION_BITS + 20] {
                        let mut offsets = [0usize; 1024];
                        for &x in part.iter() {
                            offsets[((x >> shift) & 1023) as usize] += 1;
                        }
                        let mut sum = 0;
                        for offset in &mut offsets {
                            let n = *offset;
                            *offset = sum;
                            sum += n;
                        }
                        for &x in part.iter() {
                            let d = ((x >> shift) & 1023) as usize;
                            tmp[offsets[d]] = x;
                            offsets[d] += 1;
                        }
                        part.copy_from_slice(tmp);
                    }
                }
            });
    }

    pub fn candidates(&self, hash: u64) -> impl Iterator<Item = usize> + '_ {
        let h = key(hash);
        let p = (h >> (64 - BITS)) as usize;
        let tag = ((h >> (64 - BITS - 30)) & TAG_MASK) << POSITION_BITS;
        let part = &self.entries[self.offsets[p]..self.offsets[p] + self.lengths[p]];
        let lo = part.partition_point(|&v| v < tag);
        part[lo..]
            .iter()
            .take_while(move |&&v| v <= tag | POSITION_MASK)
            .map(|&v| (v & POSITION_MASK) as usize)
    }
}

const PREFETCH_BATCH: usize = 16;

const BASE: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
fn finalize(mut h: u64) -> u64 {

    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h | (h == 0) as u64
}

#[inline]
fn tok(t: i32) -> u64 {

    (t as u32 as u64).wrapping_add(0x5851_F42D_4C95_7F2D)
}

#[inline]
fn window_hashes(
    tokens: &[i32],
    start: usize,
    end: usize,
    k: usize,
    pow_k: u64,
    mut emit: impl FnMut(usize, u64),
) {
    if start + k > end {
        return;
    }
    let mut h: u64 = 0;
    for &t in &tokens[start..start + k] {
        h = h.wrapping_mul(BASE).wrapping_add(tok(t));
    }
    emit(start + k, finalize(h));
    for local in (start + k + 1)..=end {
        h = h
            .wrapping_mul(BASE)
            .wrapping_sub(tok(tokens[local - k - 1]).wrapping_mul(pow_k))
            .wrapping_add(tok(tokens[local - 1]));
        emit(local, finalize(h));
    }
}

#[inline]
fn shard_of(hash: u64) -> usize {
    ((hash >> 40) & 63) as usize
}

#[inline(always)]
fn prefetch<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

struct Shard {

    ctrl: Vec<u8>,
    keys: Vec<u64>,

    slots: Vec<u32>,
    mask: usize,
    len: usize,
    fixed: bool,
}

#[inline(always)]
fn tag_of(h: u64) -> u8 {
    0x80 | (h >> 57) as u8
}

impl Shard {
    const MAX_LOAD_PERMILLE: usize = 800;

    fn with_capacity(keys: usize) -> Self {
        let buckets = (keys * 1000 / Self::MAX_LOAD_PERMILLE + 1)
            .next_power_of_two()
            .max(64);
        let (mut ctrl, mut keys, mut slots) = (vec![0; buckets], vec![0; buckets], vec![0; buckets * 4]);
        prefault(&mut ctrl);
        prefault(&mut keys);
        prefault(&mut slots);
        Self {
            ctrl,
            keys,
            slots,
            mask: buckets - 1,
            len: 0,
            fixed: false,
        }
    }

    #[inline(always)]
    fn bucket(&self, h: u64) -> usize {
        (h as usize) & self.mask
    }

    #[inline(always)]
    fn prefetch(&self, h: u64) {
        prefetch(unsafe { self.ctrl.as_ptr().add(self.bucket(h)) });
    }

    #[inline(always)]
    fn prefetch_entry(&self, b: usize) {
        prefetch(unsafe { self.keys.as_ptr().add(b) });
        prefetch(unsafe { self.slots.as_ptr().add(b * 4) });
    }

    #[inline(always)]
    fn find(&self, h: u64) -> Option<usize> {
        let tag = tag_of(h);
        let mut b = self.bucket(h);
        loop {
            let c = self.ctrl[b];
            if c == tag && self.keys[b] == h {
                return Some(b);
            }
            if c == 0 {
                return None;
            }
            b = (b + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn positions(&self, b: usize) -> &[u32] {
        &self.slots[b * 4..(b + 1) * 4]
    }

    fn grow_to(&mut self, keys: usize) {
        let mut fresh = Self::with_capacity(keys.max(self.len * 2));
        for (b, &c) in self.ctrl.iter().enumerate() {
            if c != 0 {
                let k = self.keys[b];
                let mut nb = fresh.bucket(k);
                while fresh.ctrl[nb] != 0 {
                    nb = (nb + 1) & fresh.mask;
                }
                fresh.ctrl[nb] = c;
                fresh.keys[nb] = k;
                let (src, dst) = (b * 4, nb * 4);
                fresh.slots[dst..dst + 4].copy_from_slice(&self.slots[src..src + 4]);
            }
        }
        fresh.len = self.len;
        *self = fresh;
    }

    #[inline]
    pub fn insert(&mut self, h: u64, position: u32) -> Result<(), String> {
        if (self.len + 1) * 1000 > self.ctrl.len() * Self::MAX_LOAD_PERMILLE {
            if self.fixed {
                return Err("preallocated online hash shard is full".into());
            }
            self.grow_to(self.len * 2);
        }
        let tag = tag_of(h);
        let mut b = self.bucket(h);
        loop {
            let c = self.ctrl[b];
            if c == tag && self.keys[b] == h {
                break;
            }
            if c == 0 {
                self.ctrl[b] = tag;
                self.keys[b] = h;
                self.len += 1;
                break;
            }
            b = (b + 1) & self.mask;
        }
        let max_occ = 4;
        let slots = &mut self.slots[b * max_occ..(b + 1) * max_occ];
        match slots.iter().position(|&x| x == 0) {
            Some(free) => slots[free] = position,
            None => {
                slots.copy_within(1.., 0);
                slots[max_occ - 1] = position;
            }
        }
        Ok(())
    }
}

pub struct Online {
    min_context: usize,
    max_context: usize,
    pow_k: u64,
    tokens: Vec<i32>,
    context_lengths: Vec<u16>,
    shards: Vec<Shard>,
    pool: rayon::ThreadPool,
    capacity: Option<usize>,
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub index: usize,
    pub length: u16,
}

impl Online {
    pub fn new(min_context: usize, max_context: usize) -> Result<Self, String> {
        let threads = std::env::var("RETRIEVAL_ONLINE_THREADS").or_else(|_| std::env::var("RAYON_NUM_THREADS"))
            .unwrap_or_else(|_| "4".into())
            .parse::<usize>()
            .map_err(|e| e.to_string())?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            min_context,
            max_context,
            pow_k: BASE.wrapping_pow(min_context as u32),
            tokens: Vec::new(),
            context_lengths: Vec::new(),
            shards: (0..64).map(|_| Shard::with_capacity(0)).collect(),
            pool,
            capacity: None,
        })
    }

    pub fn reserve(&mut self, tokens: usize) -> Result<(), String> {
        if !self.tokens.is_empty() || self.capacity.is_some() || tokens > u32::MAX as usize {
            return Err("reserve requires an empty cache and at most 2^32-1 tokens".into());
        }

        self.tokens = vec![0; tokens];
        self.context_lengths = vec![0; tokens];
        prefault(&mut self.tokens);
        prefault(&mut self.context_lengths);
        self.tokens.clear();
        self.context_lengths.clear();

        let keys = tokens.div_ceil(64) * 5 / 4 + 1024;
        self.shards = (0..64)
            .map(|_| {
                let mut shard = Shard::with_capacity(keys);
                shard.fixed = true;
                shard
            })
            .collect();
        self.capacity = Some(tokens);
        Ok(())
    }

    fn segments(&self, inputs: &[i32], cum_seqlens: &[i32]) -> Vec<(usize, usize)> {
        let n = inputs.len();
        if n == 0 {
            return Vec::new();
        }
        let mut starts: Vec<usize> = Vec::with_capacity(cum_seqlens.len() + 16);
        starts.push(0);
        for &c in cum_seqlens {
            if c >= 0 && (c as usize) < n {
                starts.push(c as usize);
            }
        }
        {
            starts.extend(
                inputs
                    .iter()
                    .enumerate()
                    .filter(|(_, &t)| t == 50256)
                    .map(|(i, _)| i),
            );
        }
        starts.sort_unstable();
        starts.dedup();
        let mut out = Vec::with_capacity(starts.len());
        for (i, &s) in starts.iter().enumerate() {
            let e = if i + 1 < starts.len() {
                starts[i + 1]
            } else {
                n
            };
            out.push((s, e));
        }
        out
    }

    pub fn insert(
        &mut self,
        inputs: &[i32],
        targets: &[i32],
        cum_seqlens: &[i32],
    ) -> Result<(), String> {
        let n = inputs.len();
        if targets.len() != n || (n > 0 && inputs[1..] != targets[..n - 1]) {
            return Err("expected contiguous next-token inputs and targets".into());
        }
        if n == 0 {
            return Ok(());
        }
        let base = self.tokens.len();
        let stop = base + n + 1;
        if stop > u32::MAX as usize {
            return Err("exact-match cache exceeds 2^32 tokens".into());
        }
        if self.capacity.is_some_and(|capacity| stop > capacity) {
            return Err("preallocated online token capacity exceeded".into());
        }
        self.tokens.extend_from_slice(inputs);
        self.tokens.push(targets[n - 1]);
        self.context_lengths.resize(stop, 0);

        let min = self.min_context;
        let max_context = self.max_context;
        let pow_k = self.pow_k;
        let segments = self.segments(inputs, cum_seqlens);
        let tokens = &self.tokens;

        let (items, contexts): (Vec<Vec<(u64, u32)>>, Vec<Vec<(u32, u16)>>) =
            self.pool.install(|| {
                segments
                    .par_iter()
                    .map(|&(start, end)| {
                        let mut items = Vec::with_capacity((end + 1).saturating_sub(start + min));
                        let mut contexts = Vec::with_capacity(items.capacity());
                        window_hashes(
                            tokens,
                            base + start,
                            base + end,
                            min,
                            pow_k,
                            |position, h| {
                                let local = position - base;
                                contexts.push((
                                    position as u32,
                                    (local - start).min(max_context) as u16,
                                ));
                                items.push((h, position as u32));
                            },
                        );
                        (items, contexts)
                    })
                    .unzip()
            });
        for (position, length) in contexts.iter().flatten() {
            self.context_lengths[*position as usize] = *length;
        }
        let items: Vec<(u64, u32)> = items.concat();

        let nshards = self.shards.len();
        let group = (nshards / self.pool.current_num_threads()).max(1);
        let shards = &mut self.shards;
        self.pool.install(|| {
            shards.par_chunks_mut(group).enumerate().try_for_each(
                |(g, chunk)| -> Result<(), String> {
                    let first = g * group;
                    let last = first + chunk.len();
                    let mine: Vec<(u64, u32)> = items
                        .iter()
                        .copied()
                        .filter(|&(h, _)| {
                            let s = shard_of(h);
                            s >= first && s < last
                        })
                        .collect();
                    for batch in mine.chunks(PREFETCH_BATCH) {
                        for &(h, _) in batch {
                            let shard = &chunk[shard_of(h) - first];
                            shard.prefetch(h);
                            shard.prefetch_entry(shard.bucket(h));
                        }
                        for &(h, position) in batch {
                            chunk[shard_of(h) - first].insert(h, position)?;
                        }
                    }
                    Ok(())
                },
            )
        })?;
        Ok(())
    }

    #[inline]
    fn match_at(
        &self,
        inputs: &[i32],
        start: usize,
        local: usize,
        shard: &Shard,
        b: usize,
        best: &mut Vec<i32>,
    ) -> usize {
        best.clear();
        let min = self.min_context;
        let key = &inputs[local - min..local];
        let tokens = &self.tokens;
        let mut best_len = 0usize;
        for &cand in shard.positions(b) {
            if cand == 0 {
                break;
            }
            let c = cand as usize;
            if tokens[c - min..c] != *key {
                continue;
            }
            let maximum = self
                .max_context
                .min(local - start)
                .min(self.context_lengths[c] as usize);
            let mut len = min;
            while len + 8 <= maximum
                && tokens[c - len - 8..c - len] == inputs[local - len - 8..local - len]
            {
                len += 8;
            }
            while len < maximum && inputs[local - len - 1] == tokens[c - len - 1] {
                len += 1;
            }
            if len > best_len {
                best_len = len;
                best.clear();
            }
            if len == best_len {
                best.push(tokens[c]);
            }
        }
        best_len
    }

    pub fn query(&self, inputs: &[i32], cum_seqlens: &[i32]) -> (Vec<Hit>, Vec<(usize, Vec<i32>)>) {
        let min = self.min_context;
        let pow_k = self.pow_k;
        let segments = self.segments(inputs, cum_seqlens);
        let work: Vec<(usize, usize, u64)> = self.pool.install(|| {
            segments
                .par_iter()
                .map(|&(start, end)| {
                    let mut items = Vec::with_capacity((end + 1).saturating_sub(start + min));
                    window_hashes(inputs, start, end, min, pow_k, |local, h| {
                        items.push((start, local, h))
                    });
                    items
                })
                .collect::<Vec<_>>()
                .concat()
        });
        let min = self.min_context;
        self.pool.install(|| {
            work.par_chunks(1024)
                .map(|chunk| {
                    let mut best = Vec::with_capacity(4);
                    let mut hits = Vec::new();
                    let mut cands = Vec::new();
                    let mut found: Vec<(usize, usize, usize, usize)> =
                        Vec::with_capacity(PREFETCH_BATCH);
                    for batch in chunk.chunks(PREFETCH_BATCH) {

                        for &(_, _, h) in batch {
                            self.shards[shard_of(h)].prefetch(h);
                        }

                        found.clear();
                        for &(start, local, h) in batch {
                            let s = shard_of(h);
                            if let Some(b) = self.shards[s].find(h) {
                                for &cand in self.shards[s].positions(b) {
                                    if cand == 0 {
                                        break;
                                    }
                                    let c = cand as usize;
                                    prefetch(unsafe { self.tokens.as_ptr().add(c - min) });
                                    prefetch(unsafe { self.tokens.as_ptr().add(c) });
                                    prefetch(unsafe { self.context_lengths.as_ptr().add(c) });
                                }
                                found.push((start, local, s, b));
                            }
                        }

                        for &(start, local, s, b) in &found {
                            let len =
                                self.match_at(inputs, start, local, &self.shards[s], b, &mut best);
                            if best.is_empty() {
                                continue;
                            }
                            hits.push(Hit {
                                index: local - 1,
                                length: len as u16,
                            });
                            best.reverse();
                            cands.push((local - 1, best.clone()));
                        }
                    }
                    (hits, cands)
                })
                .reduce(
                    || (Vec::new(), Vec::new()),
                    |mut a, b| {
                        a.0.extend(b.0);
                        a.1.extend(b.1);
                        a
                    },
                )
        })
    }
}
