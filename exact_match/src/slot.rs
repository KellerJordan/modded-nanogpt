//! SlotIndex: the validation index, approximate and corpus-free, with lookups of at most two bucket reads.
//! key 3/2: the same index keyed on the last 3 or 2 tokens, 2^12 buckets per partition.
//!
//! One u64 entry per training position with a MIN-gram in its shard:
//!
//!   next token (16 bits) | fingerprint of each level (BITS[k] each) | key check (KBITS) ,
//!
//! where the MIN-gram key's top PBITS pick a partition and the top NBITS of the check the entry's home bucket b1 in
//! that partition; its second bucket b2 is another hash of the whole check. Level L's fingerprint hashes only the
//! tokens the shorter level does not cover (it is compared only once those agree).
//!
//! Build (on the clock, rank 0 alone, on its own pinned pool, no GIL):
//!   1. scan: pread 256K-token pieces, hash 4096 positions at a time, and scatter every entry through a per-thread
//!      write-combining burst of one cache line per partition into 2048-entry blocks of a private huge-page arena,
//!      with non-temporal stores (no count pass, no sort).
//!   2. finish: per partition (largest first, handed out dynamically), place its entries into a scratch copy of the
//!      partition's slice of the table (2^NBITS buckets of SLOTS entries): into b1 while it has room, else b2, else
//!      dropped (a hot key keeps its first 2 * SLOTS entries). The slice is then streamed to the table, a file in
//!      /dev/shm, and this process's mappings of it dropped, so no rank keeps big page tables on the clock.
//!
//! Lookup (validation tokens only, after training, every rank its own positions): a position is hashed the same way
//! and bucket b1 is read in place (mmap) and, only if b1 is full, b2. Among the entries whose check equals the query's, a
//! candidate's level is the last of the unbroken ascending run of agreeing fingerprints (levels beyond the query's
//! segment start never agree), and the row is made of the candidates at the best level.
use crate::{fence, mmap, open_shard, row, stream, touch, Map, Output, BOS, LEVELS, MIN, STOP};
use numpy::{PyReadonlyArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::arch::x86_64::{_mm512_loadu_si512, _mm512_maskz_loadu_epi64, _mm512_stream_si512, _mm_prefetch, _MM_HINT_T0};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{LazyLock, Mutex};

const BITS: [u32; 2] = [12, 6]; // fingerprint bits per level
const SHIFTS: [u32; 2] = [16, 28]; // where each level's fingerprint sits in an entry
const CHECK_SHIFT: u32 = 34;
const KBITS: u32 = 64 - CHECK_SHIFT; // key check bits
const PBITS: u32 = 14; // partitions: 2^14
const SLOTS: usize = 48; // entries per bucket: 6 cache lines
const BACK: usize = LEVELS[1]; // tokens before a context end that its hash reads
const STEP: usize = 1 << 12; // positions hashed at once
const PIECE: usize = 1 << 18; // context ends per piece of the scan (one pread)
const BURST: usize = 8; // entries per write-combining burst: one cache line
const BLOCK: usize = 2048; // entries per arena block, a multiple of BURST
const NONE: u32 = u32::MAX;
const PF: usize = 16; // entries ahead whose destination line the placement prefetches
const K0: u64 = 0x9e3779b97f4a7c15;
const K1: u64 = 0xc2b2ae3d27d4eb4f;
const K2: u64 = 0xff51afd7ed558ccd;

static AVX512: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
});

const fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// The multiplier of word w of level k.
const fn mul(k: u64, w: u64) -> u64 {
    splitmix((k + 1) << 32 | w) | 1
}

/// Per level, the words its fingerprint hashes as (offset o, multiplier): word q(j - o) holds the 4 tokens before
/// j - o, so level 10 covers the tokens 7-10 before the context end j and level 20 the tokens 11-20.
const LEVEL10: [(usize, u64); 1] = [(6, mul(0, 0))];
const LEVEL20: [(usize, u64); 3] = [(10, mul(1, 0)), (14, mul(1, 1)), (16, mul(1, 2))];
const FINALS: [u64; 2] = [splitmix(0xabcdef) | 1, splitmix(0xabcdef ^ 1) | 1];

/// One term of an NH hash: a 32x32->64 multiply.
#[inline(always)]
fn nh(q: u64, m: u64) -> u64 {
    ((q as u32).wrapping_add(m as u32) as u64).wrapping_mul(((q >> 32) as u32).wrapping_add((m >> 32) as u32) as u64)
}

/// A level's fingerprint (`bits` wide, at `shift`) of the context ending at q index j.
#[inline(always)]
fn fingerprint(q: *const u64, j: usize, words: &[(usize, u64)], fin: u64, bits: u32, shift: u32) -> u64 {
    let acc = words.iter().fold(0u64, |acc, &(o, m)| acc.wrapping_add(nh(unsafe { *q.add(j - o) }, m)));
    let x = ((acc ^ (acc >> 32)) as u32 as u64).wrapping_mul(fin as u32 as u64);
    ((x << 32) >> (64 - bits)) << shift
}

#[inline(always)]
fn check(e: u64) -> u64 {
    (e >> CHECK_SHIFT) & ((1 << KBITS) - 1)
}

fn nbits(key: usize) -> u32 { // key: 2^nbits buckets per partition (6: a 103 GB table, 3/2: 26 GB)
    if key == MIN { 14 } else { 12 }
}

#[inline(always)]
fn b1(check: u64, nbits: u32) -> usize {
    (check >> (KBITS - nbits)) as usize
}

#[inline(always)]
fn b2(check: u64, nbits: u32) -> usize {
    (check.wrapping_mul(K0) >> (64 - nbits)) as usize
}

/// pread exactly `buf` at byte `offset` of `file`.
fn pread<T>(file: &File, buf: &mut [T], offset: usize) -> std::io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, std::mem::size_of_val(buf)) };
    file.read_exact_at(bytes, offset as u64)
}

/// Scratch of one hashing step.
struct Hashes {
    q: Vec<u64>,
    e: Vec<u64>,
    part: Vec<u16>,
}

impl Hashes {
    fn new() -> Self {
        Self { q: vec![0; STEP + BACK + 1], e: vec![0; STEP], part: vec![0; STEP] }
    }

    /// The entries (with the next token if `next`) and partitions of the n contexts ending at t[j0..j0 + n]
    /// (exclusive ends): context r ends at t[j0 + r], which is its next token. Needs j0 >= BACK.
    fn hash(&mut self, t: &[u16], j0: usize, n: usize, next: bool, key: usize) {
        if *AVX512 {
            unsafe { self.hash_avx512(t, j0, n, next, key) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { self.hash_avx2(t, j0, n, next, key) }
        } else {
            self.hash_impl(t, j0, n, next, key)
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
    unsafe fn hash_avx512(&mut self, t: &[u16], j0: usize, n: usize, next: bool, key: usize) {
        self.hash_impl(t, j0, n, next, key)
    }

    #[target_feature(enable = "avx2,bmi2")]
    unsafe fn hash_avx2(&mut self, t: &[u16], j0: usize, n: usize, next: bool, key: usize) {
        self.hash_impl(t, j0, n, next, key)
    }

    /// Array passes that vectorise: q[i] holds the 4 tokens before t[j0 - BACK + i] as one word; then per position
    /// the key (from its last 6 tokens, words j and j - 2), its partition and check, and the level fingerprints.
    #[inline(always)]
    fn hash_impl(&mut self, t: &[u16], j0: usize, n: usize, next: bool, key: usize) {
        match key { // key: a vectorised loop per key
            2 => self.hash_key::<2>(t, j0, n, next),
            3 => self.hash_key::<3>(t, j0, n, next),
            _ => self.hash_key::<MIN>(t, j0, n, next),
        }
    }

    #[inline(always)]
    fn hash_key<const KEY: usize>(&mut self, t: &[u16], j0: usize, n: usize, next: bool) {
        let base = j0 - BACK;
        let (q, e, part) = (&mut self.q[..], &mut self.e[..], &mut self.part[..]);
        assert!(t.len() >= base + n + BACK - 1 + next as usize && q.len() >= n + BACK && e.len() >= n && part.len() >= n);
        let tp = t.as_ptr();
        for i in 4..n + BACK {
            unsafe {
                let b = tp.add(base + i - 4);
                q[i] = *b as u64 | (*b.add(1) as u64) << 16 | (*b.add(2) as u64) << 32 | (*b.add(3) as u64) << 48;
            }
        }
        let qp = q.as_ptr();
        let short = [(KEY, mul(0, 0)), (6, mul(0, 1))]; // key 3/2: level 10 covers the tokens key + 1 to 10
        for r in 0..n {
            let j = BACK + r;
            let mut h = match KEY {
                2 => nh(unsafe { *qp.add(j) } >> 32, K0),
                3 => nh(unsafe { *qp.add(j) } >> 16, K0),
                _ => nh(unsafe { *qp.add(j) }, K0).wrapping_add(nh(unsafe { *qp.add(j - 2) }, K1)),
            };
            h ^= h >> 29;
            h = h.wrapping_mul(K2);
            h ^= h >> 32;
            part[r] = (h >> (64 - PBITS)) as u16;
            e[r] = ((h << PBITS) >> (64 - KBITS)) << CHECK_SHIFT
                | fingerprint(qp, j, if KEY == MIN { &LEVEL10 } else { &short }, FINALS[0], BITS[0], SHIFTS[0])
                | fingerprint(qp, j, &LEVEL20, FINALS[1], BITS[1], SHIFTS[1]);
        }
        if next {
            for r in 0..n {
                e[r] |= t[j0 + r] as u64;
            }
        }
    }
}

/// Stream one 64-byte aligned cache line (8 entries) with a single non-temporal store.
#[target_feature(enable = "avx512f")]
unsafe fn stream_line_avx512(dst: *mut u64, src: *const u64) {
    _mm512_stream_si512(dst.cast(), _mm512_loadu_si512(src.cast()));
}

/// Stream a slice's buckets to `dst`, each bucket's slots beyond its fill count as zeros (masked loads, so the
/// scratch is never cleared).
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
unsafe fn out_avx512(slice: &[u64], fill: &[u8], dst: *mut u64) {
    let mut d = dst;
    for (j, &f) in fill.iter().enumerate() {
        let src = slice.as_ptr().add(j * SLOTS);
        for l in (0..SLOTS).step_by(8) {
            let mask = ((1u32 << (f as usize).saturating_sub(l).min(8)) - 1) as u8;
            _mm512_stream_si512(d.cast(), _mm512_maskz_loadu_epi64(mask, src.add(l).cast()));
            d = d.add(8);
        }
    }
}

/// A rayon pool of `threads` named "{name}-{i}", thread i pinned to cpus[i % len] if `cpus` is non-empty.
fn pinned_pool(threads: usize, cpus: Vec<usize>, name: &'static str) -> rayon::ThreadPool {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |i| format!("{name}-{i}"))
        .start_handler(move |i| {
            if !cpus.is_empty() {
                unsafe {
                    let mut set: libc::cpu_set_t = std::mem::zeroed();
                    libc::CPU_SET(cpus[i % cpus.len()], &mut set);
                    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
                }
            }
        })
        .build()
        .unwrap();
    pool.broadcast(|_| ());
    pool
}

/// One scan thread's state: its bursts, the current block of each partition, and its block log.
struct Scatter {
    buf: Map<u64>,
    used: Map<u8>,
    block: Vec<u32>,
    fill: Vec<u32>,
    log: Vec<(u32, u32)>, // (partition, block) in allocation order
    local: std::ops::Range<usize>, // this thread's own blocks
}

impl Scatter {
    /// Where partition p's next entries go: its current block, or a new one (its own, else from the shared rest of
    /// the arena) if that is full.
    fn dst(&mut self, p: usize, builder: &Builder) -> std::io::Result<*mut u64> {
        if self.block[p] == NONE || self.fill[p] as usize == BLOCK {
            let b = self.local.next().unwrap_or_else(|| builder.overflow.0.fetch_add(1, Relaxed));
            if b >= builder.arena.len() / BLOCK {
                return Err(std::io::Error::other("scan arena is full"));
            }
            (self.block[p], self.fill[p]) = (b as u32, 0);
            self.log.push((p as u32, b as u32));
        }
        Ok(unsafe { builder.arena.ptr.add(self.block[p] as usize * BLOCK + self.fill[p] as usize) })
    }

    #[inline(never)]
    fn flush(&mut self, p: usize, builder: &Builder) -> std::io::Result<()> {
        let dst = self.dst(p, builder)?;
        unsafe {
            if *AVX512 {
                stream_line_avx512(dst, self.buf.ptr.add(p * BURST))
            } else {
                stream(dst, &self.buf[p * BURST..(p + 1) * BURST])
            }
        }
        self.fill[p] += BURST as u32;
        Ok(())
    }
}

/// The shared-block counter, on cache lines of its own: every scan thread bumps it and every burst flush reads the
/// builder, so on a line with the builder's fields (by the object's address) it slowed the scan about twofold.
#[repr(align(128))]
struct Counter(AtomicUsize);

/// Rank 0's build state.
struct Builder {
    files: Vec<File>,
    sizes: Vec<usize>,
    pool: rayon::ThreadPool,
    arena: Map<u64>,
    overflow: Box<Counter>, // the next arena block past the threads' own
    table: Map<u64>,
    scatters: Vec<Mutex<Scatter>>,
    finishes: Vec<Mutex<(Map<u64>, Map<u8>)>>, // a partition's slice of the table and its bucket fill counts
}

/// The table is one shared-memory file: rank 0 creates and builds it, the other ranks open it read-only. Every rank
/// looks up its own positions on a pool of its own, reading buckets in place from its mapping of the table.
#[pyclass]
pub struct SlotIndex {
    view: Map<u64>, // mmap: the table, mapped and faulted in before the clock
    lookup: rayon::ThreadPool,
    builder: Option<Builder>,
    key: usize,
}

fn view(file: &File, key: usize) -> std::io::Result<Map<u64>> { // mmap: the table of index `key`, faulted in
    let len = SLOTS << (PBITS + nbits(key));
    Ok(Map { ptr: mmap(std::ptr::null_mut(), len * 8, libc::MAP_SHARED | libc::MAP_POPULATE, file.as_raw_fd())?.cast(), len, bytes: len * 8 })
}

#[pymethods]
impl SlotIndex {
    /// Rank 0, before the clock: create the table file `path` (it must not exist) for `files`, all training shards,
    /// and fault in the build's memory; nothing is read yet. The build runs on `threads` threads pinned round-robin
    /// to `cpus`, lookups one per `lookup_cpus` (either may be empty: unpinned).
    #[staticmethod]
    fn create(py: Python<'_>, path: String, files: Vec<String>, threads: usize, cpus: Vec<usize>, lookup_cpus: Vec<usize>, key: usize) -> PyResult<Self> {
        let (files, sizes): (Vec<File>, Vec<usize>) = files.iter().map(|f| open_shard(f)).collect::<PyResult<Vec<_>>>()?.into_iter().unzip();
        // Full blocks, one partial block per (thread, partition), 2% slack. A thread takes blocks from its own
        // first-touch slice (half its fair share, so a slow thread strands little) and then from the shared rest.
        let full = sizes.iter().map(|&n| n.saturating_sub(MIN)).sum::<usize>().div_ceil(BLOCK);
        let local = full / 2 / threads;
        let file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path)?;
        let (slice, len) = (SLOTS << nbits(key), SLOTS << (PBITS + nbits(key)));
        file.set_len((len * 8) as u64)?;
        let table = mmap(std::ptr::null_mut(), len * 8, libc::MAP_SHARED, file.as_raw_fd())?;
        let builder = Builder {
            files,
            sizes,
            pool: pinned_pool(threads, cpus, "slotindex"),
            arena: Map::new((full * 51 / 50 + threads * (1 << PBITS) + 64) * BLOCK)?,
            overflow: Box::new(Counter(AtomicUsize::new(threads * local))),
            table: Map { ptr: table.cast(), len, bytes: len * 8 },
            scatters: (0..threads)
                .map(|t| {
                    let (buf, used) = (Map::new((1 << PBITS) * BURST)?, Map::new(1 << PBITS)?);
                    let (block, fill) = (vec![NONE; 1 << PBITS], vec![0; 1 << PBITS]);
                    Ok(Mutex::new(Scatter { buf, used, block, fill, log: Vec::new(), local: t * local..(t + 1) * local }))
                })
                .collect::<std::io::Result<_>>()?,
            finishes: (0..threads).map(|_| Ok(Mutex::new((Map::new(slice + 8)?, Map::new(slice / SLOTS)?)))).collect::<std::io::Result<_>>()?,
        };
        // Fault in the arena (each thread its own slice first), the table and the per-thread buffers, then drop the
        // table's mappings (its pages now exist in the file): each slice is mapped again only while it is written.
        let b = &builder;
        py.detach(|| {
            b.pool.broadcast(|ctx| {
                let (i, n) = (ctx.index(), ctx.num_threads());
                let part = |m: &Map<u64>, a: usize, b: usize| unsafe { std::slice::from_raw_parts_mut(m.ptr.add(a), b - a) };
                let (own, rest) = (local * BLOCK, b.arena.len() - n * local * BLOCK);
                touch(part(&b.arena, i * own, (i + 1) * own));
                touch(part(&b.arena, n * own + rest * i / n, n * own + rest * (i + 1) / n));
                touch(part(&b.table, len * i / n, len * (i + 1) / n));
                let (mut scatter, mut finish) = (b.scatters[i].lock().unwrap(), b.finishes[i].lock().unwrap());
                touch(&mut scatter.buf);
                touch(&mut scatter.used);
                touch(&mut finish.0);
                touch(&mut finish.1);
            });
            mmap(b.table.ptr.cast(), len * 8, libc::MAP_SHARED | libc::MAP_FIXED, file.as_raw_fd()).map(drop)
        })?;
        let lookup = pinned_pool(lookup_cpus.len().max(1), lookup_cpus, "slotindex-q");
        Ok(Self { view: py.detach(|| view(&file, key))?, lookup, builder: Some(builder), key })
    }

    /// The other ranks, before the clock, once rank 0 has created `path`.
    #[staticmethod]
    fn attach(path: String, lookup_cpus: Vec<usize>, key: usize) -> PyResult<Self> {
        let lookup = pinned_pool(lookup_cpus.len().max(1), lookup_cpus, "slotindex-q");
        let file = std::fs::OpenOptions::new().read(true).write(true).open(&path)?; // mmap: shared, as the build maps it
        Ok(Self { view: view(&file, key)?, lookup, builder: None, key })
    }

    /// Rank 0, on the clock: build the table.
    fn build(&self, py: Python<'_>) -> PyResult<()> {
        let (b, nbits) = (self.builder.as_ref().unwrap(), nbits(self.key));
        let pieces: Vec<(usize, usize)> = b.sizes.iter().enumerate().flat_map(|(f, &n)| (MIN..n).step_by(PIECE).map(move |a| (f, a))).collect();
        let next = AtomicUsize::new(0);
        let lens: Vec<AtomicUsize> = (0..b.arena.len() / BLOCK).map(|_| AtomicUsize::new(BLOCK)).collect();
        py.detach(|| {
            // 1. Scan.
            let scanned = b.pool.broadcast(|ctx| -> std::io::Result<()> {
                let mut guard = b.scatters[ctx.index()].lock().unwrap();
                let sc = &mut *guard;
                let (mut hashes, mut tokens) = (Hashes::new(), vec![STOP; PIECE + BACK]);
                while let Some(&(f, a)) = pieces.get(next.fetch_add(1, Relaxed)) {
                    // tokens[BACK + r] = shard token a + r; tokens before the shard's start are STOP.
                    let (n, lo) = ((a + PIECE).min(b.sizes[f]) - a, a.saturating_sub(BACK));
                    let pad = BACK - (a - lo);
                    tokens[..pad].fill(STOP);
                    pread(&b.files[f], &mut tokens[pad..BACK + n], 1024 + lo * 2)?;
                    for r in (0..n).step_by(STEP) {
                        let m = STEP.min(n - r);
                        hashes.hash(&tokens, BACK + r, m, true, self.key);
                        // Raw pointers in locals: through &mut the compiler reloads every field after each store.
                        let (buf, used) = (sc.buf.ptr, sc.used.ptr);
                        for (&e, &p) in hashes.e[..m].iter().zip(&hashes.part[..m]) {
                            let p = p as usize;
                            unsafe {
                                let u = *used.add(p) as usize;
                                *buf.add(p * BURST + u) = e;
                                if u + 1 == BURST {
                                    sc.flush(p, b)?;
                                    *used.add(p) = 0;
                                } else {
                                    *used.add(p) = u as u8 + 1;
                                }
                            }
                        }
                    }
                }
                // The partial bursts, then every current block's length.
                for p in 0..1 << PBITS {
                    let u = sc.used[p] as usize;
                    if u > 0 {
                        let dst = sc.dst(p, b)?;
                        unsafe { std::ptr::copy_nonoverlapping(sc.buf.ptr.add(p * BURST), dst, u) };
                        sc.fill[p] += u as u32;
                    }
                    if sc.block[p] != NONE {
                        lens[sc.block[p] as usize].store(sc.fill[p] as usize, Relaxed);
                    }
                }
                fence();
                Ok(())
            });
            scanned.into_iter().collect::<std::io::Result<()>>()?;

            // 2. Each partition's blocks: thread by thread, each in allocation order; the partitions largest first.
            let mut blocks = vec![Vec::new(); 1 << PBITS];
            for sc in &b.scatters {
                for &(p, k) in &sc.lock().unwrap().log {
                    blocks[p as usize].push(k as usize);
                }
            }
            let len = |k: usize| lens[k].load(Relaxed);
            let mut order: Vec<usize> = (0..1 << PBITS).collect();
            order.sort_by_cached_key(|&p| std::cmp::Reverse(blocks[p].iter().map(|&k| len(k)).sum::<usize>()));

            // 3. Finish: place each partition into its slice and stream the slice to the table.
            let next = AtomicUsize::new(0);
            b.pool.broadcast(|ctx| {
                let mut guard = b.finishes[ctx.index()].lock().unwrap();
                let (slice, fill) = &mut *guard;
                while let Some(&p) = order.get(next.fetch_add(1, Relaxed)) {
                    for &k in &blocks[p] {
                        let src = unsafe { std::slice::from_raw_parts(b.arena.ptr.add(k * BLOCK), len(k)) };
                        // Prefetch the input 4 KB ahead and the b1 slot of the entry PF ahead: random stores that miss
                        // L1 commit one at a time, so the line is fetched early and the store then hits L1.
                        for i in 0..src.len() {
                            unsafe {
                                if i % 8 == 0 {
                                    _mm_prefetch(src.as_ptr().wrapping_add(i + 512).cast(), _MM_HINT_T0);
                                }
                                if let Some(&e) = src.get(i + PF) {
                                    let h = b1(check(e), nbits);
                                    _mm_prefetch(slice.as_ptr().add(h * SLOTS + (*fill.get_unchecked(h) as usize).min(SLOTS - 1)).cast(), _MM_HINT_T0);
                                }
                                let e = *src.get_unchecked(i);
                                let c = check(e);
                                if let Some(h) = [b1(c, nbits), b2(c, nbits)].into_iter().find(|&h| (*fill.get_unchecked(h) as usize) < SLOTS) {
                                    let f = fill.get_unchecked_mut(h);
                                    *slice.get_unchecked_mut(h * SLOTS + *f as usize) = e;
                                    *f += 1;
                                }
                            }
                        }
                    }
                    // Map the slice in one call rather than a write fault per page, write it, and drop the mapping again.
                    let dst = unsafe { b.table.ptr.add(p * (SLOTS << nbits)) };
                    unsafe { libc::madvise(dst.cast(), (SLOTS << nbits) * 8, libc::MADV_POPULATE_WRITE) };
                    if *AVX512 {
                        unsafe { out_avx512(slice, fill, dst) };
                    } else {
                        for (j, &f) in fill.iter().enumerate() {
                            slice[j * SLOTS + f as usize..(j + 1) * SLOTS].fill(0);
                        }
                        unsafe { stream(dst, &slice[..SLOTS << nbits]) };
                    }
                    fill.fill(0);
                    fence();
                    unsafe { libc::madvise(dst.cast(), (SLOTS << nbits) * 8, libc::MADV_DONTNEED) };
                }
                fence();
            });
            Ok(())
        })
    }

    /// Rank 0, after the build: free the build's memory (a 90 GB scan arena takes a while) and unmap the table (the
    /// finished slices' page-table pages remain before Linux 6.14).
    fn release(&mut self, py: Python<'_>) {
        let builder = self.builder.take();
        py.detach(|| drop(builder));
    }

    /// After the build: the rows (cell, 8 slots) of every position of `tokens`, split into segments at BOS and
    /// every `chunk` tokens, into the rows of `out` still empty (backoff), their cells offset by `tag`.
    fn query_into(&self, py: Python<'_>, tokens: PyReadonlyArray1<u16>, chunk: usize, mut out: PyReadwriteArray2<i32>, tag: i32) -> PyResult<()> {
        let (x, nbits) = (tokens.as_slice()?, nbits(self.key));
        let mut starts: Vec<usize> = (0..x.len()).step_by(chunk).chain([x.len()]).collect();
        starts.extend(x.iter().enumerate().filter_map(|(i, &v)| (v == BOS).then_some(i)));
        starts.sort_unstable();
        starts.dedup();
        let out = out.as_slice_mut()?;
        assert_eq!(out.len(), x.len() * 9);
        let out = Output(out.as_mut_ptr());
        py.detach(|| {
            self.lookup.install(|| {
                starts.par_windows(2).try_for_each_init(
                    || (Hashes::new(), Vec::with_capacity(2 * SLOTS), Vec::new()),
                    |(hashes, tokens, t), w| -> std::io::Result<()> {
                        let (start, len) = (w[0], w[1] - w[0]);
                        t.clear();
                        t.resize(BACK, STOP);
                        t.extend_from_slice(&x[start..w[1]]);
                        for c in (self.key..=len).step_by(STEP) {
                            let m = STEP.min(len + 1 - c);
                            hashes.hash(t, BACK + c, m, false, self.key);
                            let bucket = |r: usize, b: usize| &self.view[hashes.part[r] as usize * (SLOTS << nbits) + b * SLOTS..][..SLOTS];
                            let filled = |r: usize| unsafe { *out.0.add((start + c + r - 1) * 9) } != 0;
                            for r in 0..m {
                                if r + 1 < m && !filled(r + 1) {
                                    let next = bucket(r + 1, b1(check(hashes.e[r + 1]), nbits));
                                    (0..SLOTS).step_by(8).for_each(|l| unsafe { _mm_prefetch(next.as_ptr().add(l).cast(), _MM_HINT_T0) });
                                }
                                if filled(r) {
                                    continue;
                                }
                                let e = hashes.e[r];
                                let levels = LEVELS.iter().filter(|&&l| l <= c + r).count();
                                let mut best = 0;
                                tokens.clear();
                                let mut scan = |bucket: &[u64]| {
                                    for &v in bucket.iter().filter(|&&v| v != 0 && check(v) == check(e)) {
                                        let d = v ^ e;
                                        let agree = (0..levels).take_while(|&k| (d >> SHIFTS[k]) & ((1 << BITS[k]) - 1) == 0).count();
                                        let level = if agree == 0 { self.key } else { LEVELS[agree - 1] };
                                        if level > best {
                                            best = level;
                                            tokens.clear();
                                        }
                                        if level == best {
                                            tokens.push((v & 0xffff) as i32);
                                        }
                                    }
                                };
                                let bucket1 = bucket(r, b1(check(e), nbits));
                                scan(bucket1);
                                let (h1, h2) = (b1(check(e), nbits), b2(check(e), nbits));
                                if h2 != h1 && bucket1[SLOTS - 1] != 0 {
                                    scan(bucket(r, h2));
                                }
                                if best > 0 {
                                    let mut rw = row(best, tokens);
                                    rw[0] += tag;
                                    unsafe { out.write((start + c + r - 1) * 9, &rw) };
                                }
                            }
                        }
                        Ok(())
                    },
                )
            })
        })?;
        Ok(())
    }
}
