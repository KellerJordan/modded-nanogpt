//! SlotIndex: the validation index, approximate and corpus-free, with lookups of at most two bucket reads.
//!
//! One u64 entry per training position with a MIN-gram in its shard:
//!
//!   next token (16 bits) | fingerprint of each level (BITS[k] each) | key check (KBITS) ,
//!
//! where the MIN-gram key's top PBITS pick a partition and the top NBITS of the check the entry's home bucket b1 in
//! that partition; its second bucket b2 is another hash of the whole check. Level 10's fingerprint hashes the tokens
//! 7-10 before the context end, level 20's the tokens 9-20 (it is compared only once level 10's agrees). A bucket holds
//! `slots` entries, chosen at create from the corpus size so that the table is about LOAD full.
//!
//! Build (on the clock, rank 0 alone, on its own pinned pool, no GIL; AVX-512, with no branch on the data in the
//! inner loops):
//!   1. scan: thread t preads the 64K-token pieces t, t + T, ..., hashes 16 positions at a time (one group ahead of
//!      their inserts), and scatters every entry through a per-thread write-combining burst of one cache line per
//!      partition into 8192-entry blocks of a private huge-page arena, with non-temporal stores (no count pass, no
//!      sort). Each thread then lists its blocks by partition. A thread done with its own pieces takes the back half
//!      of the busiest remaining run of pieces into a spare burst state; the finish places a stolen run right after
//!      the pieces before it, so the table is the same as without stealing.
//!   2. finish: per partition (largest first, handed out dynamically), place its entries, thread by thread and each
//!      thread's blocks in allocation order, into b1 while it has room, else b2, else drop them (a hot key keeps its
//!      first 2 * slots entries). Only each bucket's filling line is staged; it is streamed to the table (shared
//!      memory, mapped and faulted in before the clock) the moment it is complete.
//!
//! Lookup (validation tokens only, after training, every rank its own positions, one batch at a time): every position
//! is hashed the same way, the positions are sorted by their home bucket's place in the table, and in that order pread
//! reads bucket b1 and, only if b1 is full, b2. Among the entries whose check equals the query's, a candidate's level
//! is the last of the unbroken ascending run of agreeing fingerprints (levels beyond the query's segment start never
//! agree), and the row is made of the candidates at the best level (lib.rs's row).
use crate::{compact, create_empty, fence, madvise, mmap, HEADER_BYTES, open_shard, pinned_pool, row, touch, Map, AVX512, BOS, LEVELS, MIN, STOP};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::fs::File;
use std::hint::select_unpredictable;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::{AcqRel, Acquire, Relaxed, Release}};
use std::sync::Mutex;

const BITS: [u32; 2] = [12, 6]; // fingerprint bits per level
const SHIFTS: [u32; 2] = [16, 28]; // where each level's fingerprint sits in an entry
const CHECK_SHIFT: u32 = 34;
const KBITS: u32 = 64 - CHECK_SHIFT; // key check bits
const PBITS: u32 = 14; // partitions: 2^14
const NBITS: u32 = 14; // buckets per partition: 2^14
const BUCKETS: usize = 1 << (PBITS + NBITS);
const LOAD: f64 = 0.8; // entries per slot (10.26B tokens: 48 slots, 103 GB; 41.2B: 192, 412 GB)
const BACK: usize = LEVELS[1]; // tokens before a context end that hash_avx512 reads
const STEP: usize = 1 << 12; // positions hashed at once by Hashes (the lookups; the scan hashes 16 at a time)
const PIECE: usize = 1 << 16; // context ends per piece of the scan (one pread)
const BURST: usize = 8; // entries per write-combining burst and per staging line: one cache line
const BLOCK: usize = 8192; // entries per arena block, a multiple of BURST
const CHUNK: usize = 64; // entries between the queued line stores of the scatter and the placement
const QN: usize = 32; // entries between the placement's queued line streams (u16 lanes of one zmm)
const PF: usize = 256; // entries the placement prefetches the arena ahead
const PF_HINT: i32 = _MM_HINT_T0;
const NONE: u32 = u32::MAX;
const K0: u64 = 0x9e3779b97f4a7c15; // 2^64 / the golden ratio: b2 is a Fibonacci hash of the check

/// The splitmix64 generator's output at `x`: its golden-ratio increment, then its finalizer (Steele, Lea and Flood).
const fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// The odd 52-bit multiplier and the 52-bit addend of IFMA word i, from splitmix64 at two arbitrary seeds (one for
/// the multipliers, one for the addends). hash_avx512's multipliers: 0-1 the key's words, 8 its high products, 2-3
/// level 10's words, 4-7 level 20's; its addends: 0 the key, 1 level 10, 2 level 20.
const fn m52(i: u64) -> u64 {
    (splitmix(0x1f3a_0000 + i) >> 12) | 1
}
const fn a52(i: u64) -> u64 {
    splitmix(0x2e4b_0000 + i) >> 12
}

/// How many of the first `levels` fingerprints agree in a row between entries that differ by `d` (their xor).
fn agree(levels: usize, d: u64) -> usize {
    (0..levels).take_while(|&k| (d >> SHIFTS[k]) & ((1 << BITS[k]) - 1) == 0).count()
}

#[inline(always)]
fn check(e: u64) -> u64 {
    (e >> CHECK_SHIFT) & ((1 << KBITS) - 1)
}

#[inline(always)]
fn b1(check: u64) -> usize {
    (check >> (KBITS - NBITS)) as usize
}

#[inline(always)]
fn b2(check: u64) -> usize {
    (check.wrapping_mul(K0) >> (64 - NBITS)) as usize
}

/// Bucket slots for a corpus of `tokens`: a multiple of BURST, below 256 (a bucket's fill is a byte).
fn slots_for(tokens: usize) -> PyResult<usize> {
    let slots = (tokens as f64 / (BUCKETS as f64 * LOAD) / BURST as f64).ceil() as usize * BURST;
    if slots >= 256 {
        return Err(PyValueError::new_err(format!("{tokens} tokens need {slots} slots per bucket, more than 248")));
    }
    Ok(slots.max(BURST))
}

/// pread exactly `buf` at byte `offset` of `file`.
fn pread<T>(file: &File, buf: &mut [T], offset: usize) -> std::io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, std::mem::size_of_val(buf)) };
    file.read_exact_at(bytes, offset as u64)
}

/// vpermb (1 uop; LLVM turns it into the 2-uop vpermw when the bytes move in pairs).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
unsafe fn vpermb(idx: __m512i, x: __m512i) -> __m512i {
    let d: __m512i;
    std::arch::asm!("vpermb {d}, {i}, {x}", d = lateout(zmm_reg) d, i = in(zmm_reg) idx, x = in(zmm_reg) x, options(pure, nomem, nostack));
    d
}

/// Scratch of one hashing step.
struct Hashes {
    e: Vec<u64>,
    part: Vec<u32>,
}

impl Hashes {
    fn new() -> Self {
        Self { e: vec![0; STEP], part: vec![0; STEP] }
    }

    /// The entries (with the next token if `next`) and partitions of the n contexts ending at t[j0..j0 + n]
    /// (exclusive ends): context r ends at t[j0 + r], which is its next token. Needs j0 >= BACK.
    fn hash(&mut self, t: &[u16], j0: usize, n: usize, next: bool) {
        assert!(*AVX512);
        unsafe { self.hash_avx512(t, j0, n, next) }
    }

    /// Multiply-add-shift hashes mod 2^52 (one vpmadd52luq per word) of 3-token words, word o of a context the 3
    /// tokens before its end - o: the key of words 0 and 3 (also the high halves of their products, then one more
    /// IFMA; bits 38-51 the partition, 8-37 the check), the level
    /// fingerprints of words 6, 7 (tokens 7-10) and 8, 11, 14, 17 (tokens 9-20), their top bits. Per 8 positions one
    /// load of the 32 tokens from 20 before the first context end; a word (lane: 3 tokens, then a copy of the first,
    /// as IFMA reads 52 bits) and the next token are byte permutes of it, and words 8, 11, 14 are the previous 8
    /// positions' 0, 3, 6.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma,avx512vbmi")]
    unsafe fn hash_avx512(&mut self, t: &[u16], j0: usize, n: usize, next: bool) {
        assert!(j0 >= BACK && t.len() >= j0 + n - 1 + next as usize && n <= STEP);
        let h = Hasher::new();
        let (ep, pp) = (self.e.as_mut_ptr(), self.part.as_mut_ptr());
        let mut p = h.start(t, j0);
        for r in (0..n).step_by(16) {
            let (lo, rlo, q) = h.eight(Hasher::load(t, j0, r), p, next);
            let (hi, rhi, q) = h.eight(Hasher::load(t, j0, r + 8), q, next);
            p = q;
            _mm512_storeu_si512(ep.add(r).cast(), lo);
            _mm512_storeu_si512(ep.add(r + 8).cast(), hi);
            _mm512_storeu_si512(pp.add(r).cast(), h.part(rlo, rhi));
        }
    }
}

/// hash_avx512's constants (see there), and its steps: 8 positions at a time, carrying the previous 8 positions'
/// words 0, 3, 6.
struct Hasher {
    idx: [__m512i; 9],
    keep: __m512i,
    m10: __m512i,
    low: __m512i,
    pmask: __m512i,
    pidx: __m512i,
}

impl Hasher {
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma,avx512vbmi")]
    #[inline]
    unsafe fn new() -> Self {
        const { assert!(STEP % 16 == 0 && CHUNK % 16 == 0 && CHECK_SHIFT + KBITS == 64 && KBITS + PBITS == 44 && BITS[0] == 12 && BITS[1] == 6 && SHIFTS[0] == 16 && SHIFTS[1] == 28) };
        // The offsets o of the permuted words; permute 8 is the next token.
        const OS: [usize; 8] = [0, 3, 6, 7, 17, 8, 11, 14];
        let idx: [__m512i; 9] = std::array::from_fn(|w| {
            let mut b = [0u8; 64];
            for k in 0..8 {
                for i in 0..4 {
                    let at = if w < 8 { k + BACK - 3 - OS[w] + i % 3 } else { k + BACK };
                    (b[8 * k + 2 * i], b[8 * k + 2 * i + 1]) = (2 * at as u8, 2 * at as u8 + 1);
                }
            }
            _mm512_loadu_si512(b.as_ptr().cast())
        });
        Self {
            idx,
            keep: _mm512_set1_epi64(!((1u64 << CHECK_SHIFT) - 1) as i64),
            m10: _mm512_set1_epi64((!((1u64 << CHECK_SHIFT) - 1) | ((1u64 << BITS[0]) - 1) << SHIFTS[0]) as i64),
            low: _mm512_set1_epi64(0xffff),
            pmask: _mm512_set1_epi32((1 << PBITS) - 1),
            pidx: _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
        }
    }

    /// The 32 tokens from BACK before context end j0 + r (zeros past the end of t).
    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    unsafe fn load(t: &[u16], j0: usize, r: usize) -> __m512i {
        let (start, avail) = (j0 + r - BACK, t.len() - (j0 + r - BACK));
        if avail >= 32 { _mm512_loadu_si512(t.as_ptr().add(start).cast()) } else { _mm512_maskz_loadu_epi16((1u32 << avail) - 1, t.as_ptr().add(start).cast()) }
    }

    /// The words 8, 11, 14 of the first 8 positions (the previous 8 positions' 0, 3, 6).
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    #[inline]
    unsafe fn start(&self, t: &[u16], j0: usize) -> [__m512i; 3] {
        let x = Self::load(t, j0, 0);
        [vpermb(self.idx[5], x), vpermb(self.idx[6], x), vpermb(self.idx[7], x)]
    }

    /// Entries r..r + 8 (with the next token if `next`) from x = load(t, j0, r) and the previous 8 positions' words
    /// 0, 3, 6; returns them,
    /// the key rotated (check at CHECK_SHIFT, partition at 0) and its words 0, 3, 6.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma,avx512vbmi")]
    #[inline]
    unsafe fn eight(&self, x: __m512i, p: [__m512i; 3], next: bool) -> (__m512i, __m512i, [__m512i; 3]) {
        let madd = |acc: __m512i, w: __m512i, i: u64| _mm512_madd52lo_epu64(acc, w, _mm512_set1_epi64(m52(i) as i64));
        let a = |i: u64| _mm512_set1_epi64(a52(i) as i64);
        let w = |i: usize| vpermb(self.idx[i], x);
        let (w0, w3, w6) = (w(0), w(1), w(2));
        // the key: low and high 52 bits of the products, combined (alone, the low sum mod 2^52 of tokens at
        // 16-bit offsets collides at 2^-20 for contexts differing only in tokens 1 and 4)
        let hi = |acc: __m512i, w: __m512i, i: u64| _mm512_madd52hi_epu64(acc, w, _mm512_set1_epi64(m52(i) as i64));
        let key = madd(madd(madd(a(0), w0, 0), w3, 1), hi(hi(_mm512_setzero_si512(), w0, 0), w3, 1), 8);
        let f10 = madd(madd(a(1), w6, 2), w(3), 3);
        let f20 = madd(madd(madd(madd(a(2), p[0], 4), p[1], 5), p[2], 6), w(4), 7);
        let rot = _mm512_rol_epi64::<{ 64 - (52 - PBITS as i32) }>(key);
        let x10 = _mm512_srli_epi64::<{ 52 - BITS[0] - SHIFTS[0] }>(f10);
        let x20 = _mm512_srli_epi64::<{ 52 - BITS[1] - SHIFTS[1] }>(f20);
        let e = _mm512_ternarylogic_epi64::<0xca>(self.keep, rot, x10); // keep ? rot : x10
        let e = _mm512_ternarylogic_epi64::<0xca>(self.m10, e, x20);
        let e = if next { _mm512_ternarylogic_epi64::<0xca>(self.low, vpermb(self.idx[8], x), e) } else { _mm512_andnot_si512(self.low, e) };
        (e, rot, [w0, w3, w6])
    }

    /// The partitions (u32) of 16 positions from their two rotated keys.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn part(&self, lo: __m512i, hi: __m512i) -> __m512i {
        _mm512_and_si512(_mm512_permutex2var_epi32(lo, self.pidx, hi), self.pmask)
    }
}

/// One scan thread's state: its bursts, the current block of each partition, its block log, and after the scan its
/// blocks by partition (partition p's are ks[off[p]..off[p + 1]], in allocation order).
struct Scatter {
    buf: Map<u64>,
    used: Map<u8>,
    block: Vec<u32>,
    fill: Vec<u32>,
    log: Vec<(u32, u32)>, // (partition, block) in allocation order
    local: std::ops::Range<usize>, // this thread's own blocks
    off: Vec<u32>,
    ks: Vec<u32>,
}

impl Scatter {
    /// Where partition p's next entries go: its current block, or a new one if that is full (a partition without a
    /// block has fill BLOCK, so one test covers both).
    #[inline(always)]
    fn dst(&mut self, p: usize, builder: &Builder) -> std::io::Result<*mut u64> {
        if self.fill[p] as usize == BLOCK {
            self.next_block(p, builder)?;
        }
        Ok(unsafe { builder.arena.ptr.add(self.block[p] as usize * BLOCK + self.fill[p] as usize) })
    }

    /// A new block: the thread's own, else one from the shared rest of the arena.
    #[cold]
    #[inline(never)]
    fn next_block(&mut self, p: usize, builder: &Builder) -> std::io::Result<()> {
        let b = self.local.next().unwrap_or_else(|| builder.overflow.0.fetch_add(1, Relaxed));
        if b >= builder.arena.len() / BLOCK {
            return Err(std::io::Error::other("scan arena is full"));
        }
        (self.block[p], self.fill[p]) = (b as u32, 0);
        self.log.push((p as u32, b as u32));
        Ok(())
    }

    /// The scan's hash and scatter, fused: the entries of the n contexts ending at t[j0..j0 + n] (Hasher, with the
    /// next token), 16 positions at a time into a small stack buffer, one group ahead of their inserts (so the buffer
    /// stays in L1, its loads never wait on the store just made, and the out-of-order core overlaps the vector hash
    /// with the scalar inserts), then scattered without a branch on a burst filling up: each entry is inserted into
    /// its burst line in a register (a 64 B load and a plain 64 B store, volatile: LLVM would make it a masked store,
    /// which a later load of the line cannot forward from), the line is also copied to a small queue whose index
    /// advances (by the lane bit) only when the burst is full, and every CHUNK entries the queued lines are streamed
    /// to their blocks, in order (the order the bursts filled up, so any chunking gives the same arena).
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma,avx512vbmi,bmi2")]
    unsafe fn hash_scatter(&mut self, b: &Builder, t: &[u16], j0: usize, n: usize) -> std::io::Result<()> {
        assert!(j0 >= BACK && t.len() >= j0 + n + 20);
        let h = Hasher::new();
        let (buf, used) = (self.buf.ptr, self.used.ptr);
        let mut q = [std::mem::MaybeUninit::<__m512i>::uninit(); CHUNK];
        let mut qp = [0u32; CHUNK];
        let (qv, qpp) = (q.as_mut_ptr().cast::<__m512i>(), qp.as_mut_ptr());
        #[repr(align(64))]
        struct Group([u64; 16], [u32; 16]);
        let mut gs = [Group([0; 16], [0; 16]), Group([0; 16], [0; 16])];
        let gp = gs.as_mut_ptr();
        // Hash the 16 positions from r into group buffer `g` (inserted one group later). Unmasked loads: t has 20
        // tokens to spare past the last group (lanes past n read stale tokens, and are never inserted).
        let mut p = h.start(t, j0);
        macro_rules! hash16 {
            ($r:expr, $g:expr) => {{
                let (ep, pp) = ((*$g).0.as_mut_ptr(), (*$g).1.as_mut_ptr());
                let (lo, rlo, x) = h.eight(_mm512_loadu_si512(t.as_ptr().add(j0 + $r - BACK).cast()), p, true);
                let (hi, rhi, x) = h.eight(_mm512_loadu_si512(t.as_ptr().add(j0 + $r + 8 - BACK).cast()), x, true);
                p = x;
                _mm512_store_si512(ep.cast(), lo);
                _mm512_store_si512(ep.add(8).cast(), hi);
                _mm512_store_si512(pp.cast(), h.part(rlo, rhi));
            }};
        }
        hash16!(0, gp);
        let (mut i, mut k) = (0, 0usize);
        while i < n {
            let c = CHUNK.min(n - i);
            let mut nq = 0usize;
            let mut r = 0;
            while r < c {
                if i + r + 16 < n {
                    hash16!(i + r + 16, gp.add((k + 1) & 1));
                }
                let (ep, pp) = ((*gp.add(k & 1)).0.as_ptr(), (*gp.add(k & 1)).1.as_ptr());
                macro_rules! one {
                    ($j:expr) => {{
                        let (x, p) = (*ep.add($j), *pp.add($j) as usize);
                        let u = *used.add(p) as u32;
                        let line = buf.add(p * BURST);
                        let bit = 1u32 << u;
                        let v = _mm512_mask_set1_epi64(_mm512_load_si512(line.cast()), bit as u8, x as i64);
                        std::ptr::write_volatile(line.cast::<__m512i>(), v);
                        *used.add(p) = ((u + 1) % BURST as u32) as u8;
                        *qv.add(nq) = v;
                        *qpp.add(nq) = p as u32;
                        nq += (bit >> (BURST - 1)) as usize;
                    }};
                }
                if c - r >= 16 {
                    for j in 0..16 {
                        one!(j);
                    }
                } else {
                    for j in 0..c - r {
                        one!(j);
                    }
                }
                r += 16;
                k += 1;
            }
            for x in 0..nq {
                let p = *qpp.add(x) as usize;
                let d = self.dst(p, b)?;
                _mm512_stream_si512(d.cast(), *qv.add(x));
                *self.fill.get_unchecked_mut(p) += BURST as u32;
            }
            i += c;
        }
        Ok(())
    }

    /// After the scan: the partial bursts, every current block's length, and the blocks by partition.
    fn close(&mut self, b: &Builder) -> std::io::Result<()> {
        for p in 0..1 << PBITS {
            let u = self.used[p] as usize;
            if u > 0 {
                let dst = self.dst(p, b)?;
                unsafe { std::ptr::copy_nonoverlapping(self.buf.ptr.add(p * BURST), dst, u) };
                self.fill[p] += u as u32;
            }
            if self.block[p] != NONE {
                b.lens[self.block[p] as usize].store(self.fill[p], Relaxed);
            }
        }
        fence();
        self.off.fill(0);
        for &(p, _) in &self.log {
            self.off[p as usize + 1] += 1;
        }
        for p in 0..1 << PBITS {
            self.off[p + 1] += self.off[p];
        }
        self.ks.resize(self.log.len(), 0);
        let mut at = self.off.clone();
        for &(p, k) in &self.log {
            self.ks[at[p as usize] as usize] = k;
            at[p as usize] += 1;
        }
        Ok(())
    }
}

/// A fresh scan state whose own blocks are `local` (then the shared rest), its block lists presized for `blocks`.
fn new_scatter(blocks: usize, local: std::ops::Range<usize>) -> std::io::Result<Mutex<Scatter>> {
    let (buf, used) = (Map::new((1 << PBITS) * BURST)?, Map::new(1 << PBITS)?);
    let (block, fill) = (vec![NONE; 1 << PBITS], vec![BLOCK as u32; 1 << PBITS]);
    let blocks = blocks + (1 << PBITS); // ample: presized, not grown on the clock
    let (log, ks) = (Vec::with_capacity(blocks), Vec::with_capacity(blocks));
    let off = vec![0; (1 << PBITS) + 1];
    Ok(Mutex::new(Scatter { buf, used, block, fill, log, local, off, ks }))
}

/// Place partition p's entries (its arena `blocks`, in order) and stream its buckets to the table at `dst`. Both fill
/// counts are read and the bucket chosen with selects; a fill stops at `slots`, and an entry that fits neither bucket
/// goes to b2 with fill `slots`: lane 0 of the staging line of a full bucket, whose last line is complete (slots is a
/// multiple of 8). Each entry is one 8 B store into its bucket's staging line `sp` (no line load, insert and store).
/// A completed line's bucket is queued in a zmm of u16 lanes (the lane index advancing only when the line is
/// complete), and every QN entries the queued lines are streamed to the table. A queued bucket's line is its last
/// complete one (at fill - 8) as long as no entry has reached the bucket since: an entry whose bucket is in the queue
/// (it would overwrite lane 0 of the queued line, or is dropped into it) first streams the queue (rare, a cold call).
/// At the end, the partial last lines (zeros past the fill). Never-written lines stay zero (the table is zero from
/// create).
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
unsafe fn place(b: &Builder, blocks: &[u32], sp: *mut u64, fp: *mut u8, dst: *mut u64) {
    let slots = b.slots;
    let none = _mm512_set1_epi16(-1);
    let mut queue = none;
    for (i, &k) in blocks.iter().enumerate() {
        let len = b.lens[k as usize].load(Relaxed) as usize;
        let start = b.arena.ptr.add(k as usize * BLOCK) as *const u64;
        let nxt = b.arena.ptr.add(*blocks.get(i + 1).unwrap_or(&k) as usize * BLOCK) as *const u64;
        let mut s = start;
        let end = s.add(len);
        while s < end {
            let mut n = 0usize;
            macro_rules! one {
                ($s:expr) => {{
                    let e = *$s;
                    let (h1, h2) = (b1(check(e)), b2(check(e)));
                    let (f1, f2) = (*fp.add(h1) as usize, *fp.add(h2) as usize);
                    let in1 = f1 < slots;
                    let h = select_unpredictable(in1, h1, h2);
                    let f = select_unpredictable(in1, f1, f2);
                    let bh = _mm512_set1_epi16(h as i16);
                    if _mm512_cmpeq_epi16_mask(queue, bh) != 0 {
                        flush(queue, n, sp, fp, dst, slots);
                        (n, queue) = (0, none);
                    }
                    *sp.add(h * BURST + f % BURST) = e;
                    *fp.add(h) = (f + (f < slots) as usize) as u8;
                    let done = (f % BURST == BURST - 1) as u32;
                    queue = _mm512_mask_mov_epi16(queue, done << n, bh);
                    n += done as usize;
                }};
            }
            if end.offset_from(s) >= QN as isize {
                // Prefetch the arena PF entries ahead, into the next block past this one's end (the hardware
                // prefetcher restarts at every 4 KB page and every block).
                let a = s.offset_from(start) as usize + PF;
                let pf = if a < len { start.add(a) } else { nxt.add((a - len) & !(BURST - 1)) };
                for j in 0..QN / BURST {
                    _mm_prefetch::<PF_HINT>(pf.add(BURST * j).cast());
                }
                for j in 0..QN {
                    one!(s.add(j));
                }
                s = s.add(QN);
            } else {
                while s < end {
                    one!(s);
                    s = s.add(1);
                }
            }
            stream_queued(queue, n, sp, fp, dst, slots);
            queue = none;
        }
    }
    for h in 0..1usize << NBITS {
        let f = *fp.add(h) as usize;
        if f % BURST != 0 {
            let mask = ((1u32 << (f % BURST)) - 1) as u8;
            _mm512_stream_si512(dst.add(h * slots + f - f % BURST).cast(), _mm512_maskz_loadu_epi64(mask, sp.add(h * BURST).cast()));
        }
        *fp.add(h) = 0;
    }
}

/// Stream the last complete line of each of the `n` queued buckets (u16 lanes of `q`) from its staging line.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
#[inline]
unsafe fn stream_queued(q: __m512i, n: usize, sp: *const u64, fp: *const u8, dst: *mut u64, slots: usize) {
    let mut hs = [0u16; QN];
    _mm512_storeu_si512(hs.as_mut_ptr().cast(), q);
    for &h in &hs[..n] {
        let h = h as usize;
        let at = h * slots + *fp.add(h) as usize - BURST;
        _mm512_stream_si512(dst.add(at).cast(), _mm512_load_si512(sp.add(h * BURST).cast()));
    }
}

/// place's conflict path: stream the queue now.
#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,bmi2")]
unsafe fn flush(q: __m512i, n: usize, sp: *const u64, fp: *const u8, dst: *mut u64, slots: usize) {
    stream_queued(q, n, sp, fp, dst, slots);
}

/// A value in cache lines of its own (a pair: the adjacent-line prefetcher fetches them together). The arena's shared
/// block counter, which every scan thread bumps, made the build a third slower (30B tokens: 10.4 s, not 7.8) when it
/// shared a line with the arena's pointer, which every burst reads: in about half the runs, depending on where Python
/// allocated the index.
#[repr(align(128))]
struct Padded<T>(T);

/// Rank 0's build state.
struct Builder {
    files: Vec<File>,
    sizes: Vec<usize>,
    pieces: Vec<(usize, usize)>, // (file, first context end) of each scan piece
    slots: usize,
    pool: rayon::ThreadPool,
    arena: Map<u64>,
    lens: Vec<AtomicU32>,   // each arena block's entries
    overflow: Box<Padded<AtomicUsize>>, // the next arena block past the lanes' own, in lines of its own
    table: Map<u64>,
    scatters: Vec<Mutex<Scatter>>,
    // Tail stealing of the scan, exact: lane t's pieces are t, t + T, ... (indices 0..n_t of its list); its owner claims
    // them from the front, a thread done with its own lane claims the back half of the busiest lane's rest into a spare
    // Scatter. Lane t's entries are then its owner's, followed by its stolen chunks in piece order: the same order as
    // without stealing, so the same table.
    claims: Vec<Padded<AtomicU64>>, // per stream (the lanes, then the spares): back << 32 | front
    lane_of: Vec<AtomicUsize>,      // per stream: its lane (a spare's is set when it steals)
    spares: Vec<Mutex<Scatter>>,
    spare_next: AtomicUsize,
    stolen: Mutex<Vec<(usize, usize, usize)>>, // (lane, first piece index, spare)
    finishes: Vec<Mutex<(Map<u64>, Map<u8>)>>, // a thread's staging line and fill count of every bucket of a slice
    built: AtomicBool, // build ran (the arena and scatter state are single-use)
}

/// The table is one shared-memory file: rank 0 creates and builds it, the other ranks open it read-only. Every rank
/// looks up its own positions on a pool of its own, reading buckets with pread (mapping nothing).
#[pyclass]
pub struct SlotIndex {
    file: File,
    slots: usize,
    lookup: rayon::ThreadPool,
    builder: Option<Builder>,
}

#[pymethods]
impl SlotIndex {
    /// Rank 0, before the clock: create the table file `path` (new or empty) for `files`, the training shards it
    /// indexes, fault in the build's memory and the table, and read the shards into the page cache (nothing is
    /// indexed). The build runs on `threads` threads pinned round-robin to `cpus`, lookups one per `lookup_cpus`
    /// (either may be empty: unpinned).
    #[staticmethod]
    fn create(py: Python<'_>, path: String, files: Vec<String>, threads: usize, cpus: Vec<usize>, lookup_cpus: Vec<usize>) -> PyResult<Self> {
        if !*AVX512 {
            return Err(PyValueError::new_err("SlotIndex needs AVX-512 with IFMA and VBMI"));
        }
        let (files, sizes): (Vec<File>, Vec<usize>) = files.iter().map(|f| open_shard(f)).collect::<PyResult<Vec<_>>>()?.into_iter().unzip();
        let slots = slots_for(sizes.iter().sum())?;
        let table_len = BUCKETS * slots;
        // Each (thread, partition) fills whole blocks and at most one partial one: at most `full` + threads * 2^PBITS
        // blocks in all. A thread takes blocks from its own first-touch slice of the arena first, then from the shared
        // rest. The slice is half a thread's fair share: the static piece split gives every thread about its fair share,
        // so each uses its slice up and strands none. The 2% and 64 blocks on top are slack for an uneven split; past
        // them the scan fails ("scan arena is full").
        let full = sizes.iter().map(|&n| n.saturating_sub(MIN)).sum::<usize>().div_ceil(BLOCK);
        let local = full / 2 / threads;
        let npieces = sizes.iter().map(|&n| n.saturating_sub(MIN).div_ceil(PIECE)).sum::<usize>();
        let spares = threads; // at most this many steals (each spare's partial blocks are slack below)
        let arena = Map::new((full * 51 / 50 + (threads + spares) * (1 << PBITS) + 64) * BLOCK)?;
        let file = create_empty(&path)?;
        file.set_len((table_len * 8) as u64)?;
        let table = mmap(table_len * 8, libc::MAP_SHARED, file.as_raw_fd())?;
        let builder = Builder {
            files,
            pieces: sizes.iter().enumerate().flat_map(|(f, &n)| (MIN..n).step_by(PIECE).map(move |a| (f, a))).collect(),
            sizes,
            slots,
            pool: pinned_pool(threads, cpus, "slotindex")?,
            lens: (0..arena.len() / BLOCK).map(|_| AtomicU32::new(BLOCK as u32)).collect(),
            arena,
            overflow: Box::new(Padded(AtomicUsize::new(threads * local))),
            table: Map { ptr: table.cast(), len: table_len, bytes: table_len * 8 },
            scatters: (0..threads).map(|t| new_scatter(2 * full / threads, t * local..(t + 1) * local)).collect::<std::io::Result<_>>()?,
            claims: (0..threads + spares).map(|t| Padded(AtomicU64::new(if t < threads { (((npieces + threads - 1 - t) / threads) as u64) << 32 } else { 0 }))).collect(),
            lane_of: (0..threads + spares).map(|t| AtomicUsize::new(t.min(threads))).collect(),
            spares: (0..spares).map(|_| new_scatter(full / threads / 2, 0..0)).collect::<std::io::Result<_>>()?,
            spare_next: AtomicUsize::new(0),
            stolen: Mutex::new(Vec::with_capacity(spares)),
            finishes: (0..threads).map(|_| Ok(Mutex::new((Map::new((1 << NBITS) * BURST)?, Map::new(1 << NBITS)?)))).collect::<std::io::Result<_>>()?,
            built: AtomicBool::new(false),
        };
        assert_eq!(builder.pieces.len(), npieces);
        // Fault in the per-thread buffers first, while 2 MB pages are still easy to get (on a fragmented host, after
        // hundreds of GB of arena and table they fell back to 4 KB pages, and the scan's burst stores then missed the
        // TLB), then the arena (each thread its own slice first) and the table, which stays mapped. Last, the shards
        // into the page cache (after the build memory, which could evict them), so that the scan's preads never wait on
        // the disk.
        let b = &builder;
        py.detach(|| {
            b.pool.broadcast(|ctx| -> std::io::Result<()> {
                let (i, n) = (ctx.index(), ctx.num_threads());
                {
                    let (mut scatter, mut finish) = (b.scatters[i].lock().unwrap(), b.finishes[i].lock().unwrap());
                    let scatter = &mut *scatter;
                    touch(&mut scatter.buf);
                    touch(&mut scatter.used);
                    touch(scatter.log.spare_capacity_mut());
                    touch(scatter.ks.spare_capacity_mut());
                    touch(&mut finish.0);
                    touch(&mut finish.1);
                }
                for spare in b.spares.iter().skip(i).step_by(n) {
                    let mut spare = spare.lock().unwrap();
                    touch(&mut spare.buf);
                    touch(&mut spare.used);
                    touch(spare.log.spare_capacity_mut());
                    touch(spare.ks.spare_capacity_mut());
                }
                let part = |m: &Map<u64>, a: usize, b: usize| unsafe { std::slice::from_raw_parts_mut(m.ptr.add(a), b - a) };
                let (own, rest) = (local * BLOCK, b.arena.len() - n * local * BLOCK);
                touch(part(&b.arena, i * own, (i + 1) * own));
                touch(part(&b.arena, n * own + rest * i / n, n * own + rest * (i + 1) / n));
                touch(part(&b.table, table_len * i / n, table_len * (i + 1) / n));
                for (f, &len) in b.files.iter().zip(&b.sizes).skip(i).step_by(n) {
                    let bytes = HEADER_BYTES + 2 * len;
                    let p = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0) };
                    if p == libc::MAP_FAILED {
                        return Err(std::io::Error::last_os_error());
                    }
                    let read = madvise(p, bytes, libc::MADV_POPULATE_READ);
                    unsafe { libc::munmap(p, bytes) };
                    read?;
                }
                Ok(())
            })
            .into_iter()
            .collect::<std::io::Result<()>>()
        })?;
        let lookup = pinned_pool(lookup_cpus.len().max(1), lookup_cpus, "slotindex-q")?;
        Ok(Self { file, slots, lookup, builder: Some(builder) })
    }

    /// The other ranks, before the clock, once rank 0 has created `path`.
    #[staticmethod]
    fn attach(path: String, lookup_cpus: Vec<usize>) -> PyResult<Self> {
        let file = File::open(&path)?;
        let bytes = file.metadata()?.len() as usize;
        if !*AVX512 || bytes % (BUCKETS * 8) != 0 {
            return Err(PyValueError::new_err("SlotIndex needs AVX-512 with IFMA and VBMI and a table of whole buckets"));
        }
        let lookup = pinned_pool(lookup_cpus.len().max(1), lookup_cpus, "slotindex-q")?;
        Ok(Self { file, slots: bytes / (BUCKETS * 8), lookup, builder: None })
    }

    /// Rank 0, on the clock: build the table.
    fn build(&self, py: Python<'_>) -> PyResult<()> {
        let b = self.builder.as_ref().ok_or_else(|| PyValueError::new_err("only the creating rank builds, before release"))?;
        if b.built.swap(true, Relaxed) {
            return Err(PyValueError::new_err("the validation index is already built"));
        }
        py.detach(|| {
            // 1. Scan.
            let scanned = b.pool.broadcast(|ctx| -> std::io::Result<()> {
                let (t, lanes) = (ctx.index(), ctx.num_threads());
                let mut tokens = vec![STOP; PIECE + BACK + 20]; // 20 to spare for hash_scatter's last loads
                // Scan piece index i of lane l into sc.
                let mut piece = |sc: &mut Scatter, l: usize, i: usize| -> std::io::Result<()> {
                    let (f, a) = b.pieces[l + i * lanes];
                    // tokens[BACK + r] = shard token a + r; tokens before the shard's start are STOP.
                    let (n, lo) = ((a + PIECE).min(b.sizes[f]) - a, a.saturating_sub(BACK));
                    let pad = BACK - (a - lo);
                    tokens[..pad].fill(STOP);
                    pread(&b.files[f], &mut tokens[pad..BACK + n], HEADER_BYTES + lo * 2)?;
                    unsafe { sc.hash_scatter(b, &tokens, BACK, n)? };
                    Ok(())
                };
                // Claim stream st's pieces from its front and scan them into sc.
                let mut run = |sc: &mut Scatter, st: usize| -> std::io::Result<()> {
                    let (claim, l) = (&b.claims[st].0, b.lane_of[st].load(Relaxed));
                    let mut c = claim.load(Acquire);
                    while (c as u32) < (c >> 32) as u32 {
                        match claim.compare_exchange_weak(c, c + 1, AcqRel, Acquire) {
                            Ok(_) => {
                                piece(sc, l, c as u32 as usize)?;
                                c = claim.load(Acquire);
                            }
                            Err(now) => c = now,
                        }
                    }
                    Ok(())
                };
                {
                    // Own lane.
                    let mut guard = b.scatters[t].lock().unwrap();
                    run(&mut guard, t)?;
                    guard.close(b)?;
                }
                // Then the back half of the busiest stream's rest (a lane's or a stolen chunk's, so a big stolen chunk
                // is split again), while some stream has STEAL pieces left, as a new stream of the same lane.
                const STEAL: u64 = 4;
                loop {
                    let k = b.spare_next.fetch_add(1, Relaxed);
                    if k >= b.spares.len() {
                        break;
                    }
                    let claimed = loop {
                        let (v, c) = (0..b.claims.len()).map(|x| (x, b.claims[x].0.load(Acquire))).max_by_key(|&(_, c)| (c >> 32).saturating_sub(c & 0xffff_ffff)).unwrap();
                        let (front, back) = (c & 0xffff_ffff, c >> 32);
                        if back < front + STEAL {
                            break None;
                        }
                        let first = back - (back - front) / 2;
                        if b.claims[v].0.compare_exchange(c, first << 32 | front, AcqRel, Acquire).is_ok() {
                            break Some((b.lane_of[v].load(Relaxed), first, back));
                        }
                    };
                    let Some((l, first, back)) = claimed else { break };
                    let st = lanes + k;
                    b.lane_of[st].store(l, Relaxed);
                    b.claims[st].0.store(back << 32 | first, Release);
                    let mut spare = b.spares[k].lock().unwrap();
                    run(&mut spare, st)?;
                    spare.close(b)?;
                    b.stolen.lock().unwrap().push((l, first as usize, k));
                }
                Ok(())
            });
            scanned.into_iter().collect::<std::io::Result<()>>()?;

            // 2. Finish: the partitions largest first, each placed into the table.
            // Every lane's owner, then its stolen chunks in piece order.
            let mut stolen = std::mem::take(&mut *b.stolen.lock().unwrap());
            stolen.sort();
            let mut scatters = Vec::with_capacity(b.scatters.len() + stolen.len());
            for (l, s) in b.scatters.iter().enumerate() {
                scatters.push(s.lock().unwrap());
                scatters.extend(stolen.iter().filter(|x| x.0 == l).map(|x| b.spares[x.2].lock().unwrap()));
            }
            let size = |p: usize| -> usize {
                scatters.iter().flat_map(|s| &s.ks[s.off[p] as usize..s.off[p + 1] as usize]).map(|&k| b.lens[k as usize].load(Relaxed) as usize).sum()
            };
            let sizes: Vec<usize> = b.pool.install(|| (0..1 << PBITS).into_par_iter().map(size).collect());
            let mut order: Vec<usize> = (0..1 << PBITS).collect();
            order.sort_by_key(|&p| std::cmp::Reverse(sizes[p]));
            let next = AtomicUsize::new(0);
            b.pool.broadcast(|ctx| {
                let mut guard = b.finishes[ctx.index()].lock().unwrap();
                let (stage, fill) = &mut *guard;
                let mut blocks = Vec::new();
                while let Some(&p) = order.get(next.fetch_add(1, Relaxed)) {
                    blocks.clear();
                    for s in &scatters {
                        blocks.extend_from_slice(&s.ks[s.off[p] as usize..s.off[p + 1] as usize]);
                    }
                    unsafe { place(b, &blocks, stage.ptr, fill.ptr, b.table.ptr.add(p * (1 << NBITS) * b.slots)) };
                }
                fence();
            });
            Ok(())
        })
    }

    /// Rank 0, after the lookups (off the clock): free the build's memory and unmap the table (on the clock its TLB
    /// shootdowns stalled the training thread for seconds).
    fn release(&mut self, py: Python<'_>) {
        let builder = self.builder.take();
        py.detach(|| drop(builder));
    }

    /// After the build: the compact rows (lib.rs's compact) of the positions of one batch `tokens`, a segment start
    /// (as are its BOS tokens), with `capacity` entries, and how many entries did not fit.
    fn query<'py>(&self, py: Python<'py>, tokens: PyReadonlyArray1<u16>, capacity: usize) -> PyResult<(Bound<'py, PyArray1<i32>>, usize)> {
        let x = tokens.as_slice()?;
        let slots = self.slots;
        let mut starts: Vec<usize> = [0].into_iter().chain(x.iter().enumerate().filter_map(|(i, &v)| (v == BOS).then_some(i))).chain([x.len()]).collect();
        starts.dedup();
        let mut cells = vec![0i32; x.len()];
        let mut entries = py.detach(|| {
            self.lookup.install(|| -> std::io::Result<Vec<[i32; 2]>> {
                // Hash every position with MIN tokens of its segment before it, as (home bucket, entry, row, levels in
                // its reach), and sort them by home bucket: the reads go through the table in order.
                let mut queries: Vec<(usize, u64, usize, usize)> = starts
                    .par_windows(2)
                    .flat_map_iter(|w| {
                        let (start, len) = (w[0], w[1] - w[0]);
                        let t = [vec![STOP; BACK], x[start..w[1]].to_vec()].concat();
                        let (mut hashes, mut out) = (Hashes::new(), Vec::new());
                        for c in (MIN..=len).step_by(STEP) {
                            let m = STEP.min(len + 1 - c);
                            hashes.hash(&t, BACK + c, m, false);
                            out.extend((0..m).map(|r| {
                                let (e, levels) = (hashes.e[r], LEVELS.iter().filter(|&&l| l <= c + r).count());
                                ((hashes.part[r] as usize) << NBITS | b1(check(e)), e, start + c + r - 1, levels)
                            }));
                        }
                        out
                    })
                    .collect();
                queries.par_sort_unstable_by_key(|q| q.0);
                let rows: Vec<(Vec<(usize, i32)>, Vec<[i32; 2]>)> = queries
                    .par_chunks(1024)
                    .map_init(
                        || (Vec::new(), Vec::new(), vec![0u64; slots], vec![0u64; slots]),
                        |(found, tokens, buf, buf2), queries| -> std::io::Result<_> {
                            let (mut cells, mut entries) = (Vec::new(), Vec::new());
                            for &(home, e, r, levels) in queries {
                                found.clear();
                                let mut scan = |bucket: &[u64]| {
                                    for &v in bucket.iter().filter(|&&v| v != 0 && check(v) == check(e)) {
                                        let depth = agree(levels, v ^ e);
                                        found.push((if depth == 0 { MIN } else { LEVELS[depth - 1] }, (v & 0xffff) as i32));
                                    }
                                };
                                pread(&self.file, buf, home * slots * 8)?;
                                scan(buf);
                                let second = (home >> NBITS << NBITS) | b2(check(e));
                                if second != home && buf[slots - 1] != 0 {
                                    pread(&self.file, buf2, second * slots * 8)?;
                                    scan(buf2);
                                }
                                if !found.is_empty() {
                                    cells.push((r, row(found, tokens, r as i32, &mut entries)));
                                }
                            }
                            Ok((cells, entries))
                        },
                    )
                    .collect::<std::io::Result<_>>()?;
                let mut all = Vec::new();
                for (c, e) in rows {
                    for (r, cell) in c {
                        cells[r] = cell;
                    }
                    all.extend(e);
                }
                Ok(all)
            })
        })?;
        Ok(compact(py, &cells, &mut entries, capacity))
    }
}
