//! The canonical token mask of track_1_short/canonical_mask.py, bit-identical to its build_canonical_mask: bit x of row
//! p is set when the GPT-2 tokenizer never emits token x right after token p. No AVX-512.
//!
//! The tokens come as their bytes in id order (an id is also the rank of the merge that forms the token), with the
//! Unicode side precomputed in Python before the clock: each token's start class, and for each previous token the
//! start classes a seam merge may follow it with (none if it ends in a contraction).
//!
//!   1. trajectories (in parallel over tokens): replay BPE on each token's bytes; its first and last piece over time
//!      as (piece, start rank, end rank) intervals, and its rule, the pair its final merge joins.
//!   2. candidates: rule r = (a, b) masks x after p when p's last piece is a at rank r and x's first piece is b; per
//!      rule, the x's of each start class.
//!   3. rows (in parallel over rows, each written once): for each last-piece interval (a, s, e) of p, the rules with
//!      left piece a and rank in [s, e), and their x's of the start classes p allows.
use crate::{fence, pinned_pool, stream};
use numpy::{PyReadonlyArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const NEVER: u32 = 1 << 30; // the end of a piece's last interval
const CLASSES: usize = 8; // start classes, one bit each of a previous token's allowed set
const ROWS: usize = 64; // rows per task

/// Multiply-rotate hashing of the short byte strings of a vocabulary (SipHash would dominate the BPE replay).
#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            self.write_u64(u64::from_le_bytes(w));
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517cc1b727220a95);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type Ranks<'a> = HashMap<&'a [u8], u32, BuildHasherDefault<Fx>>;

/// A token's first and last piece over its BPE replay as (piece, start, end) with start < end, and its rule.
struct Trajectory {
    first: Vec<(u32, u32, u32)>,
    last: Vec<(u32, u32, u32)>,
    rule: Option<(u32, u32)>,
}

/// Its BPE replay, with `scratch` reused across tokens.
fn trajectory(b: &[u8], ranks: &Ranks, scratch: &mut (Vec<usize>, Vec<u32>, Vec<u32>)) -> Trajectory {
    const NONE: u32 = u32::MAX;
    let rank = |s: usize, e: usize| ranks.get(&b[s..e]).copied().unwrap_or(NONE);
    // Piece i is b[cut[i]..cut[i + 1]], token piece[i]; joined[i] is the rank of pieces i and i + 1 joined, kept up
    // to date at each merge, which only changes the pairs next to it.
    let (cut, piece, joined) = scratch;
    cut.clear();
    cut.extend(0..=b.len());
    piece.clear();
    piece.extend(b.iter().map(|&c| ranks[&[c][..]]));
    joined.clear();
    joined.extend((0..b.len() - 1).map(|i| rank(i, i + 2)));
    let (mut first, mut last, mut rule) = (vec![(0, piece[0])], vec![(0, piece[b.len() - 1])], None);
    // the lowest rank, on ties the first pair
    while let Some((i, r)) = joined.iter().copied().enumerate().min_by_key(|&(_, r)| r).filter(|&(_, r)| r != NONE) {
        rule = Some((piece[i], piece[i + 1]));
        cut.remove(i + 1);
        piece.remove(i + 1);
        joined.remove(i);
        piece[i] = r;
        if i + 1 < piece.len() {
            joined[i] = rank(cut[i], cut[i + 2]);
        }
        if i > 0 {
            joined[i - 1] = rank(cut[i - 1], cut[i + 1]);
        }
        if i == 0 {
            first.push((r + 1, r));
        }
        if i == piece.len() - 1 {
            last.push((r, r));
        }
    }
    // (start, piece) changes -> intervals, dropping empty ones
    let intervals = |traj: Vec<(u32, u32)>| {
        let ends = traj.iter().skip(1).map(|&(s, _)| s).chain([NEVER]);
        traj.iter().zip(ends).filter(|&(&(s, _), e)| s < e).map(|(&(s, piece), e)| (piece, s, e)).collect()
    };
    Trajectory { first: intervals(first), last: intervals(last), rule }
}

/// Compressed rows: the values of key k are values[offsets[k]..offsets[k + 1]], in the order pushed.
struct Csr {
    offsets: Vec<u32>,
    values: Vec<u32>,
}

impl Csr {
    /// From the (key, value) pairs that `pairs` pushes, which it is called twice for (count, then fill).
    fn new(keys: usize, pairs: impl Fn(&mut dyn FnMut(usize, u32))) -> Self {
        let mut offsets = vec![0u32; keys + 1];
        pairs(&mut |k, _| offsets[k + 1] += 1);
        for k in 0..keys {
            offsets[k + 1] += offsets[k];
        }
        let mut fill = offsets.clone();
        let mut values = vec![0u32; offsets[keys] as usize];
        pairs(&mut |k, v| {
            values[fill[k] as usize] = v;
            fill[k] += 1;
        });
        Self { offsets, values }
    }
    fn get(&self, k: usize) -> &[u32] {
        &self.values[self.offsets[k] as usize..self.offsets[k + 1] as usize]
    }
}

/// The bytes of `words`, little-endian, so bit x of a row is bit x & 63 of word x >> 6.
fn as_bytes(words: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 8) }
}

/// The ranks in [s, e) of a piece's rules (ascending).
fn in_range(rules: &[u32], s: u32, e: u32) -> &[u32] {
    &rules[rules.partition_point(|&r| r < s)..rules.partition_point(|&r| r < e)]
}

/// On the clock (no GIL): fill `out`, (vocab, vocab / 8) bytes, with the mask of the tokens `bytes[offsets[t]..
/// offsets[t + 1]]`, t = 0, 1, ..., given each token's start class `start_cls` and each previous token's allowed start
/// classes `allowed` (a bit set), on `threads` threads pinned round-robin to `cpus` (may be empty: unpinned).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn canonical_mask(
    py: Python<'_>,
    mut out: PyReadwriteArray2<u8>,
    bytes: PyReadonlyArray1<u8>,
    offsets: PyReadonlyArray1<i64>,
    start_cls: PyReadonlyArray1<u8>,
    allowed: PyReadonlyArray1<u8>,
    threads: usize,
    cpus: Vec<usize>,
) -> PyResult<()> {
    let (vocab, width) = (out.shape()[0], out.shape()[1]);
    let (bytes, offsets, start_cls, allowed) = (bytes.as_slice()?, offsets.as_slice()?, start_cls.as_slice()?, allowed.as_slice()?);
    let n = offsets.len().saturating_sub(1);
    let tok = |t: usize| &bytes[offsets[t] as usize..offsets[t + 1] as usize];
    if n > vocab || width != vocab.div_ceil(8) || start_cls.len() < n || allowed.len() < n
        || offsets.first() != Some(&0) || offsets.windows(2).any(|w| w[0] >= w[1]) || offsets[n] as usize != bytes.len()
        || start_cls.iter().any(|&c| c as usize >= CLASSES)
    {
        return Err(PyValueError::new_err("canonical_mask: inconsistent shapes or tables"));
    }
    let out = out.as_slice_mut()?;
    py.detach(|| {
        pinned_pool(threads.max(1), cpus, "canonmask")?.install(|| {
            let ranks: Ranks = (0..n).map(|t| (tok(t), t as u32)).collect();
            if ranks.len() != n || (0..=255u8).any(|c| !ranks.contains_key(&[c][..])) {
                return Err(PyValueError::new_err("canonical_mask: tokens must be distinct and include every byte"));
            }
            // 1. Trajectories.
            let trajs: Vec<Trajectory> = (0..n).into_par_iter().map_init(Default::default, |scratch, t| trajectory(tok(t), &ranks, scratch)).collect();

            // 2. Candidates: the rules by left and by right piece, and each rule's x's by start class.
            let rules = |side: fn((u32, u32)) -> u32| {
                Csr::new(n, |push| (0..n).for_each(|t| trajs[t].rule.into_iter().for_each(|ab| push(side(ab) as usize, t as u32))))
            };
            let (by_left, by_right) = (rules(|(a, _)| a), rules(|(_, b)| b));
            let by_right = &by_right;
            let pairs: Vec<(u32, u32)> = (0..n)
                .into_par_iter()
                .flat_map_iter(|x| {
                    trajs[x].first.iter().flat_map(move |&(b, s, e)| {
                        in_range(by_right.get(b as usize), s, e).iter().map(move |&r| (r * CLASSES as u32 + start_cls[x] as u32, x as u32))
                    })
                })
                .collect();
            let xs = Csr::new(n * CLASSES, |push| pairs.iter().for_each(|&(k, x)| push(k as usize, x)));

            // 3. Rows, each built in a cached buffer and then streamed out (no read for ownership of the output).
            let words = width.div_ceil(8);
            let streamed = width % 16 == 0 && out.as_ptr() as usize % 16 == 0;
            out.par_chunks_mut(width * ROWS).enumerate().for_each_init(
                || vec![0u64; words],
                |buf, (i, rows)| {
                    for (p, row) in (i * ROWS..).zip(rows.chunks_mut(width)) {
                        buf.fill(0);
                        if p < n && allowed[p] != 0 {
                            for &(a, s, e) in &trajs[p].last {
                                for &r in in_range(by_left.get(a as usize), s, e) {
                                    let mut classes = allowed[p];
                                    while classes != 0 {
                                        for &x in xs.get(r as usize * CLASSES + classes.trailing_zeros() as usize) {
                                            buf[x as usize >> 6] |= 1 << (x & 63);
                                        }
                                        classes &= classes - 1;
                                    }
                                }
                            }
                        }
                        if streamed {
                            unsafe { stream(row.as_mut_ptr().cast(), buf) };
                        } else {
                            row.copy_from_slice(&as_bytes(buf)[..width]);
                        }
                    }
                    fence();
                },
            );
            Ok(())
        })
    })
}
