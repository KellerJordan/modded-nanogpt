//! Exact-match retrieval indexes. StreamIndex gives each training position the next tokens that followed its
//! context earlier in the training stream; SlotIndex gives each validation position the same over all training
//! shards. Both key a context on its last MIN tokens and bucket a match by the deepest of MIN and LEVELS it reaches.
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sfence, _mm_stream_si128};
use std::{fs::File, io::Read};

mod slot;
mod stream;

const BOS: u16 = 50256;
const STOP: u16 = u16::MAX;
const MIN: usize = 6;
const LEVELS: [usize; 2] = [10, 20];
const COUNTS: [usize; 6] = [2, 3, 5, 9, 17, 33]; // bin edges of a match's total count
const PURITIES: [f64; 5] = [0.3, 0.5, 0.7, 0.85, 0.95]; // bin edges of its top token's share of it
const CELLS: usize = (LEVELS.len() + 2) * (COUNTS.len() + 1) * (PURITIES.len() + 1);

/// A training shard file and its token count.
fn open_shard(path: &str) -> PyResult<(File, usize)> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 1024];
    file.read_exact(&mut header)?;
    let word = |i: usize| i32::from_le_bytes(header[i..i + 4].try_into().unwrap());
    let n = word(8);
    if word(0) != 20240520 || word(4) != 1 || n < 0 || file.metadata()?.len() != 1024 + n as u64 * 2 {
        return Err(PyValueError::new_err(format!("invalid shard: {path}")));
    }
    Ok((file, n as usize))
}

/// The row of a match at `length` with next `tokens` (sorted here): its cell (length bucket, bins of the total count
/// and of the top token's share) and its top 8 tokens by count (ties: lower token) as token | count << 16, else 0.
fn row(length: usize, tokens: &mut [i32]) -> [i32; 9] {
    tokens.sort_unstable();
    let (mut runs, mut n) = ([(0i32, 0i32); 96], 0); // k8: on the stack (at most 2 * 48 candidates), the top 8 sorted
    for run in tokens.chunk_by(|a, b| a == b) {
        (runs[n], n) = ((run[0], run.len() as i32), n + 1);
    }
    let key = |&(token, count): &(i32, i32)| (std::cmp::Reverse(count), token);
    if n > 8 {
        runs[..n].select_nth_unstable_by_key(7, key);
    }
    runs[..n.min(8)].sort_unstable_by_key(key);
    let counts = &runs[..n.min(8)];
    let total = tokens.len();
    let bucket = 1 + LEVELS.iter().filter(|&&l| l <= length).count();
    let count = COUNTS.iter().filter(|&&c| c <= total).count();
    let purity = PURITIES.iter().filter(|&&p| p <= counts[0].1 as f64 / total as f64).count();
    let slot = |i: usize| counts.get(i).map_or(0, |&(token, count)| token | count << 16);
    let cell = ((bucket * (COUNTS.len() + 1) + count) * (PURITIES.len() + 1) + purity) as i32;
    std::array::from_fn(|i| if i == 0 { cell } else { slot(i - 1) })
}

/// A raw pointer that threads write disjoint parts of.
#[derive(Clone, Copy)]
struct Output<T>(*mut T);
unsafe impl<T> Sync for Output<T> {}
unsafe impl<T> Send for Output<T> {}
impl<T: Copy> Output<T> {
    unsafe fn write(self, offset: usize, values: &[T]) {
        std::ptr::copy_nonoverlapping(values.as_ptr(), self.0.add(offset), values.len());
    }
    unsafe fn slice<'a>(self, offset: usize, len: usize) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(offset), len)
    }
    unsafe fn set(self, offset: usize, value: T) {
        *self.0.add(offset) = value;
    }
}

/// mmap read-write, in transparent huge pages, split into 256 MB VMAs (alternate ones flagged MADV_DONTDUMP) so that
/// a page-table walk (Slurm reads smaps every 30 s) holds the mmap lock only briefly per VMA. Not inherited by a fork:
/// the trainer forks a child on the clock (the canonical mask build), and every write here would copy a page.
fn mmap(at: *mut libc::c_void, bytes: usize, flags: libc::c_int, fd: libc::c_int) -> std::io::Result<*mut libc::c_void> {
    const VMA: usize = 1 << 28;
    let ptr = unsafe { libc::mmap(at, bytes, libc::PROT_READ | libc::PROT_WRITE, flags, fd, 0) };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    unsafe { libc::madvise(ptr, bytes, libc::MADV_HUGEPAGE) };
    unsafe { libc::madvise(ptr, bytes, libc::MADV_DONTFORK) };
    for off in (VMA..bytes).step_by(2 * VMA) {
        unsafe { libc::madvise(ptr.cast::<u8>().add(off).cast(), VMA.min(bytes - off), libc::MADV_DONTDUMP) };
    }
    Ok(ptr)
}

/// A mapping of `len` T's, unmapped on drop.
struct Map<T> {
    ptr: *mut T,
    len: usize,
    bytes: usize,
}
unsafe impl<T> Send for Map<T> {}
unsafe impl<T> Sync for Map<T> {}

impl<T> Map<T> {
    /// Private memory, zero until written, starting at a huge page.
    fn new(len: usize) -> std::io::Result<Self> {
        const HUGE: usize = 1 << 21;
        let bytes = (len * std::mem::size_of::<T>()).max(1).next_multiple_of(HUGE);
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
        let raw = mmap(std::ptr::null_mut(), bytes + HUGE, flags, -1)? as usize;
        let ptr = raw.next_multiple_of(HUGE);
        unsafe {
            libc::munmap(raw as *mut libc::c_void, ptr - raw);
            libc::munmap((ptr + bytes) as *mut libc::c_void, raw + HUGE - ptr);
        }
        Ok(Self { ptr: ptr as *mut T, len, bytes })
    }
}

impl<T> std::ops::Deref for Map<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> std::ops::DerefMut for Map<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for Map<T> {
    fn drop(&mut self) {
        // In 1 GB pieces with a pause between them: an unmap holds the process's memory-map lock, which the training
        // threads also take, so a 90 GB arena unmapped at once stalls them for the whole unmap.
        const PIECE: usize = 1 << 30;
        for off in (0..self.bytes).step_by(PIECE) {
            unsafe { libc::munmap(self.ptr.cast::<u8>().add(off).cast(), (self.bytes - off).min(PIECE)) };
            if off + PIECE < self.bytes {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

/// Write an even number of entries at a 16-byte aligned `dst` with non-temporal stores, which skip reading the
/// destination lines for ownership. Call `fence` once the thread is done.
unsafe fn stream(dst: *mut u64, values: &[u64]) {
    for i in (0..values.len()).step_by(2) {
        _mm_stream_si128(dst.add(i).cast::<__m128i>(), _mm_loadu_si128(values.as_ptr().add(i).cast()));
    }
}

fn fence() {
    unsafe { _mm_sfence() };
}

/// Write one byte per page of `v`, so later writes take no page faults.
fn touch<T>(v: &mut [T]) {
    let bytes = v.as_mut_ptr() as *mut u8;
    for i in (0..std::mem::size_of_val(v)).step_by(4096) {
        unsafe { std::ptr::write_volatile(bytes.add(i), 0) };
    }
}

/// `touch` in parallel 16 MB pieces, on the current rayon pool.
fn prefault<T>(v: &mut [T]) {
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) };
    bytes.par_chunks_mut(1 << 24).for_each(touch);
}

#[pymodule]
fn exact_match(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CELLS", CELLS)?;
    m.add_class::<stream::StreamIndex>()?;
    m.add_class::<slot::SlotIndex>()
}
