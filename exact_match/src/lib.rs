//! Exact-match retrieval indexes. StreamIndex gives each training position the next tokens that followed its
//! context earlier in the training stream; SlotIndex gives each validation position the same over the training
//! shards it indexes. Both key a context on its last MIN tokens and bucket a match by the deepest of MIN and LEVELS
//! it reaches. canon.rs builds the canonical token mask of the final validation.
use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sfence, _mm_stream_si128};
use std::sync::LazyLock;
use std::{fs::File, io::Read};

mod canon;
mod slot;
mod stream;

/// A shard file's header: 256 int32 (magic, version, token count, ...), then the tokens as u16. The Python loader reads
/// this constant too (data.py).
const HEADER_BYTES: usize = 1024;
const BOS: u16 = 50256;
const STOP: u16 = u16::MAX;
const MIN: usize = 6;
const LEVELS: [usize; 2] = [10, 20];
const COUNTS: [usize; 6] = [2, 3, 5, 9, 17, 33]; // bin edges of a match's total count
const PURITIES: [f64; 5] = [0.3, 0.5, 0.7, 0.85, 0.95]; // bin edges of its top token's share of it
const CELLS: usize = (LEVELS.len() + 2) * (COUNTS.len() + 1) * (PURITIES.len() + 1);

/// The build loops are written for AVX-512 with IFMA and VBMI (the record's hosts have it: Sapphire Rapids, Zen 4).
static AVX512: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512ifma")
        && is_x86_feature_detected!("avx512vbmi")
        && is_x86_feature_detected!("bmi2")
});

/// A training shard file and its token count.
fn open_shard(path: &str) -> PyResult<(File, usize)> {
    let mut file = File::open(path)?;
    let mut header = [0u8; HEADER_BYTES];
    file.read_exact(&mut header)?;
    let word = |i: usize| i32::from_le_bytes(header[i..i + 4].try_into().unwrap());
    let n = word(8);
    if word(0) != 20240520 || word(4) != 1 || n < 0 || file.metadata()?.len() != (HEADER_BYTES + 2 * n as usize) as u64 {
        return Err(PyValueError::new_err(format!("invalid shard: {path}")));
    }
    Ok((file, n as usize))
}

/// The row of a match whose candidates are `found` (nonempty): each one's match length (at least MIN) and next token.
/// Its depth is how many LEVELS the longest reaches, and its candidates are those reaching that depth's length.
/// Returns its cell (length bucket 1 + depth, bins of the candidates' total count and of the top token's share of it),
/// and appends each distinct next token to `entries` as [at, token | count << 16].
fn row(found: &[(usize, i32)], tokens: &mut Vec<i32>, at: i32, entries: &mut Vec<[i32; 2]>) -> i32 {
    let depth = LEVELS.iter().filter(|&&l| l <= found.iter().map(|&(l, _)| l).max().unwrap()).count();
    let length = if depth == 0 { MIN } else { LEVELS[depth - 1] };
    tokens.clear();
    tokens.extend(found.iter().filter(|&&(l, _)| l >= length).map(|&(_, t)| t));
    tokens.sort_unstable();
    let mut top = 0;
    for run in tokens.chunk_by(|a, b| a == b) {
        top = top.max(run.len());
        entries.push([at, run[0] | (run.len() as i32) << 16]);
    }
    let total = tokens.len();
    let count = COUNTS.iter().filter(|&&c| c <= total).count();
    let purity = PURITIES.iter().filter(|&&p| p <= top as f64 / total as f64).count();
    (((1 + depth) * (COUNTS.len() + 1) + count) * (PURITIES.len() + 1) + purity) as i32
}

/// Compact rows, as the model reads them: the `cells` of a batch's positions, then `capacity` entries, the first of
/// `entries` (sorted: the order is the same in every run) and then padding with no count, spread over the positions
/// (entry i: [i % positions, 0]). Returns how many entries did not fit.
fn compact<'py>(py: Python<'py>, cells: &[i32], entries: &mut [[i32; 2]], capacity: usize) -> (Bound<'py, PyArray1<i32>>, usize) {
    entries.sort_unstable();
    let n = cells.len();
    let mut out = vec![0i32; n + 2 * capacity];
    out[..n].copy_from_slice(cells);
    for (i, e) in out[n..].chunks_exact_mut(2).enumerate() {
        e.copy_from_slice(&entries.get(i).copied().unwrap_or([(i % n) as i32, 0]));
    }
    (out.into_pyarray(py), entries.len().saturating_sub(capacity))
}

/// A caught panic's message.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic.downcast_ref::<&str>().copied().or_else(|| panic.downcast_ref::<String>().map(String::as_str)).unwrap_or("(no message)")
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

/// A new memfd (close-on-exec) named `name`, for shared memory no mount limits: its descriptor, which the caller owns.
/// A raw syscall, as standalone Python builds (uv's) lack os.memfd_create.
#[pyfunction]
fn memfd(name: &str) -> PyResult<i32> {
    let name = std::ffi::CString::new(name)?;
    let fd = unsafe { libc::syscall(libc::SYS_memfd_create, name.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(fd as i32)
}

/// Open `path` for a new shared mapping: a new file, or an empty one such as the memfd retrieval.shared makes.
fn create_empty(path: &str) -> PyResult<File> {
    let file = std::fs::OpenOptions::new().read(true).write(true).create(true).open(path)?;
    if file.metadata()?.len() != 0 {
        return Err(PyValueError::new_err(format!("{path} is not empty")));
    }
    Ok(file)
}

/// mmap read-write, in transparent huge pages, split into 256 MB VMAs (alternate ones flagged MADV_DONTDUMP) so that
/// a page-table walk (Slurm reads smaps every 30 s) holds the mmap lock only briefly per VMA. Not inherited by a fork,
/// where every write here would copy a page. An explicit local policy keeps first-touch placement and takes the mapping
/// out of automatic NUMA balancing, which migrated huge pages under the build (7.2-8.9 s at 35B tokens, 6.7 s without).
fn mmap(bytes: usize, flags: libc::c_int, fd: libc::c_int) -> std::io::Result<*mut libc::c_void> {
    const VMA: usize = 1 << 28;
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, flags, fd, 0) };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    madvise(ptr, bytes, libc::MADV_HUGEPAGE)?;
    madvise(ptr, bytes, libc::MADV_DONTFORK)?;
    // A warning only: containers without CAP_SYS_NICE (Docker's default seccomp profile) may not set a policy.
    const MPOL_LOCAL: libc::c_long = 4;
    if unsafe { libc::syscall(libc::SYS_mbind, ptr, bytes, MPOL_LOCAL, std::ptr::null::<libc::c_ulong>(), 0 as libc::c_ulong, 0 as libc::c_uint) } != 0 {
        eprintln!("mbind(MPOL_LOCAL) failed, NUMA balancing may migrate this mapping: {}", std::io::Error::last_os_error());
    }
    for off in (VMA..bytes).step_by(2 * VMA) {
        madvise(unsafe { ptr.cast::<u8>().add(off) }.cast(), VMA.min(bytes - off), libc::MADV_DONTDUMP)?;
    }
    Ok(ptr)
}

/// madvise, failing on an error.
fn madvise(ptr: *mut libc::c_void, bytes: usize, advice: libc::c_int) -> std::io::Result<()> {
    if unsafe { libc::madvise(ptr, bytes, advice) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
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
        let raw = mmap(bytes + HUGE, flags, -1)? as usize;
        let ptr = raw.next_multiple_of(HUGE);
        for (at, len) in [(raw, ptr - raw), (ptr + bytes, raw + HUGE - ptr)] {
            if len > 0 && unsafe { libc::munmap(at as *mut libc::c_void, len) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
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
        unsafe { libc::munmap(self.ptr.cast(), self.bytes) };
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

/// A rayon pool of `threads` named "{name}-{i}", thread i pinned to cpus[i % len] if `cpus` is non-empty.
fn pinned_pool(threads: usize, cpus: Vec<usize>, name: &'static str) -> PyResult<rayon::ThreadPool> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |i| format!("{name}-{i}"))
        .build()
        .map_err(|e| PyValueError::new_err(format!("{name} pool: {e}")))?;
    let pinned = pool.broadcast(|ctx| -> std::io::Result<()> {
        if cpus.is_empty() {
            return Ok(());
        }
        let cpu = cpus[ctx.index() % cpus.len()];
        if cpu >= libc::CPU_SETSIZE as usize {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("cpu {cpu} is past CPU_SETSIZE")));
        }
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::CPU_SET(cpu, &mut set) };
        if unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    });
    pinned.into_iter().collect::<std::io::Result<()>>()?;
    Ok(pool)
}

#[pymodule]
fn exact_match(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CELLS", CELLS)?;
    m.add("HEADER_BYTES", HEADER_BYTES)?;
    m.add_function(wrap_pyfunction!(canon::canonical_mask, m)?)?;
    m.add_function(wrap_pyfunction!(memfd, m)?)?;
    m.add_class::<stream::StreamIndex>()?;
    m.add_class::<slot::SlotIndex>()
}
