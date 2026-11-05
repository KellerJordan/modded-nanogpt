"""Keep freed host memory mapped: fixed glibc malloc thresholds (no per-block mmap under 32 MB, never trim).

The timed loop's helper threads allocate and free a few MB per job. With glibc's defaults that means mmap/munmap and
page faults, which take the mmap lock; a /proc/<pid>/smaps reader (Slurm's UsePss accounting) holds it for seconds on
rank 0's large validation mapping, stalling those threads while they hold the GIL. Cost: freed memory under 32 MB stays
with the process.
"""
import ctypes

M_TRIM_THRESHOLD, M_MMAP_THRESHOLD = -1, -3  # mallopt parameters (malloc.h); setting either fixes both
MMAP_THRESHOLD = 32 << 20  # glibc's largest (HEAP_MAX_SIZE / 2 on 64-bit)
TRIM_THRESHOLD = (1 << 31) - 1  # an int: never shrink a heap


def keep_freed_memory():
    """Call once, early: glibc's mallopt is process-wide, for every thread and arena."""
    libc = ctypes.CDLL("libc.so.6")
    ok = libc.mallopt(M_MMAP_THRESHOLD, MMAP_THRESHOLD) == 1 and libc.mallopt(M_TRIM_THRESHOLD, TRIM_THRESHOLD) == 1
    assert ok, "glibc mallopt rejected the malloc thresholds"
