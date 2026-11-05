"""The run log: every record's log file starts with the full source code and the environment."""
import atexit
import os
import subprocess
import sys
from pathlib import Path

import torch
import triton

PACKAGE_DIR = Path(__file__).resolve().parent
LOG_BUFFER_BYTES = 1 << 20
USABLE_CPUS = len(os.sched_getaffinity(0))


def read_source(entry_script: str) -> str:
    """Concatenate the entry script, every module of this package and the exact_match extension's Rust
    source, for the run log.

    Called first thing at startup so the log holds the code as it was at launch.
    """
    root = Path(entry_script).resolve().parent
    files = [Path(entry_script)] + sorted(PACKAGE_DIR.rglob("*.py")) + sorted((root / "exact_match" / "src").glob("*.rs"))
    chunks = []
    for path in files:
        text = path.read_text()
        if chunks:
            chunks.append(f"\n\n{'-'*40}\n# {path.resolve().relative_to(root)}\n{'-'*40}\n\n")
        chunks.append(text)
    return "".join(chunks)


def start_run_log(master_process: bool, run_id: str):
    """Create logs/<run_id>.txt on rank 0 and return (print0, flush).

    print0(s, console=False) always appends to the log file; with console=True it also prints to
    stdout. The file is one block-buffered handle rather than an open() + close() per line (record #360):
    call flush() where the clock is stopped; it is also flushed at exit.
    """
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        path = f"logs/{run_id}.txt"
        print(path)
        logfile = open(path, "a", buffering=LOG_BUFFER_BYTES)
        atexit.register(logfile.close)  # close() flushes

    def print0(s, console=False):
        if master_process:
            if console:
                print(s)
            print(s, file=logfile)

    def flush():
        if logfile is not None:
            logfile.flush()

    return print0, flush


def log_environment(print0, code: str) -> None:
    from huggingface_hub.constants import HF_HUB_CACHE
    from track_1_short.model.attention import flash_attn_interface
    print0(code)
    print0("=" * 100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0(f"Running Triton version {triton.__version__}")
    fa3 = Path(flash_attn_interface.__file__)
    print0(f"Running FA3 build {fa3.relative_to(HF_HUB_CACHE) if fa3.is_relative_to(HF_HUB_CACHE) else fa3}")
    cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()
    model = next((l.split(":", 1)[1].strip() for l in cpuinfo if l.startswith("model name")), "unknown")
    print0(f"CPU: {model}, {os.cpu_count()} threads ({USABLE_CPUS} usable)")
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    total = next(int(l.split()[1]) for l in meminfo if l.startswith("MemTotal")) << 10
    memory = f"Memory: {total / 2**30:.0f} GiB"
    cgroup = Path("/proc/self/cgroup").read_text().splitlines()[0].rpartition(":")[2]
    limit = Path(f"/sys/fs/cgroup{cgroup}/memory.max")
    if limit.exists() and (value := limit.read_text().strip()) != "max":
        memory += f" (cgroup limit {int(value) / 2**30:.0f} GiB)"
    print0(memory)
    print0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    print0("=" * 100)
