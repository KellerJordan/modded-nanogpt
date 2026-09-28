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


def read_source(entry_script: str) -> str:
    """Concatenate the entry script and every module of this package, for the run log.

    Called first thing at startup so the log holds the code as it was at launch.
    """
    files = [Path(entry_script)] + sorted(PACKAGE_DIR.rglob("*.py"))
    root = Path(entry_script).resolve().parent
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
    call flush() where the clock is stopped; it is also flushed at exit. A forked child (the canonical
    mask build) leaves with os._exit, which never flushes the inherited buffer, so no line is written twice.
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
    print0(code)
    print0("=" * 100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0(f"Running Triton version {triton.__version__}")
    print0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    print0("=" * 100)
