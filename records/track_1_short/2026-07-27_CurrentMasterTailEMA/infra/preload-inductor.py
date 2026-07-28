#!/usr/bin/env python3
"""Install a Python 3.12 fork-worker GC guard, then run training.

Python 3.12 can deadlock a forked compile worker when garbage collection runs
an inherited ProcessPoolExecutor weakref callback.  The child-only at-fork hook
disables cyclic GC in Inductor's short-lived fork workers, preventing that
callback while leaving refcounting, worker count, compiler settings, and timed
training unchanged.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


preload_hook = Path(__file__).resolve().parent / "inductor-preload"
# Inductor reconstructs child PYTHONPATH from sys.path.
sys.path.insert(0, str(preload_hook))
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(preload_hook)
    if not existing_pythonpath
    else f"{preload_hook}{os.pathsep}{existing_pythonpath}"
)

import torch._inductor.runtime.triton_heuristics  # noqa: F401


if len(sys.argv) < 2:
    raise SystemExit("usage: preload-inductor.py TRAINING_SCRIPT [ARGS ...]")

training_script = str(Path(sys.argv[1]).resolve())
sys.path.insert(0, str(Path(training_script).parent))
sys.argv = [training_script, *sys.argv[2:]]
runpy.run_path(training_script, run_name="__main__")
