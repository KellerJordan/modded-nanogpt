"""Guard Python 3.12 Inductor fork workers from an inherited-lock deadlock."""

from __future__ import annotations

import gc
import os

import torch._inductor.runtime.triton_heuristics  # noqa: F401


# The coordinator imports this module before constructing its ProcessPoolExecutor.
# Its forked workers inherit an executor whose shutdown RLock may be held by a
# vanished coordinator thread.  Cyclic GC can collect that inherited executor
# and run a weakref callback which waits on the lock forever.  Fork workers are
# short-lived compiler processes; normal reference counting remains enabled.
os.register_at_fork(after_in_child=gc.disable)
