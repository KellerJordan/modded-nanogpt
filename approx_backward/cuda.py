"""Build the two SM90 backward specializations from pinned, verified sources."""
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
DEPS = Path(os.environ.get("APPROX_BACKWARD_DEPS", ROOT.parent / ".approx_backward_deps"))
MANIFEST = json.loads((ROOT / "dependencies.json").read_text())
HEADERS = DEPS / "include"
sources = [ROOT / "csrc/wrapper.cpp", ROOT / "csrc/runner.cu"]
digest = hashlib.sha256()
for path in [ROOT / "dependencies.json", *sources]:
    digest.update(path.read_bytes())
for name, expected in MANIFEST["patched_headers"].items():
    path = HEADERS / name
    assert path.is_file(), "Run python approx_backward/prepare.py before training"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, f"Modified dependency: {path}"
    digest.update(data)
NAME = "nanogpt_ab_" + digest.hexdigest()[:12]
os.environ.setdefault("MAX_JOBS", "4")
load(name=NAME, sources=[str(p) for p in sources],
     extra_include_paths=[str(HEADERS), str(DEPS / "cutlass/include"),
                          str(DEPS / "cutlass/tools/util/include")],
     extra_cflags=["-O3", "-std=c++17"],
     extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "--ftemplate-backtrace-limit=0",
                       "-gencode=arch=compute_90a,code=sm_90a",
                       "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DCUTLASS_ENABLE_GDC_FOR_SM90",
                       "-DCUTLASS_DEBUG_TRACE_LEVEL=0", "-DNDEBUG"],
     is_python_module=False, verbose=os.environ.get("NATIVE_VERBOSE", "0") == "1")
ops = getattr(torch.ops, NAME)


@torch.library.register_fake(NAME + "::bwd")
def fake_bwd(dout, q, k, v, out, lse, cu, windows, maxlen, scale, full_window):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def backward(dout, q, k, v, out, lse, cu, windows, maxlen, scale, full_window):
    return ops.bwd(dout, q, k, v, out, lse, cu, windows, maxlen, scale, full_window)
