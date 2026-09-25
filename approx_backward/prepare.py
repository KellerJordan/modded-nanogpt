"""Fetch pinned headers, apply the source patch, and verify every build input.

Run once before torchrun. Build/capture is untimed; calibration stays in training.
The mixed QK/V dimensions in the patch are inherited from PR #360's FA3 build.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEPS = Path(os.environ.get("APPROX_BACKWARD_DEPS", ROOT.parent / ".approx_backward_deps"))
manifest = json.loads((ROOT / "dependencies.json").read_text())
DEPS.mkdir(parents=True, exist_ok=True)


def checkout(name, url, commit, sparse):
    dest = DEPS / name
    if not dest.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *sparse], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", commit], check=True)
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    assert actual == commit
    return dest


upstream = checkout("kernels-community", "https://github.com/huggingface/kernels-community.git",
                    manifest["kernels_community_commit"], ["flash-attn3"])
checkout("cutlass", "https://github.com/NVIDIA/cutlass.git", manifest["cutlass_commit"],
         ["include", "tools/util/include"])
headers = DEPS / "include"
headers.mkdir(exist_ok=True)
for name in manifest["patched_headers"]:
    shutil.copy2(upstream / "flash-attn3/flash-attn" / name, headers / name)
# Reconstruct the exact mixed-dimension headers used by PR #360 first. This
# source patch is published alongside its pinned binary; it is not our change.
patch_dir = DEPS / "fa3-source"
subprocess.run([str(Path(sys.executable).with_name("hf")), "download",
                "devenpzak/flash-attn3-12864", manifest["inherited_fa3_patch"],
                "--revision", manifest["fa3_binary_revision"], "--local-dir", str(patch_dir)], check=True)
inherited = (patch_dir / manifest["inherited_fa3_patch"]).read_bytes()
assert hashlib.sha256(inherited).hexdigest() == manifest["inherited_fa3_patch_sha256"]
# The published patch also contains forward/API files and template units that
# our two backward specializations do not compile. Apply only required headers.
header_patch, active = [], False
for line in inherited.decode().splitlines(True):
    if line.startswith("diff "):
        active = False
    elif line.startswith("--- "):
        name = line.split("\t")[0].split("/")[-1]
        active = name in manifest["patched_headers"]
        if active:
            header_patch.append("--- a/" + name + "\n")
    elif line.startswith("+++ "):
        if active:
            header_patch.append("+++ b/" + name + "\n")
    elif active:
        header_patch.append(line)
subprocess.run(["patch", "--batch", "--fuzz=0", "-p1"], cwd=headers,
               input="".join(header_patch), text=True, check=True)
subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(ROOT / "flash_attention.patch")],
               cwd=headers, check=True)
for name, expected in manifest["patched_headers"].items():
    assert hashlib.sha256((headers / name).read_bytes()).hexdigest() == expected, name
print(f"Verified {len(manifest['patched_headers'])} patched headers in {headers}")
