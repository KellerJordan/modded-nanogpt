#!/usr/bin/env python3
"""Extract the actual archived trainer and prepare (never execute) a replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex


def extract(text, name):
    begin, end = f"BEGIN_{name}\n", f"END_{name}\n"
    assert text.count(begin) == text.count(end) == 1
    return text.split(begin, 1)[1].split(end, 1)[0]


def record(text, prefix):
    lines = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1, prefix
    return json.loads(lines[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.log.read_text()
    source = extract(text, "ARCHIVED_TRAINER")
    wrapper = extract(text, "OBSERVATIONAL_WRAPPER")
    manifest = record(text, "REPRODUCTION_MANIFEST_JSON ")
    launch_args = record(text, "REPRODUCTION_ARGS_JSON ")
    assert hashlib.sha256(source.encode()).hexdigest() == manifest["trainer_sha256"]
    final = record(text, "FINAL_METRICS_JSON ")
    assert final["trainer_sha256"] == manifest["trainer_sha256"]
    assert int(launch_args[launch_args.index("--seed") + 1]) == final["seed"]
    assert launch_args[launch_args.index("--train-steps") + 1] == "3200"
    assert final["updates"] == final["step"] == 3200
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "trainer.py").write_text(source)
    (output / "observational_wrapper.py").write_text(wrapper)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    (output / "MANIFEST.json").write_bytes(manifest_bytes)
    repo_root = Path(__file__).resolve().parents[4]
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    launcher = '''#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
'''
    launcher += "cd " + shlex.quote(str(repo_root)) + "\n"
    launcher += 'export HYM_RELEASE_FILE="$script_dir/MANIFEST.json"\n'
    launcher += f"export HYM_RELEASE_SHA256={manifest_hash}\n"
    launcher += "export OMP_NUM_THREADS=1\nexport TORCHINDUCTOR_COMPILE_THREADS=1\n"
    launcher += 'python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" '
    launcher += '"$script_dir/observational_wrapper.py" --source "$script_dir/trainer.py" --out "$script_dir" '
    launcher += shlex.join(launch_args) + "\n"
    (output / "run.sh").write_text(launcher)
    print(f"Prepared archived seed {final['seed']} replay in {output}; training was not started.")


if __name__ == "__main__":
    main()
