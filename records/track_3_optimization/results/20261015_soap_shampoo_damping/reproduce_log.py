"""Extract a research trainer and prepare its recorded launch command; does not run training."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex

from claim_stats import parse_resolved_config

ROOT = Path(__file__).resolve().parent


def prepare(logfile, output):
    source, metadata = logfile.read_text().split('=' * 100, 1)
    ast.parse(source)
    config = parse_resolved_config(source, metadata)
    # Validate numeric values using the same declared types as the research trainer.
    tree = ast.parse(source)
    declaration = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'HPARAMS' for t in n.targets))
    defaults = {item.arg: ast.literal_eval(item.value) for item in declaration.keywords}
    int(config['seed'])
    if config['stop_step'] != 'None':
        int(config['stop_step'])
    for key, default in defaults.items():
        type(default)(config[key])
    args = []
    for key, value in config.items():
        if key == 'stop_step' and value == 'None':
            continue
        args += ['--' + key.replace('_', '-'), value]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = output / 'train_from_log.py'
    script.write_text(source)
    command = shlex.join([str(script), *args])
    launcher = output / 'run.sh'
    launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                        f'cd -- {shlex.quote(str(ROOT.parents[3]))}\n'
                        'exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" '
                        + command + '\n')
    launcher.chmod(0o755)
    (output / 'resolved_config.json').write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    digest = hashlib.sha256(source.encode()).hexdigest()
    (output / 'trainer.sha256').write_text(f'{digest}  train_from_log.py\n')
    return launcher, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logfile', type=Path)
    parser.add_argument('output_directory', type=Path, help='new directory; must not already exist')
    args = parser.parse_args()
    launcher, digest = prepare(args.logfile, args.output_directory)
    print(f'Extracted trainer SHA-256: {digest}')
    print(f'Prepared command: bash {shlex.quote(str(launcher))}')


if __name__ == '__main__':
    main()
