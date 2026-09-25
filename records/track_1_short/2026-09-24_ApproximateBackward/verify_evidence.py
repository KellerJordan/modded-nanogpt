"""Read-only verification of hashes, raw run logs and reported statistics.

Requires SciPy, but neither PyTorch nor a GPU. No evidence files are rewritten.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from scipy import stats


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime-root', type=Path, help='Also compare the live training sources with the benchmarked snapshot')
args = parser.parse_args()
root = Path(__file__).resolve().parent
manifest = read_json(root / 'packet_manifest.json')
actual_files = {str(p.relative_to(root)) for p in root.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.name != 'packet_manifest.json'}
assert set(manifest) == actual_files, 'Unlisted or missing evidence files'
for name, expected in manifest.items():
    path = root / name
    assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], name

bootstrap = read_json(root / 'provenance/bootstrap_manifest.json')
runtime_checked = 0
for name, expected in bootstrap['candidate_sources'].items():
    assert digest(root / 'source' / name) == expected, name
    # Documentation and container installation can change after measurement.
    if args.runtime_root and (Path(name).suffix in ('.py', '.cpp', '.cu', '.patch', '.json')
                              or name in ('run.sh', 'requirements.txt')):
        assert digest(args.runtime_root / name) == expected, name
        runtime_checked += 1

rows = []
protocols = {}
for phase, count in (('pilot', 4), ('certify', 18), ('short_control', 8), ('accepted', 2)):
    directory = root / 'runs' / phase
    protocol = protocols[phase] = read_json(directory / 'protocol.json')
    assert (directory / 'COMPLETE').is_file()
    files = sorted(directory.glob('*/result.json'))
    assert len(files) == len(protocol['runs']) == count, phase
    phase_rows = []
    for index, path in enumerate(files):
        record = read_json(path)
        assert record['index'] == index and record['arm'] == protocol['runs'][index], path
        assert record['status'] == 'complete' and record['exit_code'] == 0, path
        raw = (path.parent / 'console.log').read_text()
        pattern = r'^step:(\d+)/(\d+) val_loss:([\d.]+) train_time:([\d.]+)ms'
        step, total, rounded_loss, rounded_ms = re.findall(pattern, raw, re.M)[-1]
        assert int(step) == int(total) == record['steps'] == record['complete_steps'], path
        expected_steps = dict(baseline=1194, candidate=1194, short12=1182, short24=1170, accepted=1290)
        assert int(total) == expected_steps[record['arm']], path
        source_logs = list(path.parent.glob('*.txt'))
        assert len(source_logs) == 1 and re.findall(pattern, source_logs[0].read_text(), re.M)[-1] == (step, total, rounded_loss, rounded_ms), path
        loss, milliseconds = float(rounded_loss), float(rounded_ms)
        if record['arm'] == 'candidate':
            detail = json.loads(re.findall(r'^\[approx-backward-result\] (.+)$', raw, re.M)[-1])
            assert detail == read_json(path.parent / 'approximation.json'), path
            assert [entry['step'] for entry in detail['calibration']] == [592, 593], path
            assert detail['head_sample_group'] == 4 and detail['attention_stop'] == 1107, path
            assert detail['attention_enabled'] and detail['threshold'] == 0.2 and detail['head_min_rows'] == 32768, path
            assert abs(detail['val_loss'] - loss) <= 0.000050001, path
            assert abs(detail['training_ms'] - milliseconds) <= 0.500001, path
            loss, milliseconds = detail['val_loss'], detail['training_ms']
        assert math.isfinite(loss) and math.isfinite(milliseconds) and milliseconds > 0, path
        assert loss == record['val_loss'] and milliseconds == record['training_ms'], path
        phase_rows.append(dict(phase=phase, arm=record['arm'], loss=loss, ms=milliseconds))
    summary = read_json(directory / 'summary.json')
    assert summary['cohort_complete'], phase
    for arm in set(protocol['runs']):
        selected = [r for r in phase_rows if r['arm'] == arm]
        reported = summary['arms'][arm]
        assert reported['n'] == len(selected), (phase, arm)
        for field, key in (('mean_loss', 'loss'), ('mean_training_ms', 'ms')):
            assert math.isclose(reported[field], statistics.mean(r[key] for r in selected), abs_tol=1e-10), (phase, arm, field)
        for field, key in (('sd_loss', 'loss'), ('sd_training_ms', 'ms')):
            assert math.isclose(reported[field], statistics.stdev(r[key] for r in selected), abs_tol=1e-10), (phase, arm, field)
    rows.extend(phase_rows)

assert protocols['pilot']['sources']['candidate'] == protocols['certify']['sources']['candidate']
losses = [r['loss'] for r in rows if r['arm'] == 'candidate']
assert len(losses) == 14
p_value = float(stats.ttest_1samp(losses, 3.28, alternative='less').pvalue)
reported = read_json(root / 'runs/certify/summary.json')['quality']
assert reported['n'] == len(losses) and math.isclose(reported['loss_p_one_sided'], p_value, abs_tol=1e-12)
assert p_value < 0.01
fixed_losses = [r['loss'] for r in rows if r['arm'] == 'candidate' and r['phase'] == 'certify']
fixed_p_value = float(stats.ttest_1samp(fixed_losses, 3.28, alternative='less').pvalue)
assert len(fixed_losses) == 12 and fixed_p_value < 0.01

numeric_log = (root / 'provenance/distributed_check.log').read_text()
checks = [json.JSONDecoder().raw_decode(numeric_log[m.start():])[0]
          for m in re.finditer(r'\{"passed": true,', numeric_log)]
assert len(checks) == 8 and {r['rank'] for r in checks} == set(range(8))
assert all(r['passed'] and r['stride'] == 8 and r['results']['distributed_error_max'] for r in checks)
times = {arm: statistics.mean(r['ms'] for r in rows if r['phase'] == 'certify' and r['arm'] == arm)
         for arm in ('baseline', 'candidate')}
print(json.dumps(dict(passed=True, artifacts_verified=len(manifest),
                     frozen_source_files=len(bootstrap['candidate_sources']), runtime_files_checked=runtime_checked,
                     raw_training_logs_verified=len(rows), distributed_check_ranks=len(checks),
                     candidate_quality_runs=len(losses), mean_loss=statistics.mean(losses),
                     loss_p_one_sided=p_value, fixed_cohort_p_one_sided=fixed_p_value,
                     saving_seconds=(times['baseline'] - times['candidate']) / 1000,
                     speedup_percent=100 * (1 - times['candidate'] / times['baseline'])), indent=2))
