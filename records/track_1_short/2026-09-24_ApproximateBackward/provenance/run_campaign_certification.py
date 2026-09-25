"""Frozen, interleaved eight-H100 comparisons. No failed/poor run is replaced."""
import argparse, datetime, hashlib, json, math, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('phase',choices=['pilot','certify','short_control','accepted'])
parser.add_argument('--tag',help='Unique directory name for an explicitly retained pilot retry')
parser.add_argument('--root',type=Path,default=Path('/workspace/nanogpt_submission'))
a=parser.parse_args();root=a.root.resolve()
if a.phase=='pilot':
    runs=['baseline','candidate','candidate','baseline']
elif a.phase=='certify':
    runs=['baseline','candidate','candidate','candidate','baseline','candidate',
          'candidate','candidate','baseline']*2  # Six baseline, twelve candidate.
elif a.phase=='accepted':
    runs=['accepted','accepted']
else:
    runs=['short12','short24','short24','short12']*2
tag=a.tag or a.phase
assert re.fullmatch(r'[A-Za-z0-9_-]+',tag)
out=root/'campaigns'/tag
out.mkdir(parents=True,exist_ok=True)
assert not (out/'protocol.json').exists(), 'Existing phase must be inspected, not silently resumed/replaced'
# Do not leave a parent CUDA context consuming GPU0 memory during training.
subprocess.run([sys.executable, '-c', "import torch; assert torch.cuda.device_count()==8; assert all('H100' in torch.cuda.get_device_name(i) for i in range(8))"], check=True)

def hash_sources(directory):
    paths=list(directory.glob('*.py'))+list((directory/'approx_backward').rglob('*'))
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file() and '__pycache__' not in p.parts}

sources={name:hash_sources(root/name) for name in ('baseline','candidate','accepted') if (root/name).is_dir()}
protocol=dict(phase=a.phase,accepted_master_commit='bc3a0c2d640d0d73dedaef87eae26148d2e32afb',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              runs=runs,unseeded=True,sources=sources,baseline_commit='c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7',
              cache_policy='Shared normal compiler caches; pilot timing reported separately; matching-source pilot losses included in quality test',
              loss_rule='All twelve planned certification candidate runs plus all complete matching-source pilot candidate runs; one-sided one-sample t-test vs 3.28; p < .01',
              step_control={'short12':1110,'short24':1098},
              clock='Unchanged PR360 clock: data loading, calibration, communication, optimizer and terminal ships included')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for i,arm in enumerate(runs):
    src='candidate' if arm=='candidate' else ('accepted' if arm=='accepted' else 'baseline')
    assert hash_sources(root/src)==sources[src], 'Source changed inside frozen cohort'
    run_dir=out/f'{i:02d}_{arm}';run_dir.mkdir()
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('AB_','HEAD_SAMPLE_','KX_')):env.pop(k)
    for k in ('TRAIN_SEED','NUM_EXTENSION_ITERATIONS'):
        env.pop(k,None)
    if arm=='accepted':
        env['LOCAL_KERNELS']='kernels-community/flash-attn3='+str(root/'fa3_accepted')
    if arm.startswith('short'):
        env['KX_STEPS']=str(protocol['step_control'][arm])
    started=time.time(); before=set((root/src/'logs').glob('*.txt'))
    cmd=[str(Path(sys.executable).with_name('torchrun')),'--standalone','--nproc_per_node=8','train_gpt.py']
    record=dict(index=i,arm=arm,source=src,command=cmd,started=started,
                step_override=env.get('KX_STEPS'),status='running')
    (run_dir/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    with (run_dir/'console.log').open('w') as f:
        process=subprocess.Popen(cmd,cwd=root/src,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=45*60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    record.update(exit_code=code,elapsed=time.time()-started,status='complete' if code==0 else 'failed')
    for p in sorted(set((root/src/'logs').glob('*.txt'))-before):shutil.copy2(p,run_dir/p.name)
    text=(run_dir/'console.log').read_text()
    matches=re.findall(r'step:(\d+)/(\d+) val_loss:([\d.]+) train_time:([\d.]+)ms',text)
    if matches:
        step,total,loss,ms=matches[-1]
        record.update(steps=int(total),complete_steps=int(step),val_loss=float(loss),training_ms=float(ms),loss_precision=4)
    for line in text.splitlines():
        if line.startswith('[approx-backward-result] '):
            detail=json.loads(line.split(' ',1)[1]);record.update(val_loss=detail['val_loss'],training_ms=detail['training_ms'],loss_precision='full')
            (run_dir/'approximation.json').write_text(json.dumps(detail,indent=2)+'\n')
    valid = (bool(matches) and record.get('complete_steps') == record.get('steps')
             and math.isfinite(record.get('val_loss', float('nan')))
             and math.isfinite(record.get('training_ms', float('nan'))))
    if not valid: record['status']='invalid'
    (run_dir/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if code!=0 or not valid:
        raise RuntimeError(f'Run {i} failed or incomplete: inspect {run_dir}; do not replace it')
(out/'COMPLETE').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
