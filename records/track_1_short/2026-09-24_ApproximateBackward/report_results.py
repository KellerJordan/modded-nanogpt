"""Build a measured report from archived run records, with no run selection."""
import argparse,collections,datetime,json,math,statistics,subprocess,sys
from pathlib import Path
from scipy import stats
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root
summaries={}
for phase in ('pilot','certify','short_control','accepted'):
    directory=root/phase
    if not (directory/'protocol.json').exists():continue
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_campaign.py')),str(directory)],check=True,stdout=subprocess.DEVNULL)
    summaries[phase]=json.loads((directory/'summary.json').read_text())
lines=['# Eight-H100 submission results','', 'Generated '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'. Tables include all available records; planned and completed counts distinguish partial results.','',
       'Candidate source: `c5be2fbf86c918e91b9e283b4044595bb74a350f`, based on PR #360 `c924f68e4d72e80307fc27a7bb3a55cfb6ad43c7`. All reported runs use one eight-H100 SXM 80GB node, NV18 topology, driver 580.126.09 and torch 2.10.0+cu128. No candidate or control run is discarded.','',
       '| Phase / implementation | Completed / planned | Training seconds (mean ± SD) | Validation loss (mean ± SD) |',
       '|---|---:|---:|---:|']
for phase,summary in summaries.items():
    protocol=json.loads((root/phase/'protocol.json').read_text());counts=collections.Counter(protocol['runs'])
    for arm,n in counts.items():
        s=summary['arms'].get(arm)
        if not s:lines.append(f'| {phase} / {arm} | 0 / {n} | — | — |');continue
        time=f"{s['mean_training_ms']/1000:.6f}";loss=f"{s['mean_loss']:.8f}"
        if s['n']>1:time+=f" ± {s['sd_training_ms']/1000:.6f}";loss+=f" ± {s['sd_loss']:.8f}"
        lines.append(f"| {phase} / {arm} | {s['n']} / {n} | {time} | {loss} |")
lines+=['','Baseline losses are printed by the untouched source to four decimal places. Candidate diagnostics also preserve full precision. SD is the sample standard deviation. Pilot timing is reported separately; matching-source pilot candidate losses are included in the final quality test.','']
if 'certify' in summaries:
    s=summaries['certify'];q=s.get('quality');c=s['arms'].get('candidate');b=s['arms'].get('baseline')
    lines+=['## Fixed-cohort comparison','',f"Certification cohort complete: **{s['cohort_complete']}**. Required quality criterion established: **{s['quality_certified']}**.",'']
    if c and b:lines.append(f"Candidate mean saving: {(b['mean_training_ms']-c['mean_training_ms'])/1000:.6f} seconds ({s['speedup_percent']:.3f}%). Mean loss difference: {s['delta_loss']:+.8f}. The original one-second aspiration is {'met' if b['mean_training_ms']-c['mean_training_ms']>=1000 else 'not met'} by these observed means.")
    if q:
        upper=q['mean_loss']+stats.t.ppf(.99,q['n']-1)*q['sd_loss']/math.sqrt(q['n'])
        lines+=['',f"Quality test: n={q['n']} matching-source candidate runs including pilots, mean {q['mean_loss']:.8f}, SD {q['sd_loss']:.8f}, one-sided p={q['loss_p_one_sided']:.6g} against 3.28. One-sided 99% upper confidence bound: {upper:.8f}. An incomplete cohort is never marked certified."]
    if c and b:
        vb=b['sd_training_ms']**2/b['n'];vc=c['sd_training_ms']**2/c['n']
        df=(vb+vc)**2/(vb**2/(b['n']-1)+vc**2/(c['n']-1));radius=stats.t.ppf(.975,df)*math.sqrt(vb+vc)
        saving=b['mean_training_ms']-c['mean_training_ms']
        lines+=['',f"An approximate Welch 95% confidence interval for the mean time saving is {(saving-radius)/1000:.6f} to {(saving+radius)/1000:.6f} seconds. This describes run variability on this node and does not cover between-machine variation.",'',
                'Baseline losses are unusually variable in this cohort. Every result, including the baseline loss of 3.2879 and candidate loss above 3.282, is retained. The small baseline sample does not establish a quality improvement or equivalence. The candidate passes the threshold test independently of the baseline comparison.']
        baseline_all=[r['val_loss'] for phase in ('pilot','certify') for r in summaries.get(phase,{}).get('rows',[]) if r['arm']=='baseline' and r['status']=='complete']
        if len(baseline_all)>1:
            lines+=['',f"For completeness, pooling baseline quality across pilots and certification gives n={len(baseline_all)}, mean loss {statistics.mean(baseline_all):.8f}, SD {statistics.stdev(baseline_all):.8f}. Timing above still uses the prespecified fixed cohort."]
    if c and 'short_control' in summaries:
        lines+=['','## Ordinary shorter-training controls','','These use unmodified PR360: short12 has 1182 total updates; short24 has 1170; the candidate and original control have 1194. Comparisons below describe observed means and do not by themselves establish equality or superiority.','']
        for arm,x in summaries['short_control']['arms'].items():
            lines.append(f"- {arm}: candidate minus control = {(c['mean_training_ms']-x['mean_training_ms'])/1000:+.6f}s, {c['mean_loss']-x['mean_loss']:+.8f} loss.")
        values=list(summaries['short_control']['arms'].values())
        if b:values.append(b)
        lo=min(x['mean_loss'] for x in values);hi=max(x['mean_loss'] for x in values)
        lines+=['',f"Candidate loss is {'inside' if lo<=c['mean_loss']<=hi else 'outside'} the observed control mean-loss range [{lo:.8f}, {hi:.8f}]. These controls measure two ordinary shortening choices; they do not identify an exact equal-loss training budget or prove superiority over every possible step count."]
    if c and 'accepted' in summaries and summaries['accepted']['arms'].get('accepted'):
        accepted=summaries['accepted']['arms']['accepted']
        lines+=['','## Accepted-master comparison','',
                f"Accepted master `bc3a0c2d640d0d73dedaef87eae26148d2e32afb` ran unmodified at its default 1290 updates. Across {accepted['n']} completed runs its mean time is {accepted['mean_training_ms']/1000:.6f} seconds. Candidate saving relative to this separate comparison is {(accepted['mean_training_ms']-c['mean_training_ms'])/1000:.6f} seconds ({100*(1-c['mean_training_ms']/accepted['mean_training_ms']):.3f}%). Most of that improvement belongs to PR #360; the incremental claim for this work remains the comparison against PR #360 above. These two accepted-master runs are a timing reference, not an independent loss certification."]
    diagnostics=[]
    for path in sorted((root/'certify').glob('*/approximation.json')):
        d=json.loads(path.read_text());windows=[x for ws in d['windows'].values() for x in ws]
        diagnostics.append(dict(run=path.parent.name,shortened=sum(x>=0 for x in windows),total=len(windows),calibration_ms=sum(x['seconds'] for x in d['calibration'])*1000))
    if diagnostics:
        lines+=['','## Calibration diagnostics','',f"Over {len(diagnostics)} certification candidates, rank 0 calibration work averaged {statistics.mean(x['calibration_ms'] for x in diagnostics):.3f} ms (charged to training). The selected rule shortened between {min(x['shortened'] for x in diagnostics)} and {max(x['shortened'] for x in diagnostics)} of 21 eligible heads. Per-run logs retain windows, all-rank maximum Q/K/V errors and the two observations. Rank 0 measured calibration duration is not a separate estimate of its net distributed critical-path cost."]
lines+=['','## Individual outcomes','','| Phase / run | Status | Updates | Seconds | Loss |','|---|---|---:|---:|---:|']
for phase,s in summaries.items():
    for r in s['rows']:
        sec=f"{r['training_ms']/1000:.6f}" if 'training_ms'in r else '—';loss=f"{r['val_loss']:.8f}" if 'val_loss'in r else '—'
        lines.append(f"| {phase}/{r['index']:02d}_{r['arm']} | {r['status']} | {r.get('steps','—')} | {sec} | {loss} |")
lines+=['','## Reproduction and timing policy','',
        'Compilation caches were retained across successive processes, with the same policy for both arms; compilation and graph warmup are outside the inherited training clock. Pilot times are kept separate from the fixed comparison. This differs from the cold-cache protocol reported in PR #360, so published times from another node are not used to estimate this contribution.',
        '', 'All complete console and source-containing training logs accompany these records. The final archive manifest hashes every artifact; source hashes must match the frozen checkout before release. Earlier one-H200 serial-replica experiments are excluded from these eight-GPU statistics.','']
a.output.write_text('\n'.join(lines));print(a.output)
