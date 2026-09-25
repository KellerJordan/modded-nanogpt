"""Report every completed/failed run and the prespecified certification test."""
import json,sys
from pathlib import Path
import numpy as np
from scipy import stats
root=Path(sys.argv[1]);protocol=json.loads((root/'protocol.json').read_text())
rows=[json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))]
summary={'phase':protocol['phase'],'planned_runs':len(protocol['runs']),'attempted_runs':len(rows),'rows':rows,'arms':{}}
for arm in sorted(set(protocol['runs'])):
    values=[r for r in rows if r['arm']==arm and r['status']=='complete' and 'val_loss' in r]
    loss=np.array([r['val_loss'] for r in values]);ms=np.array([r['training_ms'] for r in values])
    if not len(values):continue
    s={'n':len(values),'mean_loss':float(loss.mean()),'mean_training_ms':float(ms.mean())}
    if len(values)>1:
        s.update(sd_loss=float(loss.std(ddof=1)),sd_training_ms=float(ms.std(ddof=1)),
                 loss_p_one_sided=float(stats.ttest_1samp(loss,3.28,alternative='less').pvalue))
    summary['arms'][arm]=s
if {'candidate','baseline'}<=summary['arms'].keys():
    c,b=summary['arms']['candidate'],summary['arms']['baseline']
    summary.update(speedup_percent=100*(1-c['mean_training_ms']/b['mean_training_ms']),
                   delta_loss=c['mean_loss']-b['mean_loss'])
summary['cohort_complete']=len(rows)==len(protocol['runs']) and all(r['status']=='complete' and 'steps' in r and r.get('complete_steps')==r['steps'] and np.isfinite(r.get('val_loss',float('nan'))) and np.isfinite(r.get('training_ms',float('nan'))) for r in rows)
quality_rows=[dict(phase_directory=root.name,**r) for r in rows if r['arm']=='candidate' and r['status']=='complete' and 'val_loss' in r]
if protocol['phase']=='certify':
    for previous in sorted(root.parent.glob('*/protocol.json')):
        if previous.parent==root:continue
        old=json.loads(previous.read_text())
        if old.get('phase')!='pilot' or old['sources']['candidate']!=protocol['sources']['candidate']:continue
        for file in sorted(previous.parent.glob('*/result.json')):
            row=json.loads(file.read_text())
            if row['arm']=='candidate' and row['status']=='complete' and row.get('complete_steps')==row.get('steps') and 'val_loss' in row:
                quality_rows.append(dict(phase_directory=previous.parent.name,**row))
summary['quality_runs_including_matching_pilots']=quality_rows
if len(quality_rows)>1:
    loss=np.array([r['val_loss'] for r in quality_rows])
    summary['quality']=dict(n=len(loss),mean_loss=float(loss.mean()),sd_loss=float(loss.std(ddof=1)),
                            loss_p_one_sided=float(stats.ttest_1samp(loss,3.28,alternative='less').pvalue))
summary['quality_certified']=bool(protocol['phase']=='certify' and summary['cohort_complete'] and summary.get('quality',{}).get('loss_p_one_sided',1)<.01)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
