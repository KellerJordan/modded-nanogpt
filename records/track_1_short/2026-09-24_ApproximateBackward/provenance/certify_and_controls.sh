#!/bin/bash
set -euo pipefail
cd /workspace/nanogpt_submission
source env.sh
trap 'rc=$?; if [ "$rc" -ne 0 ]; then date -u > CERTIFICATION_PIPELINE_FAILED; fi' EXIT
python summarize_campaign_next.py campaigns/pilot > pilot_summary_next.log
python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('campaigns/pilot/summary.json').read_text())
assert s['cohort_complete'], 'Pilot is incomplete: inspect before continuing'
assert s['arms']['candidate']['mean_loss'] < 3.28, 'Pilot does not support the planned candidate'
assert s['speedup_percent'] > 0, 'Pilot does not show a speed advantage'
PY
python run_campaign_next.py certify > certify_campaign.log 2>&1
python summarize_campaign_next.py campaigns/certify > certify_summary.log
python run_campaign_next.py short_control > short_control_campaign.log 2>&1
python summarize_campaign_next.py campaigns/short_control > short_control_summary.log
python run_campaign_next.py accepted > accepted_campaign.log 2>&1
python summarize_campaign_next.py campaigns/accepted > accepted_summary.log
date -u > CERTIFICATION_PIPELINE_COMPLETE
