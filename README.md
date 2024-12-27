# Replicating ESM2 at the speed of sound
This repo is an open-source collaboration to reproduce ESM2-150M validation loss in as little time as possible inspired by the fantastic [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repo.

## Quick Start

Setup environment and train ESM2

```
git clone https://github.com/Synthyra/SpeedRunningESM2
cd SpeedRunningESM2
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20241203+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade # install torch 2.6.0
python data/download_omgprot50.py --num_chunks 10 # downloads only the first 1.0B training tokens to save time
./run.sh
```

## Benchmarks to match
[OMGprot50](https://huggingface.co/datasets/Synthyra/omg_prot50) validation and test sets, 15% BERT-like MLM objective.
Loss is standard cross-entropy loss, perplexity $e^{loss}$. Sequence reconstruction metrics are calculated via exact match betweeen predictions and labels.

Validation set, random 10,000 sequences from OMGprot50.
|model    |loss  |perplexity|precision|recall|f1    |accuracy|mcc   |
|---------|------|----------|---------|------|------|--------|------|
|ESM2-8M  |2.4662|11.7775   |0.3074   |0.278 |0.2726|0.278   |0.2262|
|ESM2-35M |2.3572|10.5613   |0.3464   |0.3205|0.3161|0.3205  |0.2726|
|ESM2-150M|2.255 |9.5349    |0.3806   |0.3596|0.356 |0.3596  |0.3152|
|ESM2-650M|2.1382|8.4841    |0.4218   |0.4024|0.4   |0.4024  |0.3615|


Test set, random 10,0000 sequences from OMGprot50 and 3,000+ newly discovered sequences after OMGprot50 creation (well after ESM2 training date).
|model    |loss  |perplexity|precision|recall|f1    |accuracy|mcc   |
|---------|------|----------|---------|------|------|--------|------|
|ESM2-8M  |2.452 |11.6116   |0.3079   |0.278 |0.2735|0.278   |0.2274|
|ESM2-35M |2.3063|10.0374   |0.3616   |0.338 |0.3346|0.338   |0.2928|
|ESM2-150M|2.1587|8.6602    |0.4149   |0.3973|0.3949|0.3973  |0.3568|
|ESM2-650M|1.998 |7.3743    |0.4723   |0.4576|0.4561|0.4576  |0.4217|


These match the [results](https://github.com/Synthyra/SpeedRunningESM2/pull/2#issue-2756280840) from the original paper well.
