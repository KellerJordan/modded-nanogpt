# Replicating ESM2 at the speed of sound
This repo is an open-source collaboration to reproduce ESM2-150M validation loss in as little time as possible inspired by the fantastic [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repo.

## Quick Start

Setup environment and train ESM2

```
git clone https://github.com/Synthyra/SpeedRunningESM2 && cd SpeedRunningESM2
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20241203+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade # install torch 2.6.0
python data/cached_omgprot50.py 10 # downloads only the first 1.0B training tokens to save time
./run.sh
```

## Benchmarks to match
[OMGprot50](https://huggingface.co/datasets/Synthyra/omg_prot50) validation and test sets, 15% BERT-like MLM objective.
Loss is standard cross-entropy loss, perplexity $e^{loss}$. Sequence reconstruction metrics are calculated via exact match betweeen predictions and labels.

Validation set, random 10,000 sequences from OMGprot50.
|model                       |loss  |perplexity|precision|recall|f1    |accuracy|mcc   |
|----------------------------|------|----------|---------|------|------|--------|------|
|ESM2-8M   |2.3071|10.0454   |0.3547   |0.3077|0.3055|0.3077  |0.2578|
|ESM2-35M |2.2062|9.0809    |0.3868   |0.3494|0.3471|0.3494  |0.3036|
|ESM2-150M|2.1242|8.3662    |0.416    |0.3855|0.3847|0.3855  |0.3430|
|ESM2-650M|2.0061|7.4342    |0.456    |0.4321|0.4311|0.4321  |0.3930|

Test set, random 10,0000 sequences from OMGprot50 and 3,000+ newly discovered sequences after OMGprot50 creation (well after ESM2 training date).
| model      | loss         | perplexity | precision | recall |  f1    | accuracy |  mcc   |
|------------|--------------|------------|-----------|--------|--------|----------|--------|
| ESM2-8M    | 2.3388       | 10.3688    | 0.3445    | 0.3007 | 0.2979 | 0.3007   | 0.2515 |
| ESM2-35M   | 2.2094       | 9.1102     | 0.3877    | 0.3537 | 0.3517 | 0.3537   | 0.3093 |
| ESM2-150M  | 2.0982       | 8.1515     | 0.4316    | 0.4058 | 0.4049 | 0.4058   | 0.3657 |
| ESM2-650M  | 1.9609       | 7.1057     | 0.4796    | 0.4578 | 0.4575 | 0.4578   | 0.4217 |

These match the [results](https://github.com/Synthyra/SpeedRunningESM2/pull/2#issue-2756280840) from the original paper well. As expected, the test set appears slightly harder than our validation set or the ESM2 evaluation sets.
