# Replicating ESM2 at the speed of sound
This repo is an open-source collaboration to reproduce ESM2-150M validation loss in as little time as possible inspired by the fantastic [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repo. 

## Benchmarks to beat
[OMGprot50](https://huggingface.co/datasets/Synthyra/omg_prot50) test set, 15% BERT-like MLM objective.
Loss is standard cross-entropy loss, perplexity $e^{loss}$. Sequence reconstruction metrics are calculated via exact match betweeen predictions and labels.
| model      | loss         | perplexity | precision | recall |  f1    | accuracy |  mcc   |
|------------|--------------|------------|-----------|--------|--------|----------|--------|
| ESM2-8M    | 2.3388       | 10.3688    | 0.3445    | 0.3007 | 0.2979 | 0.3007   | 0.2515 |
| ESM2-35M   | 2.2094       | 9.1102     | 0.3877    | 0.3537 | 0.3517 | 0.3537   | 0.3093 |
| ESM2-150M  | 2.0982       | 8.1515     | 0.4316    | 0.4058 | 0.4049 | 0.4058   | 0.3657 |
| ESM2-650M  | 1.9609       | 7.1057     | 0.4796    | 0.4578 | 0.4575 | 0.4578   | 0.4217 |

These match the [results](https://github.com/Synthyra/SpeedRunningESM2/pull/2#issue-2756280840) from the original paper well, but our evaluation set appears slightly harder.
