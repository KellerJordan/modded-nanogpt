# Replicating ESM2 at the speed of sound
This repo is an open-source collaboration to reproduce ESM2 models with the same or less parameters in as little time as possible, inspired by the fantastic [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repo. Mostly interested in 8xH100 or 1xH200 runs which are currently available through many vendors.

## Quick Start

Setup environment and train ESM2

```
git clone https://github.com/Synthyra/SpeedRunningESM2
cd SpeedRunningESM2
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20241203+cu124 torchvision==0.20.0dev20241204 --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade
python data/download_omgprot50.py # --num_chunks 10 you can download less chunks to save time
./run.sh
```
torchvision is needed to fix an import error with transformers.

## Benchmarks to match
[OMGprot50](https://huggingface.co/datasets/Synthyra/omg_prot50) validation and test sets, 15% BERT-like MLM objective.
Loss is standard cross-entropy loss, perplexity $e^{loss}$. [Sequence reconstruction metrics](https://github.com/Synthyra/SpeedRunningESM2/blob/master/benchmark_esm.py) are calculated via exact match betweeen predictions and labels and weighted averages.

Validation set, random 10,000 sequences from OMGprot50.
| model | loss &darr;   | perplexity &darr; | precision &uarr; | recall &uarr; | f1 &uarr; | accuracy &uarr; | mcc &uarr;|
|-----------|--------|------------|-----------|--------|--------|----------|--------|
| ESM2-8M   | 2.4662 | 11.7775    | 0.3074    | 0.2780 | 0.2726 | 0.2780   | 0.2262 |
| ESM2-35M  | 2.3572 | 10.5613    | 0.3464    | 0.3205 | 0.3161 | 0.3205   | 0.2726 |
| ESM2-150M | 2.2550 | 9.5349     | 0.3806    | 0.3596 | 0.3560 | 0.3596   | 0.3152 |
| ESMC-300M | 2.1996 | 9.0214     | 0.3936    | 0.3648 | 0.3605 | 0.3648   | 0.3206 |
| ESMC-600M | 2.1549 | 8.6267     | 0.4068    | 0.3802 | 0.3762 | 0.3802   | 0.3373 |
| ESM2-650M | 2.1382 | 8.4841     | 0.4218    | 0.4024 | 0.4000 | 0.4024   | 0.3615 |

Test set, random 10,0000 sequences from OMGprot50 and 3,000+ newly discovered sequences after OMGprot50 creation (well after ESM2 training date).
| model | loss &darr; | perplexity &darr; | precision &uarr; | recall &uarr; | f1 &uarr; | accuracy &uarr; | mcc &uarr;|
|-----------|--------|------------|-----------|--------|--------|----------|--------|
| ESM2-8M   | 2.4520 | 11.6116    | 0.3079    | 0.2780 | 0.2735 | 0.2780   | 0.2274 |
| ESM2-35M  | 2.3063 | 10.0374    | 0.3616    | 0.3380 | 0.3346 | 0.3380   | 0.2928 |
| ESM2-150M | 2.1587 | 8.6602     | 0.4149    | 0.3973 | 0.3949 | 0.3973   | 0.3568 |
| ESMC-300M | 2.0523 | 7.7854     | 0.4549    | 0.4296 | 0.4278 | 0.4296   | 0.3916 |
| ESMC-600M | 1.9942 | 7.3466     | 0.4741    | 0.4516 | 0.4498 | 0.4516   | 0.4152 |
| ESM2-650M | 1.9980 | 7.3743     | 0.4723    | 0.4576 | 0.4561 | 0.4576   | 0.4217 |

These match the [results](https://github.com/Synthyra/SpeedRunningESM2/pull/2#issue-2756280840) from the original paper well.


## Successful runs showcase

|~Matches |Parameters|Time      |Hardware |Log | Val loss | Test loss |
|--------|----------|----------|---------|----|-----------|-----------|
|ESM2-150|140M      |9.44 hours|1 x GH200|[Link](https://github.com/Synthyra/SpeedRunningESM2/blob/master/logs/f48932cb-f41f-4c0c-8f24-90c839e9dc9e.txt)| 2.2272 | Bugged |
|ESM2-150|132M      |9.00 hours|1 x GH200|[Link](https://github.com/Synthyra/SpeedRunningESM2/blob/master/logs/e631bf18-f202-492b-a3b8-fbae2cb7484a.txt)| 2.2137 | 2.2093 |
|ESM2-650|132M      |45.16 hours|1 x GH200|[Link](https://github.com/Synthyra/SpeedRunningESM2/blob/master/logs/a0a3dc4e-6f27-43e0-96fb-b1c2372a164b.txt)| 2.1044 | 2.1058 |




