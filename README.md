# Hyperbolic nanoGPT

This project explores the benefits of hyperbolic geometry in language models by modifying various components of nanoGPT. The hypothesis is that certain NLP tasks and relationships might be better represented in hyperbolic rather than Euclidean space.

### Current Modifications

The following components can be switched between Euclidean and hyperbolic versions:

1. Language Model Head (`model/model.py` -> `LorentzMLR` class)
2. Attention Layer (file: `model/model.py` -> `HyperbolicSelfAttention` class)
TODO: 3. Embeddings 

### Installation
git clone https://github.com/Alex2034/hyp-nanogpt
cd hyp-nanogpt
conda env create -f env.yaml
conda activate hyp-nanogpt


### Experiment Scripts

For convenience, we provide shell scripts to run multiple experiments:

1. `run_hyp.sh` - Runs experiments with adjustable components
2. `run_euc.sh` - Runs baseline Euclidean experiments
3. `run_single.sh` - Useful for single experiment runs

### Key Parameters

- `head_mode`: Choose between 'hyp' (Hyperbolic) or 'euc' (Euclidean) for the LM head
- `attn_mode`: Choose between 'hyp' or 'euc' for the attention 
- `curvature`: Initial curvature value for hyperbolic space (if using hyperbolic components)
- `k_lr`: Learning rate for the curvature parameter (set to 0 to keep curvature fixed)


