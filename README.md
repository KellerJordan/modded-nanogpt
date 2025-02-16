# Hyperbolic nanoGPT

Forked from [kellerjordan/nanoGPT](https://github.com/kellerjordan/nanoGPT)

This project explores the benefits of hyperbolic geometry in language models by modifying various components of nanoGPT. The hypothesis is that relationships in languagemight be better represented in hyperbolic rather than Euclidean space.

### Current Modifications

The following components can be switched between Euclidean and hyperbolic versions:

##### Language Model Head (`model/model.py` -> `LorentzMLR` class)
(currently outperforms the original slightly but has to be studied more)

##### Attention Layer (file: `model/model.py` -> `HyperbolicSelfAttention` class)
(currently unstable but learns some curvatures)

##### TBD: 
Embeddings 

### Installation
```bash
git clone https://github.com/Alex2034/hyp-nanogpt
cd hyp-nanogpt
conda env create -f env.yaml
conda activate hyp-nanogpt
```

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

### Acknowledgements

- [kellerjordan/nanoGPT](https://github.com/kellerjordan/nanoGPT) for the baseline implementation
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) for the original nanoGPT 
- [kschwethelm/HyperbolicCV](https://github.com/kschwethelm/HyperbolicCV/tree/main/code) for the LorentzMLR code
- [geoopt](https://github.com/geoopt/geoopt) for the Riemannian optimization code
