# Modded-NanoGPT Optimization Benchmark

The goal of this benchmark is to collaboratively|competitively find efficient neural network optimizers.
Unlike the main NanoGPT speedrun which seeks to minimize *wallclock time* by any means, here we aim to minimize *step count* by improving the optimization algorithm (⇒ methods that are slow in terms of wallclock are perfectly OK).

[Longform announcement](https://x.com/kellerjordan0/status/2049193527440187494)

Thank you to everyone who's contributed results so far:
[@kaiyue-wen](https://github.com/kaiyue-wen), [@nilin](https://github.com/nilin), [@alint77](https://github.com/alint77), [@wilsoncwu](https://github.com/wilsoncwu), [@kumarkrishna](https://github.com/kumarkrishna), [@lliu606](https://github.com/lliu606), [@zhenghaoxu-gatech](https://github.com/zhenghaoxu-gatech), [@bentherien](https://github.com/bentherien), [@Sam_Acqua](https://x.com/Sam_Acqua), [@zhehangdu](https://github.com/zhehangdu), [@SPThole](https://github.com/SPThole), [@liyang2019](https://github.com/liyang2019), [@zzp1012](https://github.com/zzp1012), Yash Pande, [@fhueb](https://github.com/fhueb), [@kcc-lion](https://github.com/kcc-lion), [@zhiweixx](https://github.com/zhiweixx), [@chenchenygu](https://github.com/chenchenygu), [@breskanu](https://github.com/breskanu), [@fangzhou_wu](https://x.com/fangzhou_wu), [@eliebak](https://github.com/eliebak), [@wakamex](https://github.com/wakamex), [@varunneal](https://github.com/varunneal), [@tomoqt](https://github.com/tomoqt), [@rohan-anil](https://github.com/rohan-anil), [@konstmish](https://github.com/konstmish), [@jn2clark](https://github.com/jn2clark), [@OscarYau525](https://github.com/OscarYau525), [@ypwang61](https://github.com/ypwang61), and [@nooraovo](https://github.com/nooraovo).

**Table of Contents**

- [Benchmark definition tl;dr](#benchmark-definition-tldr)
- [Quickstart](#quickstart)
- [Notable results history](#notable-results-history)
- [Active techniques in current record](#active-techniques-in-current-record)
- [Rules](#rules)
  - [Freedoms](#freedoms)
  - [Skeptical results](#skeptical-results)
  - [Pairwise statistical significance](#pairwise-statistical-significance)
- [Motivation](#motivation)
- [Addressing a potential critique](#addressing-a-potential-critique)
- [Details on relation to the main speedrun](#details-on-relation-to-the-main-speedrun)
- [Guidelines](#guidelines)
- [Citation](#citation)


## Benchmark definition tl;dr

Runs submitted to this benchmark are considered valid if they meet the following conditions:
* Runs must not modify the dataset, batch size, or architecture used by the baseline. Runs also must not perform more than one forward-backward pass per step.
* Runs must attain below 3.28 val loss, thereby matching [Andrej Karpathy's GPT-2 replication](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).

Beyond those constraints, runs are completely free to arbitrarily modify the optimization algorithm and hyperparameters.
The general goal is to achieve 3.28 in the fewest steps possible. But we also accept academic results
which add to our knowledge of neural network optimization without setting a new record
(e.g., baselines for optimizers from the literature).

The precise rules around statistical significance etc. can be found [below](#rules).


## Quickstart

The baseline setup (tuned Muon with aux AdamW = result #36) can be run using the following command on any {1,2,4,8}x-{A100,H100} machine:
```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install torch==2.11 huggingface_hub
python data/cached_fineweb10B.py 20  # 2B tokens, which is sufficient for 4000 steps. increase if needed.
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) records/track_3_optimization/train_gpt_simple.py
```

Note: [Beware that](https://github.com/KellerJordan/modded-nanogpt/issues/268) on A100, using `torch==2.10` with `torch.compile` enabled will lead to `nan`s.

## Notable results history

| # | Steps to 3.28 | Evidence | Description | Date | Log | Contributors |
| - | - | - | - | - | - | - |
| 1 | 3600(!) | 3.2777 (n=1)Ⓧ | [Muon](https://kellerjordan.github.io/posts/muon/) with aux Adam, lr=.02 wd=.01 | 2026/04/26 | [log](results/7b8270c5-a9cd-4a73-b7d8-5d86a2d1e428.txt) | [@kellerjordan0](https://x.com/kellerjordan0) |
| 2 | 5625 | 3.2790 (n=1)Ⓧ | [Adam](https://arxiv.org/abs/1412.6980) lr=0.0015 betas=(0.9, 0.95) warmup_steps=250 (note: this is most likely undertuned) | 2026/04/26 | [log](results/a63a68d1-24aa-4a22-af9a-224e43209ea4.txt) | [@kellerjordan0](https://x.com/kellerjordan0) |
| 3 | 3500(!) | 3.2767 (n=1)Ⓧ | Muon with aux Adam, lr=.025 wd=.0125 | 2026/04/26 | [log](results/311d7833-8dfc-43ea-a55c-fd313a11c4a8.txt) | [@kellerjordan0](https://x.com/kellerjordan0) |
| 4 | 4875 | 3.2741 (n=5)✓ | [AdamH](https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd) (Adam preconditioning + hyperball constraint on hidden matrices) with per-module init std (attn.proj std=.026, mlp.proj std=.031, mlp.fc std=.031, qkv default), lr=.018 betas=(0.9, 0.95) warmup_steps=250 h_cooldown_frac=1.0 aux_cooldown_frac=.4 | 2026/04/30 | [log](results/20260430_adamh/7533dd87-107f-4a4f-8229-acbec0fb00ac.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/272) by [@kaiyue-wen](https://github.com/kaiyue-wen) (Hyperball author) |
| 5 | 3325(!) | 3.2782 (n=10)✓ | [MuonH](https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd) (Muon + hyperball constraint on hidden matrices) with per-module init std (attn.proj std=.026, mlp.proj std=.031, mlp.fc std=.031, qkv default), lr=.018 h_cooldown_frac=1.0 aux_cooldown_frac=.4 | 2026/04/30 | [log](results/20260430_muonh/9319c798-6643-464a-b407-b05468e468f5.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/267) by [@kaiyue-wen](https://github.com/kaiyue-wen) |
| 6 | 3375 | 3.2788 (n=20)✓ | Muon with aux Adam, lr=.025 wd=.025 | 2026/05/01 | [log](results/51ece938-03c5-4343-8dcc-3f3336b07008.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/271) by [@nilin](https://github.com/nilin) and [@alint77](https://github.com/alint77) |
| 7 | 3325 | 3.2752 (n=1)✓ | [Muon²](https://arxiv.org/abs/2604.09967) with aux Adam, lr=.10 wd=.0125 β₂=.95 ε=1e-10 | 2026/04/29 | [log](results/20260501_muonsq/bb903816-ea27-4f5f-8028-c963d38c6a7f.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/266) by [@wilsoncwu](https://github.com/wilsoncwu) |
| 8 | 3250 | 3.2778 (n=10)✓ | [NorMuon](https://arxiv.org/abs/2510.05491)[H](https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd) (Muon NS direction + Adafactor-style row/col variance preconditioning, then hyperball constraint on hidden matrices) with per-module init std (attn.proj std=.026, mlp.proj std=.031, mlp.fc std=.031, qkv default), lr=.018 mu=0.95 beta2=0.95 h_cooldown_frac=1.0 aux_cooldown_frac=.4, end 25 steps early | 2026/04/30 | [log](results/20260430_normuonh/f45b5dcf-16bb-4e83-b5c7-4ef4981f0e9f.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/273) by [@kaiyue-wen](https://github.com/kaiyue-wen) |
| 9 | 3250(!) | 3.2771 (n=8)✓ | NorMuon with aux Adam + u/w-floor (wd-free strategy that clamps ‖u‖\_F / ‖w‖\_F to 0.35), lr=.0375 | 2026/04/29 | [log](results/20260501_skylight001/f78af80a-2ba3-4cf7-b9f7-e6e56ff2c54d.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/274) by [@kumarkrishna](https://github.com/kumarkrishna) |
| 10 | 3250 | 3.2789 (n=20)✓ | NorMuon lr=0.035 wd=0.025, end 50 steps early | 2026/05/03 | [log](results/20260503_normuon/e0d0185f-ccb8-426d-8265-a4e762ec69f6.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/276) by [@lliu606](https://github.com/lliu606) (NorMuon author) and [@zhenghaoxu-gatech](https://github.com/zhenghaoxu-gatech) |
| 11 | 3225(!) | 3.2785 (n=16)✓ | Setup from #9 plus [Contra-Muon](https://github.com/nilin/contra-muon) technique (note: p-value vs. #9 is p=0.69) | 2026/05/01 | [log](results/20260501_contra_muon/08cd60f9-99e2-4e28-b1ac-19136dd42a05.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/275) by [@nilin](https://github.com/nilin) |
| 12 | 3325 | 3.2790 (n=20)✓ | Muon with aux Adam, lr=.035 wd=.025 following #10, end 25 steps early following #8 | 2026/05/03 | [log](results/1bd8db7a-f3a3-4195-856d-cab7e0816443.txt) | [@kellerjordan0](https://x.com/kellerjordan0) |
| 13 | 3210(!) | 3.2785 (n=10)✓ | NorMuonH (#8) wrapped in [MuLoCo](https://arxiv.org/abs/2502.07314)-style outer Nesterov SGD (Algorithm 1, K=1) over all trainable params, outer_lr=0.7 outer_momentum=0.5 sync_interval=30 (= 107 outer steps) (note: p-value vs. #11 is p=0.099, and vs. #8 is p=0.029) | 2026/05/04 | [log](results/20260504_muloco_normuonh/7fba9434-58d8-4166-b6a7-d62ef8d17e5d.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/277) by [@bentherien](https://github.com/bentherien) |
| 14 | 3150(!) | 3.2776 (n=4)✓ | Setup from #11, plus SOAP preconditioning before Muon orthogonalization, as in [SOAP-Muon](https://nikhilvyas.github.io/SOAP_Muon.pdf), for the MLP weights. | 2026/05/04 | [log](results/20260504_contra_muon_mlp_soapish/0248394b-0d6c-4133-9ff7-e7ff2763cdd9.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/278) by [@Sam_Acqua](https://x.com/Sam_Acqua) |
| 15 | 3275 | 3.2785 (n=15)✓ | [Newton-Muon](https://arxiv.org/abs/2604.01472) with activation-covariance right-preconditioning refreshed every 64 steps before the Muon Newton-Schulz update ([details](results/20260505_newton_muon/README.md)); tuned lr/wd per param type | 2026/05/05 | [log](results/20260505_newton_muon/6fb302c7-d271-491b-906f-75cd6ec72075.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/281) by [@zhehangdu](https://github.com/zhehangdu) (Newton-Muon author) |
| 16 | 3125(!) | 3.2784 (n=8)✓ | Setup from #14, plus SOAP precond for attention with trust gate (note: p-value vs. #14 is p=0.34) | 2026/05/05 | [log](results/20260506_trustlight/fake_log_from_seed0.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/283) by [@SPThole](https://github.com/SPThole) |
| 17 | 3175 | 3.2789 (n=20)✓ | Setup from #11, plus [Aurora](https://github.com/tilde-research/aurora-release) | 2026/05/06 | [log](results/20260505_aurora/298f02bc-dbb4-4661-9ad8-f6429d532873.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/284) by [@liyang2019](https://github.com/liyang2019) (Aurora author) |
| 18 | 3225 | 3.2776 (n=9)✓ | [PMuon](results/20260507_pmuon/README.md) (Muon + bilateral streaming covariance power preconditioning), lr=.035 wd=.025 γ=.3 β=.95 | 2026/05/07 | [log](results/20260507_pmuon/54fc0541-7a62-4772-a8f8-d3a46ad10dba.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/285) by [@zzp1012](https://github.com/zzp1012) |
| 19 | 3125 | 3.2780 (n=6)✓ | Setup from #8, with NorMuonH replaced by [KL-SOAP](https://arxiv.org/abs/2509.03378) with hyperball optimization, precondition_frequency=1, lr=.018, beta1=.95, beta2=.9, shampoo_beta=.9 | 2026/05/08 | [log](results/20260508_klsoap_h_clean_tuple_sweep/b1095_sh090/klsoap-h-b1095_sh090-K3125-seed-1.full.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/290) by [@kaiyue-wen](https://github.com/kaiyue-wen) |
| 20 | 3030(!) | 3.2790 (n=30)✓ | Setup from #16, plus PowerCool lr schedule from Yash Pande, plus interpolation between Contra-Muon and new method Soft-Muon | 2026/05/09 | [log](results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/291) by Yash Pande and [@nilin](https://github.com/nilin) |
| 21 | 4100 | 3.2776 (n=4)✓ | [Shampoo](https://arxiv.org/abs/2002.09018)(lr=0.0015, wd=0.2, betas=(.9, .95), precond_freq=5, power=-1/4) | 2026/05/13 | [log](results/20260513_shampoo_1_4_power/503575c5-6dde-425a-b461-2df4d99db974.txt) | [@kellerjordan0](https://x.com/kellerjordan0) |
| 22 | 8225 | 3.2774 (n=4)✓ | SpectralDescent(lr=.03, wd=.015) == Muon(lr=.03, wd=.015, mu=0) == Shampoo(lr=0.03, wd=0.015, betas=(0, 0), precond_freq=1, power=-1/4) | 2026/05/17 | [log](results/20260517_ortho/d5098d67-7c1b-47b4-8833-80960d633d33.txt) | [@kellerjordan0](https://x.com/kellerjordan0) |
| 23 | 3075 | 3.2790 (n=30)✓ | [Muown](https://arxiv.org/abs/2605.10797) (Muon with integrated row-norm control), direction_scale=0.2, PowerCool, V-norm schedule | 2026/05/10 | [log](results/20260508_muown/a26c8aa2-993d-443f-b931-845724d07015.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/288) by [@fhueb](https://github.com/fhueb) and [@kcc-lion](https://github.com/kcc-lion) (Muown authors) |
| 24 | 3175 | 3.2782 (n=10)✓ | Setup from #11, with split LR cooldown: aux Adam cooldown_frac=0.4 and matrix cooldown_frac=0.8 | 2026/05/09 | [log](results/20260509_contra_muon_split_cooldown/c1af0bd1-6999-44d1-a618-3d1234ea32f0.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/292) by [@zhiweixx](https://github.com/zhiweixx) |
| 25 | 3040 | 3.2781 (n=5)✓ | Setup from #19 KL-SOAP-H, plus PowerCool with nonzero Adam/KL-SOAP LR floors | 2026/05/11 | [log](results/20260511_klsoap_h_lr_power_decay/01906576-8b3d-4bd2-a73f-23997f602ec1.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/293) by [@chenchenygu](https://github.com/chenchenygu) |
| 26 | 3090 | 3.2785 (n=10)✓ | SinkSOAP: Gram-Sinkhorn SOAP-style preconditioning with NorMuon postconditioner, lr=0.04 wd=0.025, does not use PowerCool | 2026/05/14 | [log](results/20260514_sinksoap/d0155dd0-f77d-48a9-8eb4-453f894b9476.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/298) by [@lliu606](https://github.com/lliu606) and [@zhenghaoxu-gatech](https://github.com/zhenghaoxu-gatech) |
| 27 | 3125 | 3.2782 (n=6)✓ | Setup from #19, with KL-SOAP replaced by [SOAP](https://proceedings.iclr.cc/paper_files/paper/2025/file/e988664070e9591f93fdcf605f7dc623-Paper-Conference.pdf) with hyperball optimization, w.o. bias correction, precondition_frequency=1, lr=.018, beta1=.95, beta2=.9, and tuned lr schedule | 2026/05/18 | [log](results/20260518_soaph/SOAPH_run1.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/302) by Nikita Breskanu [@breskanu](https://github.com/breskanu) |
| 28 | 3175 | 3.2790 (n=25)✓ | [DynMuon](https://arxiv.org/pdf/2605.17109) (p: 0.25 -> -0.25, tau=0.04, w=0.04, lr=0.02, wd=0.025) | 2026/05/19 | [log](results/20260519_dynmuon/50172610-d038-4f90-9a12-b9a0853f035d.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/304) by [@fangzhou_wu](https://x.com/fangzhou_wu) |
| 29 | 2990(!) | 3.2787 (n=11)✓ | Setup from #20, plus radial brake: dampens outward radial updates before the u/w floor | 2026/05/11 | [log](results/20260511_dampen_radial_gradient_component/00882c75-914d-4340-8e0b-dcffcb18b73d.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/294) by [@nilin](https://github.com/nilin) |
| 30 | 2930(!) | 3.2784 (n=16)✓ | Setup from #29, plus Aurora row-balanced polar on wide `mlp.proj` matrices, Soft-Muon/NorMuon-lite disabled, Contra-Muon ramp extended to step 2500, and Muon momentum warmup/cooldown from track 1 | 2026/05/14 | [log](results/20260514_aurora_proj_pruned_extended_contra/d198124d-5e7f-4743-a683-0eb936a40dbe.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/300) by [@eliebak](https://github.com/eliebak) |
| 31 | 2995 | 3.2789 (n=20)✓ | Setup from #23 Muown, plus NorMuon & Contra-Muon | 2026/05/15 | [log](results/20260515_contranormuown/0f117ad6-2342-40f0-9e00-20cec55f6021.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/301) by [@fhueb](https://github.com/fhueb) and [@kcc-lion](https://github.com/kcc-lion) (Muown authors) |
| 32 | 3000 | 3.2778 (n=9)✓ | Setup from #20, plus a SODA-style hidden-matrix anchor correction toward initialization, faded out with a cosine schedule from step 2000 to 2750 | 2026/05/18 | [log](results/20260518_soda_fade_3000_n9/README.md) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/303) by [@wakamex](https://github.com/wakamex) |
| 33 | 3375 | 3.2779 (n=5)✓ | [PSGD](https://arxiv.org/abs/2402.04553) with Kronecker whitening preconditioners, Hyperball, lr=.025 linearly decayed over the full run, precond_lr=1.0, beta=0.95 | 2026/05/28 | [log](results/20260527_psgd/README.md) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/316) by [@varunneal](https://github.com/varunneal) |
| 34 | 2925(!) | 3.2781 (n=8)✓ | Setup from #30, plus late capped RRE vector extrapolation from step 2820 to 2925 with k=4, damping=.875, and max relative update=.001 (note: pairwise p-value vs. #30 is p=0.168) | 2026/05/20 | [log](results/20260520_rre_extrapolation_pr300_2925/README.md) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/305) by [@tomoqt](https://github.com/tomoqt) |
| 35 | 3375 | 3.2767 (n=2)✓ | One-Sided Shampoo with pseudoinverse root preconditioner, Adam grafting, precondition_frequency=1, shampoo_epsilon=0, adam_eps=1e-15, lr=.01, wd=.1, beta2=.9 | 2026/06/10 | [log](results/sh-origpinv-s3375-lr1em2-wd010-b9em1-ge15-pf1-near1-record-130563a2-b40b-43c1-8fb4-b7a3bbfa5969.txt) | [@rohan-anil](https://github.com/rohan-anil) |
| 36 | 3250 | 3.2787 (n=10)✓ | Tuned baseline Muon + aux AdamW hyperparameters: Adam embed/proj/1D lr=.7/.004/.015 wd=0.001, Muon lr=.025 wd=.05 | 2026/06/11 | [log](results/20260610_tuned_baseline_3250/263ea3c4-2b13-4adf-8a71-0410386b20e1.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/323) by [@konstmish](https://github.com/konstmish) |
| 37 | 3250 | 3.2786 (n=10)✓ | MuonH (#5) with aux Adam hyperparameters grafted from the tuned baseline in result #36 and re-tuned for MuonH: embed lr=.91, head lr=.0064, scalar lr=.0195, aux cooldown_frac=.85 (matrix cooldown_frac=1.0 and lr=.018 unchanged) | 2026/06/11 | [log](results/20260611_muonh_tuned_aux_3250/f6649181-6de2-4d67-b243-2b18adb9d594.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/324) by [@kaiyue-wen](https://github.com/kaiyue-wen) |
| 38 | 2900(!) | 3.2786 (n=9)✓ | Setup from #30 with techniques to improve late-training: Final step moves weights backward along update-EMA(0.9), extended Contra-Muon transition, Tempered-Polar (similar to Contra-Muon), ramp-up radial brake/u-w floor | 2026/05/20 | [log](results/20260520_tail_refinterp_2900/refinterp_2900_seed0.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/307) by [@jn2clark](https://github.com/jn2clark) |
| 39 | 3125 | 3.2786 (n=20)✓ | Setup from #12, wrapped in [EMA-Nesterov](https://arxiv.org/abs/2605.25395) with lookahead EMA γ=.99 and scheduled lookahead β=.6·lr/max_lr from steps 500 to 2400 | 2026/05/22 | [log](results/20260522_ema_nesterov_muon/4d9e83d4-4e3f-43d3-8aae-e3a9031586be.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/308) by [@OscarYau525](https://github.com/OscarYau525) (EMA-Nesterov author) |
| 40 | 2890(!) | 3.2788 (n=16)✓ | Setup from #30, wrapped in [EMA-Nesterov](https://arxiv.org/abs/2605.25395) with lookahead EMA γ=.99 and scheduled lookahead β=.3·lr/max_lr from steps 300 to 1950 | 2026/05/22 | [log](results/20260522_ema_nesterov_aurora_proj/345b51b2-38ee-4a78-af2c-bb96cb68470d.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/309) by [@OscarYau525](https://github.com/OscarYau525) (EMA-Nesterov author) |
| 41 | 2875(!) | 3.2790 (n=20)✓ | Setup from #40, plus Circuit-Muon coupling on attention V/O pairs: per-head partner-scalar post-whitening and trace-only gauge rebalance | 2026/05/23 | [log](results/20260523_circuit_muon/README.md) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/311) by [@liyang2019](https://github.com/liyang2019) (Circuit-Muon author) |
| 42 | 2860(!) | 3.2789 (n=16)✓ | Setup from #40 with zero-init biases, and fixed reference interpolation: capture reference weights at step 2375 and report with `gamma=-0.075` from step 2850 onward. Also got rid of rademacher init from #30 | 2026/05/25 | [log](results/20260525_aurora_ema_ref/README.md) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/312) by [@chenchenygu](https://github.com/chenchenygu) |
| 43 | 2850(!) | 3.2786 (n=13)✓ | Setup from #41, plus fixed late trajectory transforms: BroadDelta on `muon_other`, TrailDelta endpoint pulses, normalized orthogonal phase readout on `muon_other`, and an 8% final readout toward the step-2400 non-embedding anchor | 2026/05/29 | [log](results/20260529_tail_phase_readout_2850/tail_phase_readout_2850_seed0.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/318) by [@jn2clark](https://github.com/jn2clark) |
| 44 | 2750(!) | 3.2789 (n=20)✓ | Setup from #41, plus SOAP-Muon on all hidden matrices w/ precondition_frequency=1 (prev. had SOAP on MLP + attn.proj w/ precond_freq=1), tune auxiliary β2's, double mu cooldown, set rademacher init CGI α=.125, and remove neutral geometry modules including (Circuit,Contra)Muon and Aurora | 2026/06/10 | [log](results/20260609_soap_f1_auxb2_clean/H100_ff29b392-e7b7-453e-b9d5-a7dfe0605dd0.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/321) by [@ypwang61](https://github.com/ypwang61) and [@nooraovo](https://github.com/nooraovo) |
| 45 | 2720(!) | 3.2786 (n=10)✓ | Setup from #44, plus at the final step blend the weights towards EMA(horizon = 150 steps) | 2026/06/12 | [log](results/20260611_tailema_2720_submission/8878c81f-5f73-461f-a41e-c0887e15c1ca.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/325) by [@jn2clark](https://github.com/jn2clark) |
| 46 | 2690(!) | 3.2783 (n=8)✓ | Setup from #45, plus RowUpdateFloor per-output-row u/w-floor, and Cautious Weight Decay `CWD=0.025` | 2026/06/19 | [log](results/20260619_cwd_rowfloor_tailema/A40_seed0_5c87fa44-7ca7-4d54-971d-d952f9b15792.txt) | [PR](https://github.com/KellerJordan/modded-nanogpt/pull/328) by [@ypwang61](https://github.com/ypwang61) |


Notes:
* To reproduce any of these runs, simply rip their python script out of their logfile (take everything before `===`), and then run it using the quickstart above.
If it fails to reproduce (i.e., there's an error or we get statistical evidence that its mean is above 3.28), then please raise an issue to let us know, as this will be grounds to remove the run from the history.
* The number in the leftmost column reflects the order in which these runs were accepted. This does not necessarily line up with the Date column, which is the date at which the PR appeared.
* The (!) symbol next to the step count indicates a new world record.


## Active techniques in current record

Codex offers the following description of the techniques used in the current record (#46).
Note that it is not entirely known which of these techniques is most beneficial.
Several techniques that were considered beneficial in the past have now been abandoned.

Record #46 is: **#44’s clean SOAP-Muon stack, plus Tail-EMA eval readout, RowFloor, and post-pin Cautious Weight Decay**.

Active techniques:

1. **Standard [Muon](https://kellerjordan.github.io/posts/muon/) core on hidden matrices**:
   Hidden 2D block weights are optimized with Muon: momentum update, Newton-Schulz orthogonalization, then aspect-ratio scaling.
   (Introduced in result #1.)

2. **[SOAP-Muon](https://nikhilvyas.github.io/SOAP_Muon.pdf) on all hidden matrices**:
   Before Muon orthogonalization, the momentum update is preconditioned using SOAP-style row/column gradient covariance statistics. It uses `precondition_frequency=1` and `beta2=.90`. For `attn.proj`, the SOAP direction is gated by agreement with raw momentum / gradient alignment. The blend preserves the raw update norm.
   (MLP SOAP-Muon was introduced in result #14; attention SOAP and the trust gate were added in #16; the current all-hidden, frequency-1 version was introduced in #44.)

3. **u/w floor / RowUpdateFloor**:
   After Muon orthogonalization, if the update is too small relative to the weight, it is scaled up to the `0.3825` floor. In #46 this is applied per output row: each row whose update norm is below `0.3825 * ||row||` is lifted to that target, then the usual radius pin removes the global Frobenius-size change while preserving the row-shape change.
   (The scalar u/w floor was introduced in result #9; the current `0.3825` target comes from the later #29/#30 lineage; RowFloor was introduced in result #46.)

4. **Radial brake + radius rescale + post-pin CWD**:
   The Muon update is decomposed into a component parallel to the weight and a tangential component. If the proposed step would move the weight outward radially, that radial component is multiplied by `0.5`; inward radial movement is left alone. The code then computes the intended post-step weight norm from only this radial first-order effect, applies the full update, and rescales the resulting tensor to exactly that intended norm. This removes accidental norm drift from finite tangential steps while preserving the post-update direction. For 2D Muon parameters, the code also records a Cautious Weight Decay mask `1[update * weight > 0]`, i.e. coordinates where the step `weight <- weight - lr * update` is already shrinking the weight. After the radius rescale, it applies `CWD=0.025` only on those masked coordinates, so the coordinatewise shape change is not normalized away by the radius pin.
   (The radial brake and radius rescale were introduced in result #29. Cautious Weight Decay was used earlier in Track-3 PR #265; the current post-pin version on the #44/#46 stack was introduced in result #46.)

5. **PowerCool LR schedule**:
   LR is flat early, then follows a power-law cooldown with power `1.2` and schedule endpoint `2900`: `lr = min(initial_lr, power_c * (2900 - step)**1.2)`. In #46, the Muon LR is flat until about step 514, while the Adam/auxiliary LRs are flat until about step 1487.
   (Introduced in result #20.)

6. **Muon momentum schedule**:
   Muon momentum warms from `0.85 -> 0.95` over 300 steps, then cools from `0.95 -> 0.85` over the final 200 steps of the 2900-step schedule.
   (Introduced into the accepted Track 3 lineage in result #30; the current 200-step cooldown was introduced in #44.)

7. **[EMA-Nesterov](https://arxiv.org/abs/2605.25395) wrapper**:
   The whole optimizer stack is wrapped in EMA-Nesterov. It keeps an EMA of parameter update, moves the model forward along that smoothed lookahead direction before the forward/backward pass, computes gradients at that lookahead position, then lets the inner optimizer stack takes its update step from the lookahead position. The lookahead scale is `.3 * lr/max_lr`, the displacement EMA is `.99`, and it is active after a 300-step prefill until step `1950`.
   (Introduced in result #39; the current `.3` lookahead variant entered the record chain in #40.)

8. **[Adam](https://arxiv.org/abs/1412.6980) tuning**:
   The token embedding and output projection use Adam updates with no weight decay. Norm gains and other 1D/bias params use a minimal bias-correction-free Adam, with beta2 tuned per parameter family:
   gains `.99`, most biases `.997`, except `attn.proj.bias` which uses `.9965`.
   (Aux Adam is part of the result #1 baseline; the current auxiliary beta2 split was introduced in #44.)

9. **Initialization tweaks**:
   Projection params are zero-initialized. `mlp.fc` weights get depth-scaled down by a factor of `1.0 - 0.30 * (layer_idx / (num_layers - 1))`. RMSNorm gains use CGI/Rademacher paired gain init with alpha `.125`.
   (Projection zero-init is present from result #1; depth-scaled `mlp.fc` and CGI/Rademacher gains entered the accepted lineage in #30; the current CGI alpha `.125` was introduced in #44.)

10. **Tail-EMA final readout**:
   Starting at step `2400`, it keeps an EMA of every non-embedding parameter with horizon of 150 steps. At validation time, it evaluates `theta_eval = 0.4 * theta + 0.6 * EMA(theta)`.
   (Introduced in result #45, following the earlier fixed final-readout lineage from #38 and #43; the current #46 variant starts at step `2400` and excludes the token embedding.)

What it explicitly does **not** use: Contra-Muon, Soft-Muon, Circuit-Muon, Aurora, TrailDelta, fixed-anchor readout, Muon-history forecasting, CenterShrinkAdam, or NorMuon-lite row/column variance preconditioning. Some stale comments mention older machinery, but these paths are off or removed in the #46 submission defaults.

Notes: Several active details have yet to be proven independently beneficial. We do not yet know whether the attention SOAP trust gate is helping. The final Muon momentum cooldown is probably irrelevant to the accepted #46 step, since the cooldown is scheduled over steps `2700..2900`, but the accepted validation is at step `2690`. PowerCool may also be doing little in this record, since the run ends before the schedule becomes much different from a WSD-style cooldown. Likewise, it is unclear whether the Rademacher gain init matters, whether the depth-dependent `mlp.fc` init matters beyond a below-stat-sig ablation signal of about `0.00003` val loss, or whether `attn.proj.bias` beta2 `.9965` is meaningfully different from the other auxiliary beta2 value `.997`.


<br><br>

<img src="img/figure_wr_vs_base.png" width="50%">

Figure 1. The current WR is a 20.8% speedup compared to a well-tuned baseline.



## Rules

For a new result to be considered valid, it must satisfy the following constraints:
1. The dataset, batch size, and architecture must be kept the same as the baseline.
2. The trainer cannot perform multiple forward-backward passes per step.
3. (**Target loss and statistical significance**) The submitted run(s) must attain below 3.28 val loss, thereby matching [Andrej Karpathy's GPT-2 replication](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).
To ensure statistical significance, the run(s) are required to pass a one-sided z-test assuming σ=0.0013 that achieves p<.001 (hence 3.09σ = 0.004 delta below the target). E.g., for a single non-cherry-picked run, any val loss below 3.276 suffices, and for n=4 runs, any average below 3.278 suffices. **The precise condition we require is `(3.28 - avg_loss) * num_runs**0.5 >= 0.004`**, where `avg_loss` is the average result over `num_runs` non-cherry-picked runs. (Note: My first three results failed to follow this rule)
4. (**Reproducibility**) To ensure full reproducibility, all code needed to reproduce the run must be included in the logfile. In particular, third-party optimizer libraries must not be imported; instead, the necessary code must be copied in its entirety into the train script. It's okay if this leads to thousands of extra lines, in the case of complex third-party libraries.
5. (**No p-hacking using val spam**) Per-run early-stopping based on val loss (or any other form of per-run decision based on val loss) is not allowed. On the other hand, it *is* permitted to print the val loss every 25 steps near the end of training, and then select the earliest step that has stat sig for reaching the target. In other words, 
early stopping is permitted as long as the stopping point is selected the same across all trials.

### Freedoms

New results have the freedom to modify:
1. The optimization algorithm, even to something slow in terms of wallclock speed.
2. The optimization hyperparameters, including schedules thereof.
3. The model initialization.

We welcome not only new results which advance the global SOTA, but also results which advance the per-optimizer SOTA,
e.g., better hyperparameters for Adam (even if it still isn't beating the baseline).

AI-based submissions are also completely welcome. You can use AI to write the entire PR; a human does not even need to be aware
that a submission was created, as long as it follows the rules. That being said, it would be polite for you ask your AI to 
simplify any code it writes, since a tendency of AI-based results is to include techniques that neither help nor hurt, but add complexity ("barnacles"), which
makes the code more difficult for future humans (and AIs) to understand.


### Skeptical results

I typically do not reproduce new results myself before accepting. Therefore, there is a possibility of fake results being accepted, if there is a submitter
who is feeling devious.
To provide a long-term defense against this, I welcome new skeptical results which themselves challenge an old result by providing statistical evidence
that the old result either cheats (e.g., increased the model size) or does not really attain below 3.28 mean loss in the reported step count (e.g., possibly they cherrypicked logs).
Such skeptical results are welcomed as valued first-class objects, and will be broadcast.
The acceptance of such a skeptical result which disproves an old result may
warrant a ban for the rapscallion submitter of the disproven old result. Hopefully this kind of thing never actually happens though.


### Pairwise statistical significance

In some cases, new results can attain statistical significance for <3.28 at a lower step count than a previous result, while nevertheless not being statistically
significantly stronger than the previous result. In other words, we have evidence that the new result is valid, but we don't yet have evidence
that the same step count could not have been attained by running the old result with the same step and seed count as the new result.

For example, result #16 is a perfectly valid new result/record, because it attains statsig for <3.28 at 3125 steps whereas result #14 did not.
However, it is not statsig better than its predecessor #14, because #14 attains 3.2790 (n=4) at 3125 and it attains 3.2784 (n=8) at the same, which is not a statsig difference.

In cases where the final step count was changed, to determine pairwise statsig we will need to extrapolate the expected change in loss.
To do this we are aided by the following information:
Reducing the step count of result #12 by 200 increases the mean loss from 3.2790 (n=20) to [3.2881 (n=8)](results/478c0427-06ce-4952-bc0a-7e2dfaea29b6.txt). This is a gap of 0.0091 across 200 steps, or 0.0045 per 100 steps. Therefore, for example, if you run a setup and get a mean loss of 3.2720, and want to target 3.2790, then you can likely shorten your run by approximately 156 steps.

For example, result #11 is not pairwise statsig vs the prior record, because it lowers step count by 25 while increasing estimated mean val loss by 3.2785 - 3.2771 = 0.0014.
According to the above information, 0.0014 val loss is worth about 100 * 0.0014/0.0045 = 31 steps, which is greater than the step saving.
To clarify, this does not constitute evidence that the *algorithm* provided by result #11 is not really better; it only indicates that the *logfiles* provided by
result #11 do not contain sufficient evidence for that conclusion.

Another calculation: For result #13 --  a perfectly valid new <3.28 record -- we have the following two calculations.
Against result #11, we have a difference of 15 steps, with final loss being the same. These steps are worth approximately
15/100 * 0.0045 = 0.000675 units of val loss. The two seed counts are n=16 and n=10.
**The general requirement is `(final_loss_diff + exp_stepbased_loss_diff) / (1/n1 + 1/n2)**0.5 >= 0.004`.**
For this case, the LHS is 0.00167, which does not reach statsig.
If we instead compare to result #8, the LHS is `((3.2778 - 3.2785) + (40/100 * 0.0045)) / (1/10 + 1/16)**0.5 = 0.0027`, which again does not reach statsig.

A third calculation: For the Muon hparams in result #12 vs #6, we have `final_loss_diff = -0.0002`, `exp_stepbased_loss_diff = 50/100*0.0045 = 0.00225`, and `n1 = n2 = 20`. Therefore, the LHS of our general formula is `0.00648`, which is above `0.004`, so we have evidence that the muon hyperparameters of result #12 really are statsig better than those of result #6.

------
------

## Motivation

> [benchmark competitions are the prime mover of AI progress.](https://www.argmin.net/p/too-much-information#:~:text=benchmark%20competitions%20are%20the%20prime%20mover%20of%20AI%20progress.)
> -- Prof. Ben Recht

Most research into novel neural network optimizers occurs in the public research community, not in the frontier labs.
For example, since the release of Muon, there have been [40+ papers published citing it that propose a new optimizer of their own](
https://chatgpt.com/share/69ed22e3-0870-83ea-a449-b4ce97d764f3). And more broadly, there exist somewhere between [hundreds](https://chatgpt.com/c/69b10bd7-f92c-8325-b516-d999b5b2b409) and [thousands](https://claude.ai/share/fb9590de-c4b7-44f8-bfbb-7f80af30d3f9) of papers on neural network optimization across the internet.

How do these hundreds of optimizers compare - which ones are able to optimize neural networks in the fewest steps?
The reality is that as a community, we simply don't know. Why not?
Because typically, these papers each use their own unique experimental setups, making it challenging to verify whether their baselines are well-tuned or to make comparisons between papers.

For researchers interested in neural network optimization, this is daunting - a sea of methods, many of them claiming to be SOTA, and no shared infrastructure to sort signal from noise. As it stands, the burden is on the individual researcher to make sense of this madness. Calculating the outcome: If N different researchers publish N optimizer papers claiming SOTA, all of them unverifiable and mutually incomparable, then there are only two possibilities: Either (a) research grinds to a halt due to the Θ(N) growth in experiments that each researcher needs to conduct to get a private sense of the real SOTA, or (b) researchers start simply ignoring each other's papers.
Neither of these are desirable outcomes, and today we are in some mix of the two.

This benchmark aims to provide a simple, easily-accessible communally-shared way to filter signal from noise, aiming to surface ignored papers/ideas and reduce the number of experiments that each researcher must do in order to get an accurate picture of the SOTA.
It is a collaborative|competitive benchmark, meaning that, for example, if anyone can find hyperparameters that enable Adam
to reach the target loss in fewer steps than Muon, then we the benchmark authors will be keen to include this result and promote it on social media
within a short period of time, even though there is a conflict of interest since we are also Muon authors.
In contrast, in historical cases where a paper proposing a new SOTA-claiming optimizer has later turned out to have been confounded by an undertuned baseline,
it has often been difficult for such information to propagate through the community, due to the fact that negative results are typically not paper-worthy on their own,
even if they disprove another paper which has hundreds of citations.

Prior competitive optimization benchmarks already exist, but often suffer from high barriers to entry due to strenuous requirements or high complexity.
This benchmark aims for maximum convenience in order to make new results as convenient/accessible as possible:
The baseline code should be comprehensible with minimal effort, and experiments should take no more than ~15 minutes and cost no more than ~$6.


## Addressing a potential critique

Quoted from a post on X:

> The idea of SOTA in “optimization” is b.s. When the architecture changes we may get need different optimization algorithms.

Two replies:

1. Muon was originally determined empirically for the CIFAR-10 speedrun setting, where it lowered the record from 3.09 to 2.59 seconds.
It was then transferred to NanoGPT, where it continued to work well. These two settings are about as different as one can reasonably find within deep learning research. This anecdote suggests that when a properly-tuned baseline is used, the process of searching for good optimizers does not tend to produce methods that are overfit to any particular experimental setup.
2. That being said, even in the world where the best optimizer *does* depend heavily on the choice of experimental setup, the practical need for benchmarks to filter signal from noise would still remain. We would just need to set up more than one benchmark, in order to effectively cover the space of experimental setups (e.g., a more developed benchmark suite would likely cover multiple batch sizes and multiple scales). 


## Details on relation to the main speedrun

Aiming towards simplicity, for this benchmark we have removed the non-standard neural network parameters (value embeddings, skip connection lambdas) and triton kernels that are used in the main speedrun. We have also added back standard parameters which are wallclock-inefficient at small scale, namely the RMSNorm gains and Linear layer biases.
Finally, we have replaced the sophisticated local-global pattern of attention by simple causal attention across contexts of 1024 tokens.


## Guidelines

General
* Changes to the code should be concentrated to the `Optimization` and `Init & Optim Hyperparams` sections.
* Results should be submitted in the form of logfiles, like the ones linked in the [results history](#notable-results-history) section above. Logfiles must include the full code used by the run, such that if we replace `train_gpt_simple.py` by the code, then running the quickstart will reproduce the run (up to random seed variance). In particular, hardcoded hyperparameters are to be preferred as compared to command line arguments.

On tuning hyperparameters:
* Typically, the most sensitive hyperparameter is the weight decay, followed by the learning rate, and then everything else.
* For a given hyperparameter change, in general it is not possible to tell whether it will have a positive or negative effect on the final val loss until the entire run completes. For example, the val loss at step 1000 does not strongly correlate with the final loss.
On the other hand, especially for optimizers with a lot of hyperparameters where we are quite uncertain, it can often be a good strategy to say halve the entire run's step count (thereby getting worse than the target val loss), and then tune all hyperparameters for the shorter/quicker run, and then bring the step count back up, and retune just the weight decay and learning rate. Since often the optimal settings for the non-wd/lr hparams (like Adam betas) will be the same for shorter and longer runs.
* On data: The baseline trains for 3550 * 524288 = ~2B tokens. The quickstart script downloads 4B tokens of FineWeb, allowing trainings up to 7600 steps. If you'd like to train for more steps than that, then you must get more tokens via something like `python data/cached_fineweb10B.py 100`, which will download the maximum 10B tokens. However, Adam runs can reach the target val loss within around 3B tokens, so this should not be necessary except for pathologically inefficient optimizers.
* For [PSGD Kron](https://github.com/evanatyourservice/kron_torch), it seems that reasonable starting hparams are `lr=.0005, weight_decay=.625`.

## Citation

```
@misc{moddednanogpt_optimbench_2026,
  author       = {Keller Jordan},
  title        = {Modded-NanoGPT Optimization Benchmark},
  year         = {2026},
  url          = {https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization}
}
```
