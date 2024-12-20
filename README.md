# Modded-NanoGPT

This is a modified variant of the [PyTorch GPT-2 trainer](https://github.com/karpathy/llm.c/blob/7b929300217ff1a974b63791a228928b39b26409/train_gpt2.py) from
Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) repo, which attains the same final validation loss in only:
* 0.8B tokens instead of 10B
* 3.8 minutes on 8xH100 instead of 45

It has been hyperoptimized by the community, and has become a good baseline from which to perform research on the architecture/optimizer/etc.

It uses the following techniques:
* Modernized architecture: Rotary embeddings, QK-Norm, and ReLU^2.
* New optimizer: [Muon - Momentum Orthogonalized by Newton-schulz](https://kellerjordan.github.io/posts/muon/) [[standalone implementation](https://github.com/KellerJordan/Muon)].
* Untied head from embedding.
* Projection and classification layers initialized to zero (muP-like).
* Architectural shortcuts: value residual and embedding shortcut (partially following https://arxiv.org/abs/2410.17897).
* Momentum warmup.
* Tanh soft logit capping (following Gemma 2).
* FlexAttention with window size warmup.
* Extra embeddings which are fed into intermediate attention layers.

The training has attained this speed due to the contributions of meself, [@Grad62304977](https://x.com/Grad62304977),
[@jxbz](https://x.com/jxbz), [@bozavlado](https://x.com/bozavlado), [@brendanh0gan](https://x.com/brendanh0gan),
[@KoszarskyB](https://x.com/KoszarskyB), & [@fernbear.bsky.social](https://bsky.app/profile/fernbear.bsky.social).

---

## Running the current record

To install and execute the training, run the following four commands.
They should all complete within <20min on an 8xH100 with decent internet connection.
If the torch install command updates your cuda installation, you many need to reboot.
```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git & cd modded-nanogpt
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20241203+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade # install torch 2.6.0
python data/cached_fineweb10B.py 10 # downloads only the first 1.0B training tokens to save time
./run.sh
```

The result will be a transformer with 124M active parameters trained for 1480 steps on 0.75B tokens of Fineweb [1], achieving ~3.278 mean validation loss (w/ up to 0.005 inter-run stddev).
For comparison, the default llm.c PyTorch trainer yields [>3.28 validation loss after training for 19560 steps on 10B tokens](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).

**Note: torch.compile will take a long time on the first run.**

## Running it on fewer GPUs or with less memory

* To run on fewer GPUs, just modify `run.sh` to have a different `--nproc_per_node`. (this does not change the expected behavior of the training)
* If you're running out of memory, you may need to reduce the sequence length for FlexAttention (which does change the training. see [here](https://github.com/KellerJordan/modded-nanogpt/pull/38) for a guide)

## Running with Docker

For cases where CUDA or NCCL versions aren't compatible with your current system setup, Docker can be a helpful alternative.
This approach standardizes versions for CUDA, NCCL, CUDNN, and Python, reducing dependency issues and simplifying setup. 
Note: an NVIDIA driver must already be installed on the system (useful if only the NVIDIA driver and Docker are available).

```bash
sudo docker build -t modded-nanogpt .
sudo docker run -it --rm --gpus all -v $(pwd):/modded-nanogpt modded-nanogpt python data/cached_fineweb10B.py 18
sudo docker run -it --rm --gpus all -v $(pwd):/modded-nanogpt modded-nanogpt sh run.sh
```
---

## World record history

The following is the progression of world records for the task of *training a model with 124M active parameters to 3.28 validation loss on FineWeb in the minimal amount of time on an 8xH100 machine.*

| # | Record time | Description | Date | Log | Contributors |
| - | - | - | - | - | - |
1 | 45 minutes | [llm.c baseline](https://github.com/karpathy/llm.c/discussions/481) | 05/28/24 | [log](./records/101324_llmc/main.log) | @karpathy, llm.c contributors
2 | 31.4 minutes | [Architectural modernizations & tuned learning rate](https://x.com/kellerjordan0/status/1798863559243513937) | 06/06/24 | [log](./records/060624_AdamW/f66d43d7-e449-4029-8adf-e8537bab49ea.log) | @kellerjordan0
3 | 24.9 minutes | [Introduced the Muon optimizer](https://x.com/kellerjordan0/status/1842300916864844014) | 10/04/24 | none | @kellerjordan0, @jxbz
4 | 22.3 minutes | [Muon improvements](https://x.com/kellerjordan0/status/1844820919061287009) | 10/11/24 | [log](./records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt) | @kellerjordan0, @bozavlado
5 | 15.2 minutes | [Pad embeddings & architectural improvements](https://x.com/kellerjordan0/status/1845865698532450646) | 10/14/24 | [log](./records/101424_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt) | @Grad62304977, @kellerjordan0
6 | 13.1 minutes | [Distributed the overhead of Muon](https://x.com/kellerjordan0/status/1847291684016783746) | 10/18/24 | [log](./records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt) | @kellerjordan0
7 | 12.0 minutes | [Upgraded PyTorch from 2.4.1 to 2.5.0](https://x.com/kellerjordan0/status/1847358578686152764) | 10/18/24 | [log](./records/101824_PyTorch25/d4bfb25f-688d-4da5-8743-33926fad4842.txt) | @kellerjordan0
8 | 10.8 minutes | [Untied embed and lm_head](https://x.com/kellerjordan0/status/1853188916704387239) | 11/03/24 | [log](./records/110324_UntieEmbed/d6b50d71-f419-4d26-bb39-a60d55ae7a04.txt) | @Grad62304977, @kellerjordan0
9 | 8.2 minutes | [Shortcuts & tweaks](https://x.com/kellerjordan0/status/1854296101303800108) | 11/06/24 | [log](./records/110624_ShortcutsTweaks/dd7304a6-cc43-4d5e-adb8-c070111464a1.txt) | @Grad62304977, @kellerjordan0
10 | 7.8 minutes | [Bfloat16 activations](https://x.com/kellerjordan0/status/1855267054774865980) | 11/08/24 | [log](./records/110824_CastBf16/a833bed8-2fa8-4cfe-af05-58c1cc48bc30.txt) | @kellerjordan0
11 | 7.2 minutes | [U-net & 2x lr](https://x.com/kellerjordan0/status/1856053121103093922) | 11/10/24 | [log](./records/111024_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt) | @brendanh0gan
12 | 5.03 minutes | [FlexAttention](https://x.com/kellerjordan0/status/1859331370268623321) | 11/19/24 | [log](./records/111924_FlexAttention/8384493d-dba9-4991-b16b-8696953f5e6d.txt) | @KoszarskyB
13 | 4.66 minutes | [Attention window warmup](https://x.com/hi_tysam/status/1860851011797053450) | 11/24/24 | [log](./records/112424_WindowWarmup/cf9e4571-c5fc-4323-abf3-a98d862ec6c8.txt) | @fernbear.bsky.social
14 | 4.41 minutes | [Value Embeddings](https://x.com/KoszarskyB/status/1864746625572257852) | 12/04/24 | [log](./records/120424_ValueEmbed) | @KoszarskyB
15 | 3.95 minutes | [U-net pattern for value embeds, assorted code improvements](https://x.com/YouJiacheng/status/1865761473886347747) | 12/08/24 | [log](records/120824_UNetValueEmbedsTweaks) | @leloykun, @YouJiacheng
16 | 3.80 minutes | [MFU tweaks](https://x.com/YouJiacheng/status/1866734331559071981) | 12/10/24 | [log](records/121024_MFUTweaks) | @YouJiacheng

### Speedrun rules

All new record attempts:

1. Must not modify the train or validation data pipelines. (Except to change batch size, seqlen, attention structure etc. I.e., just don't change the underlying tokens.)
2. Must use ≤ 124M active parameters per token. (So MoE is fine; and extra embedding layers can be added since they only contribute hidden_dim active params.)
3. Must attain ≤ 3.28 val loss. Unfortunately, due to high inter-run variance, new record attempts must provide enough run logs to attain a statistical significance level of p<0.01 that their average val loss is lower than 3.28. You see see how to conduct a t-test [here](./records/120424_ValueEmbed).

Other than that, go crazy! Anything is fair game

<!--Note: The original llm.c baseline is intended to be closer to a replication of GPT-2 than to an optimized LLM training.
So it's no surprise that there is room to improve; as @karpathy has said, 'llm.c still has a lot of pending optimizations.'
In addition, many of the techniques used in these records are completely standard, such as rotary embeddings.
The goal of this benchmark/speedrun is simply to find out which techniques actually work, and maybe come up with some new ones.-->
<!--The goal of this benchmark is simply to find out all the techniques which actually work, because I'm going crazy reading all these
LLM training papers
which claim a huge benefit but then use their own idiosyncratic non-competitive benchmark and therefore no one in the community has any idea if it's legit for months.-->
<!--[LLM](https://arxiv.org/abs/2305.14342) [training](https://arxiv.org/abs/2402.17764) [papers](https://arxiv.org/abs/2410.01131)-->
<!--I mean hello??? We're in a completely empirical field; it is insane to not have a benchmark. Ideally everyone uses the same LLM training benchmark,
and then reviewing LLM training papers becomes as simple as checking if they beat the benchmark. It's not like this would be unprecedented, that's how things
were in the ImageNet days.
The only possible 'benefit' I can think of for any empirical field to abandon benchmarks is that it would make it easier to publish false results. Oh, I guess that's why it happened.
Hilarious to think about how, in the often-commented-upon and ongoing collapse of the peer review system, people blame the *reviewers* --
yeah, those guys doing free labor who everyone constantly musters all of their intelligence to lie to, it's *their* fault! My bad, you caught me monologuing.-->

### Notes
* For the llm.c baseline: The 90 minute time is on 8xA100; it's 45 minutes on 8xH100. This baseline is essentially a hardware-optimized GPT-2-small replication using better training data.
* All runs before 11/19/24 can be run with PyTorch 2.5.1 or below. Runs including and after 11/19/24 require PyTorch 2.6.0 (nightly) to use FlexAttention.

---

### Notable forks

* [https://github.com/BlinkDL/modded-nanogpt-rwkv](https://github.com/BlinkDL/modded-nanogpt-rwkv)
* [https://github.com/nikhilvyas/modded-nanogpt-SOAP](https://github.com/nikhilvyas/modded-nanogpt-SOAP)

---

### Q: What is the point of NanoGPT speedrunning?

A: The officially stated goal of NanoGPT speedrunning is as follows: `gotta go fast`. But for something a little more verbose involving an argument for good benchmarking, here's some kind of manifesto, adorned with a blessing from the master. [https://x.com/karpathy/status/1846790537262571739](https://x.com/karpathy/status/1846790537262571739)

### Q: What makes "NanoGPT speedrunning" not just another idiosyncratic benchmark?

A: Because it is a *competitive* benchmark. In particular, if you attain a new speed record (using whatever method you want), there is an open invitation for you
to post that record (on arXiv or X) and thereby vacuum up all the clout for yourself. I will even help you do it by reposting you as much as I can.

<!--On the contrary, for example, the benchmark used in the [Sophia](https://arxiv.org/abs/2305.14342) paper does *not* have this property.
There is no such open invitation for anyone to compete on the benchmark they used. In particular, if, for a random and definitely not weirdly specific example, you happen to find better AdamW hyperparameters for their training setup than
the ones they used which significantly close the gap between AdamW and their proposed optimizer,
then there is no clear path for you to publish that result in *any* form.
You could try posting it on X.com, but then you would be risking being perceived as aggressive/confrontational, which is *not a good look* in this racket.
So if you're rational, the result probably just dies with you and no one else learns anything
(unless you're in a frontier lab, in which case you can do a nice internal writeup. Boy I'd love to get my hands on those writeups).-->

["Artificial intelligence advances by inventing games and gloating to goad others to play" - Professor Ben Recht](https://www.argmin.net/p/too-much-information)

### Q: NanoGPT speedrunning is cool and all, but meh it probably won't scale and is just overfitting to val loss

A: This is hard to refute, since "at scale" is an infinite category (what if the methods stop working only for >100T models?), making it impossible to fully prove.
Also, I would agree that some of the methods used in the speedrun are unlikely to scale.
But if the reader cares about 1.5B models, they might be convinced by this result:

*Straightforwardly scaling up the speedrun (10/18/24 version) to 1.5B parameters yields a model with GPT-2 (1.5B)-level HellaSwag performance 2.5x more cheaply than [@karpathy's baseline](https://github.com/karpathy/llm.c/discussions/677) ($233 instead of $576):*

![](img/nanogpt_speedrun51.png)
[[reproducible log](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/102024_ScaleUp1B/ad8d7ae5-7b2d-4ee9-bc52-f912e9174d7a.txt)]
![](img/nanogpt_speedrun52.png)

---

## [Muon optimizer](https://github.com/KellerJordan/Muon)

Muon is defined as follows:

![](img/algo_optimizer.png)

Where NewtonSchulz5 is the following Newton-Schulz iteration [2, 3], which approximately replaces `G` with `U @ V.T` where `U, S, V = G.svd()`.
```python
@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T 
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T 
    return X.to(G.dtype)
```

For this training scenario, Muon has the following favorable properties:
* Lower memory usage than Adam
* ~1.5x better sample-efficiency
* <2% wallclock overhead


### Provenance

Many of the choices made to generate this optimizer were obtained experimentally by our pursuit of [CIFAR-10 speedrunning](https://github.com/KellerJordan/cifar10-airbench).
In particular, we experimentally obtained the following practices:
* Using Nesterov momentum inside the update, with orthogonalization applied after momentum.
* Using a specifically quintic Newton-Schulz iteration as the method of orthogonalization.
* Using non-convergent coefficients for the quintic polynomial in order to maximize slope at zero, and thereby minimize the number of necessary Newton-Schulz iterations.
It turns out that the variance doesn't actually matter that much, so we end up with a quintic that (rapidly) converges to the range 0.68, 1.13 upon repeated application, rather than to 1.
* Running the Newton-Schulz iteration in bfloat16 (whereas Shampoo implementations often depend on inverse-pth-roots run in fp32 or fp64).

Our use of a Newton-Schulz iteration for orthogonalization traces to [Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325),
who suggested it as a way to compute Shampoo [5, 6] preconditioners, and theoretically explored Shampoo without preconditioner accumulation.
In particular, Jeremy Bernstein @jxbz sent us the draft, which caused us to experiment with various Newton-Schulz iterations as the
orthogonalization method for this optimizer.
If we had used SVD instead of a Newton-Schulz iteration, this optimizer would have been too slow to be useful.
Bernstein & Newhouse also pointed out that Shampoo without preconditioner accumulation is equivalent to steepest descent in the spectral norm,
and therefore Shampoo can be thought of as a way to smooth out spectral steepest descent.
The proposed optimizer can be thought of as a second way of smoothing spectral steepest descent, with a different set of memory and runtime tradeoffs
compared to Shampoo.

---

## Startup script

Here's a good startup script for a fresh 8xH100 instance.

```
sudo apt-get update
sudo apt-get install vim tmux python3-pip python-is-python3 -y
git clone https://github.com/KellerJordan/modded-nanogpt.git
cd modded-nanogpt
tmux

pip install numpy==1.23.5 huggingface-hub tqdm
pip install --upgrade torch &
python data/cached_fineweb10B.py 18
```

---

## References

1. [Penedo, Guilherme, et al. "The fineweb datasets: Decanting the web for the finest text data at scale." arXiv preprint arXiv:2406.17557 (2024).](https://arxiv.org/abs/2406.17557)
2. Nicholas J. Higham. Functions of Matrices. Society for Industrial and Applied Mathematics, 2008. Equation 5.22.
3. Günther Schulz. Iterative Berechnung der reziproken Matrix. Z. Angew. Math. Mech., 13:57–59, 1933.
4. [Jeremy Bernstein and Laker Newhouse. "Old Optimizer, New Norm: An Anthology." arxiv preprint arXiv:2409.20325 (2024).](https://arxiv.org/abs/2409.20325)
5. [Vineet Gupta, Tomer Koren, and Yoram Singer. "Shampoo: Preconditioned stochastic tensor optimization." International Conference on Machine Learning. PMLR, 2018.](https://arxiv.org/abs/1802.09568)
6. [Anil, Rohan, et al. "Scalable second order optimization for deep learning." arXiv preprint arXiv:2002.09018 (2020).](https://arxiv.org/abs/2002.09018)
7. [Hägele, Alexander, et al. "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations." arXiv preprint arXiv:2405.18392 (2024).](https://arxiv.org/abs/2405.18392)

## Citation

```
@misc{modded_nanogpt_2024,
  author       = {Keller Jordan and Jeremy Bernstein and Brendan Rappazzo and
                  @fernbear.bsky.social and Boza Vlado and You Jiacheng and
                  Franz Cesista and Braden Koszarsky and @Grad62304977},
  title        = {modded-nanogpt: Speedrunning the NanoGPT baseline},
  year         = {2024},
  url          = {https://github.com/KellerJordan/modded-nanogpt},
  note         = {Accessed: 2024-12-09}
}
```

<img src="img/dofa.jpg" alt="itsover_wereback" style="width:100%;">

