# PrimeIntellect Instance Setup Plan

## Instance Details
- **Host**: `ubuntu@147.185.40.231`
- **GPUs**: 8x NVIDIA H100 80GB HBM3
- **Status**: Connected and ready

---

## Setup Steps

### Step 1: Generate SSH Key on Instance
Generate a new SSH key on the instance for GitHub access.

```bash
ssh ubuntu@147.185.40.231 << 'EOF'
# Generate new ed25519 key
ssh-keygen -t ed25519 -C "primeintellect-h100" -f ~/.ssh/id_ed25519 -N ""

# Display the public key for user to add to GitHub
echo "=== Add this key to GitHub ==="
cat ~/.ssh/id_ed25519.pub
echo "=============================="

# Add GitHub to known hosts
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Configure git
git config --global user.email "c@weill.io"
git config --global user.name "Charles Weill"
EOF
```

**User action**: Copy the displayed public key and add it to https://github.com/settings/keys

### Step 2: Clone & Install Dotfiles
```bash
ssh ubuntu@147.185.40.231 << 'EOF'
cd ~
git clone git@github.com:cweill/dotfiles.git
cd dotfiles
./install.sh
EOF
```

### Step 3: Clone & Setup modded-nanogpt
```bash
ssh ubuntu@147.185.40.231 << 'EOF'
cd ~
git clone git@github.com:cweill/modded-nanogpt.git
cd modded-nanogpt

# Install dependencies
pip install -r requirements.txt
pip install torch==2.10.0.dev20251210+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126

# Download training data (first 900M tokens)
python data/cached_fineweb10B.py 9
EOF
```

### Step 4: Run the Speedrun
```bash
ssh ubuntu@147.185.40.231 "cd ~/modded-nanogpt && ./run.sh"
```

---

## Verification
- [ ] SSH key copied and working
- [ ] Dotfiles installed
- [ ] modded-nanogpt cloned
- [ ] PyTorch nightly installed
- [ ] Training data downloaded
- [ ] Training completes with ≤3.28 validation loss

---

## Original Learning Plan

## Goal
Develop deep understanding of cutting-edge training optimization techniques before starting at xAI on Grok.

## Your Setup
- **GPU Access**: 8xH100 (can run full speedrun)
- **Background**: Expert (skip foundational material)
- **Focus**: All areas - architecture, optimizers, and systems

---

## Running on PrimeIntellect

**Good news**: Official speedrun records are validated on PrimeIntellect 8xH100 GPUs, so this is the canonical environment.

### Step 1: Launch Cluster
1. Go to https://app.primeintellect.ai/
2. Launch an 8xH100 GPU instance (single node is sufficient)

### Step 2: Environment Setup
```bash
# Clone the repo
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt

# Install dependencies
pip install -r requirements.txt
pip install torch==2.10.0.dev20251210+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126

# Download training data (first 900M tokens)
python data/cached_fineweb10B.py 9
```

### Step 3: Run Training
```bash
./run.sh  # Runs: torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Note**: First run takes ~7 minutes extra for torch.compile warmup.

### Environment Variables
| Variable | Purpose | Example |
|----------|---------|---------|
| `DATA_PATH` | Location of FineWeb data | `DATA_PATH=/data ./run.sh` |
| `DISABLE_FP8` | Disable FP8 matmul (use BF16) | `DISABLE_FP8=1 ./run.sh` |

### Docker Alternative (recommended for reproducibility)
```bash
sudo docker build -t modded-nanogpt .
sudo docker run -it --rm --gpus all -v $(pwd):/modded-nanogpt modded-nanogpt python data/cached_fineweb10B.py 9
sudo docker run -it --rm --gpus all -v $(pwd):/modded-nanogpt modded-nanogpt sh run.sh
```

---

## Enabling/Disabling Features

**Important**: There's no config file or CLI args - features are controlled by editing `train_gpt.py` directly.

### Hyperparameters (lines 1800-1826)
```python
@dataclass
class Hyperparameters:
    # Training data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"

    # Batch sizes (schedule: 8→16→24 * 2048 * 8)
    train_bs_schedule: tuple = (8 * 2048 * 8, 16 * 2048 * 8, 24 * 2048 * 8)
    train_max_seq_len: int = 128 * 16  # 2048 tokens

    # Training duration
    num_scheduled_iterations: int = 1735
    num_extension_iterations: int = 40
    cooldown_frac: float = 0.50

    # Attention window schedule
    ws_schedule: tuple = (3, 7, 11)  # short→medium→long windows
    ws_final: int = 13
```

### Feature Toggles (code locations)

| Feature | How to Disable | Location |
|---------|---------------|----------|
| **FP8 matmul** | Set `DISABLE_FP8=1` env var | `train_gpt.py:1263` |
| **Paired Head Attention** | Change `self.paired_head_layers = [0, 2, 5, 9]` to `[]` | `train_gpt.py:1257` |
| **Value embeddings** | Set `ve = None` in forward, or remove from model | `train_gpt.py:1243-1255` |
| **Skip connections** | Set `skip_in = set()` and `skip_out = set()` | `train_gpt.py:1336-1337` |
| **Smear module** | Set `self.smear_gate` output to 0 | `train_gpt.py:1233-1236` |
| **Multi-token prediction** | Set `mtp_weights = None` | Search for `mtp_weights` |
| **Logit softcapping** | Remove line with `23 * torch.sigmoid` | `train_gpt.py:1389` |
| **ReLU² activation** | Change `relu_squared` to `F.gelu` in MLP | `train_gpt.py:1196` |
| **QK-Norm** | Remove `norm(q), norm(k)` calls | `train_gpt.py:1086, 1142` |
| **Muon optimizer** | Replace NorMuon with AdamW | `train_gpt.py:463-747` |

### Example: Disable Paired Head Attention
```python
# train_gpt.py line 1257
# Change from:
self.paired_head_layers = [0, 2, 5, 9]
# To:
self.paired_head_layers = []
```

### Example: Use AdamW instead of Muon
Replace the optimizer setup (around line 1700+) to use standard AdamW for all parameters instead of the NorMuon + DistAdam hybrid.

---

## Background: What is the NanoGPT Speedrun?

A competitive benchmark to train GPT-2 to ≤3.28 validation loss on FineWeb dataset as fast as possible on 8xH100 GPUs.
- **Baseline**: 45 minutes (llm.c)
- **Current record**: 1.82 minutes (24x speedup!)
- **Key insight**: Combines architecture, optimizer, and systems innovations

---

## Learning Path

### Phase 1: Run It First (2 hours)

**1. Run the current record**
```bash
cd /Users/cweill/Dropbox/GitHub/modded-nanogpt
./run.sh
```
- Watch it achieve 3.28 loss in ~2 minutes
- Note the training dynamics in the output

**2. Read the README**
- File: `README.md` - Contains full history of 58 records
- Pay attention to which changes gave biggest speedups

**3. Map the code structure**
- File: `train_gpt.py` (1993 lines)
- Key sections:
  - Model: lines 1227-1425
  - NorMuon optimizer: lines 463-747
  - Training loop: lines 1801-1993

---

### Phase 2: Architecture Innovations (Day 2)

**Key techniques to study in depth:**

| Technique | Location | Impact |
|-----------|----------|--------|
| Paired Head Attention | `train_gpt.py:1108-1171` | Latest record breakthrough |
| Rotary Embeddings (RoPE) | Look for `cos`, `sin` usage | Better position encoding |
| ReLU² activation | `train_gpt.py` search for `relu` | 1-2% improvement over GELU |
| Value embeddings | 3 embeddings mixed into attention | Novel architectural addition |
| U-net skip connections | Layer 3→6 and embedding→prediction | Feature reuse |
| Sliding window attention | FlashAttention3 integration | Memory/speed tradeoff |

**Deep dive: Paired Head Attention**
- Read Record #58 in `records/` directory
- Understand how interleaving attention heads enables efficiency

---

### Phase 3: Optimizer Innovations (Day 2-3)

**The most impactful area to understand:**

**1. Muon Optimizer**
- Read: https://kellerjordan.github.io/posts/muon/ (Keller Jordan's blog)
- Core idea: Momentum-based gradient orthogonalization
- Implementation: `train_gpt.py:463-747` (NorMuon class)

**2. Key optimizer concepts:**
- Polar Express orthogonalization (sign-based, faster than Newton-Schulz)
- Cautious weight decay with schedule
- Hybrid approach: Muon for matrices, Adam for embeddings
- Interleaved stepping between optimizers

**3. Learning rate scheduling:**
- Trapezoidal schedules (warmup → constant → decay)
- Batch size aware scaling
- Cooldown periods for momentum

---

### Phase 4: Systems Optimizations (Day 3)

**Hardware-aware optimizations:**

| Technique | Purpose |
|-----------|---------|
| FP8 matmul | Custom Triton kernels for speed |
| torch.compile | Kernel fusion, minimize Python overhead |
| NCCL reduce_scatter | Better than all_reduce for 8 GPUs |
| Vocab padding to 50304 | Tensor core alignment (multiple of 128) |
| Async data loading | Overlap compute and I/O |

**Study the Triton kernels:**
- Look for custom symmetric matmul operations
- Understand why custom kernels beat PyTorch defaults

---

### Phase 5: Data Efficiency (Day 3-4)

**Critical insight from Tyler's blog:**
- Document fragmentation wastes computation
- 20% of documents exceed 1024 tokens
- Solution: 32K sequences with document-aware attention masks

**Key concepts:**
- Variable-length sequence handling (BOS alignment)
- FlexAttention with custom masking
- Causal + document boundary + sliding window constraints

---

### Phase 6: Ablation Experiments (Day 4-5)

**With your 8xH100 access, run ablation studies:**

1. **Baseline comparison**
   - Run current record (~1.82 min)
   - Compare against earlier records from `records/` directory

2. **Optimizer ablations**
   - Swap NorMuon → AdamW: How much does Muon matter?
   - Disable cautious weight decay
   - Change learning rate schedule to cosine

3. **Architecture ablations**
   - Replace Paired Head Attention with standard attention
   - Remove skip connections (U-net pattern)
   - Switch ReLU² → GELU
   - Disable value embeddings

4. **Systems ablations**
   - Disable torch.compile
   - Use BF16 instead of FP8 for attention
   - Change batch size schedule

**Document your findings** - this builds deep intuition for what matters

---

### Phase 7: Connect to Larger Scale (Day 5+)

**How these techniques scale to Grok-sized models:**

| Speedrun Technique | Large-Scale Relevance |
|--------------------|----------------------|
| Muon optimizer | Orthogonalization may help with larger matrices |
| Sliding window attention | Used in Gemma 2, efficient for long context |
| FP8 training | Critical for H100/H200 utilization |
| Careful scheduling | Even more important at scale |
| Data efficiency | Crucial when training on trillions of tokens |

---

## Key Papers to Read

1. **RoFormer** - Rotary Position Embeddings
2. **Gemma 2** - Sliding window attention, logit soft-capping
3. **FlexAttention** - Custom attention masks in PyTorch
4. **Primer** - Activation function comparisons (ReLU²)

---

## Key Code Files in This Repo

| File | Purpose |
|------|---------|
| `train_gpt.py` | Main speedrun script (study this most) |
| `train_gpt_medium.py` | GPT-2 Medium track (larger model) |
| `run.sh` | Launch script for distributed training |
| `data/fineweb.py` | Data loading implementation |
| `records/` | Historical logs with detailed notes |

---

## Questions to Explore

1. Why does Muon work so well for linear layers but not embeddings?
2. How do skip connections from early layers help at the end?
3. What's the tradeoff between batch size and sequence length?
4. How do the optimization techniques interact (multiplicative vs additive gains)?

---

## Verification

To confirm understanding:
- [ ] Can explain Muon optimizer in your own words
- [ ] Understand why paired head attention helps
- [ ] Know the role of each major component in train_gpt.py
- [ ] Can discuss tradeoffs of sliding window attention
- [ ] Understand FP8/BF16 precision tradeoffs
