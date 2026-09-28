"""Run configuration: hyperparameters and the training-stage table.

The run's own environment variables (DATA_PATH, NUM_SCHEDULED_ITERATIONS, TRAIN_SEED) are read here.
The others: torchrun's RANK / LOCAL_RANK / WORLD_SIZE (distributed.py; train_gpt.py also reads
LOCAL_RANK), PYTORCH_ALLOC_CONF (set by train_gpt.py), and CUDA_HOME (perf/kernels/cross_entropy.py,
for the CE kernel's CUDA headers).

Provenance: the five-stage table (batch taper, batch-8 extension, seq 3072 under a 2560 attention
cap, 52 growth steps) is record #360's.
"""
import os
import uuid
from dataclasses import dataclass

@dataclass(slots=True)
class Hyperparameters:
    # data
    data_path = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin") # input .bin to train on
    val_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin") # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    val_batch_size: int = 4 * 64 * 1024 * 8
    # schedule: the base step count of the main stages (SCHEDULE_GROWTH_STEPS are added on top), then
    # the extension stage. The override is for step-count sweeps (record #360's KX_STEPS).
    num_scheduled_iterations: int = int(os.environ.get("NUM_SCHEDULED_ITERATIONS", "1122"))
    num_extension_iterations: int = 20
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    # Every how many steps to evaluate val loss; 0 = only at the end (record #360: each intermediate
    # validation is on the clock -- the val reads and n-gram pulls, the value_embeds gather, the drained
    # CUDA-graph run-ahead and the cycle re-land after it cost ~0.2-0.5 s over a run at 250).
    val_loss_every: int = 0
    save_checkpoint: bool = False
    run_evals: bool = False  # run additional evaluations after training is completed
    # reproducibility: unset means a random init; any integer (0 included) seeds it
    train_seed: int | None = int(os.environ["TRAIN_SEED"]) if "TRAIN_SEED" in os.environ else None

@dataclass(slots=True)
class TrainingStage:
    lr_mul: float
    batch_size: int
    window_sizes: tuple[int, int]  # (short, long) in block units
    mtp_weights_start: list[float]
    mtp_weights_end: list[float]
    train_max_seq_len: int  # documents are cut to this many tokens
    # Prefix-token-prediction loss weight, stepped: the stage's steps split into equal parts, one weight each.
    prefix_weights: tuple[float, ...]
    duration: float | None = None  # fraction of num_scheduled_iterations; None for the extension stage


# Stage lengths as fractions of num_scheduled_iterations (normalized to sum to 1). The last entry is
# the batch-24 stage plus the batch-20 taper, which takes TAPER_DURATION of it.
STAGE_DURATIONS = [0.2854, 0.3209, 0.3931]
STAGE_DURATIONS = [d / sum(STAGE_DURATIONS) for d in STAGE_DURATIONS]
TAPER_DURATION = 0.06
# Record #360 grew its run by 52 steps without re-deriving the fractions: the extra steps all go to
# the batch-24 stage (index 2), which keeps every later boundary's Adam-step (odd/even) parity.
SCHEDULE_GROWTH_STEPS = 52
SCHEDULE_GROWTH_STAGE = 2
# Attention window sizes are counted in blocks of this many tokens (record #360).
BLOCK_SIZE = 128
# Documents may run to 3072 tokens in the last stages, but attention segments are cut at this many
# tokens (the loader adds segment boundaries; the token stream is untouched). A whole number of blocks.
VIRTUAL_SEQ_CAP = 2560
TAPER_BATCH_UNITS = 20
# The lr decays linearly to its floor over the last LR_COOLDOWN_FRAC of the main stages (record #360).
LR_COOLDOWN_FRAC = 0.80
# embed unties from lm_head at the start of this stage, the extension stage (record #360).
SPLIT_EMBED_STAGE = 4
# The final validation extends the long attention window to this many blocks (record #360).
WS_POST_YARN_EXT = 20

# window_sizes are in units of BLOCK_SIZE tokens.
TRAINING_STAGES = [
    TrainingStage(duration=STAGE_DURATIONS[0], train_max_seq_len=896, batch_size=8 * 2048 * 8, window_sizes=(1, 3), lr_mul=1.0,
                  mtp_weights_start=[1.0, 0.5, 0.25], mtp_weights_end=[1.0, 0.5, 0.0], prefix_weights=(0.25,)),
    TrainingStage(duration=STAGE_DURATIONS[1], train_max_seq_len=2048, batch_size=16 * 2048 * 8, window_sizes=(3, 7), lr_mul=1.52,  # (16/8)**0.6
                  mtp_weights_start=[1.0, 0.5], mtp_weights_end=[1.0, 0.0], prefix_weights=(0.20, 0.15, 0.10, 0.05)),
    TrainingStage(duration=STAGE_DURATIONS[2] - TAPER_DURATION, train_max_seq_len=3072, batch_size=24 * 2048 * 8, window_sizes=(5, 11), lr_mul=1.73,  # (24/8)**0.5
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0], prefix_weights=(0.0,)),
    # terminal batch taper: 24 -> 20, lr_mul scaled by sqrt(20/24)
    TrainingStage(duration=TAPER_DURATION, train_max_seq_len=3072, batch_size=TAPER_BATCH_UNITS * 2048 * 8, window_sizes=(5, 11),
                  lr_mul=1.73 * (TAPER_BATCH_UNITS / 24) ** 0.5,
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0], prefix_weights=(0.0,)),
    # extension stage at batch 8: wall-matched, ~18 steps at batch 8 cost what 7 do at batch 24
    TrainingStage(train_max_seq_len=3072, batch_size=8 * 2048 * 8, window_sizes=(6, 13), lr_mul=1.0,  # lr_mul is not used (lr sits at the floor)
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0], prefix_weights=(0.0,)),
]
assert VIRTUAL_SEQ_CAP % BLOCK_SIZE == 0
assert any(s.train_max_seq_len > VIRTUAL_SEQ_CAP for s in TRAINING_STAGES), "VIRTUAL_SEQ_CAP would be inert"
