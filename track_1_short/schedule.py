"""Per-step schedules: stage lookup, learning rate, MTP/prefix weights, ANVIL rail beta."""
from itertools import accumulate, pairwise

import torch

from track_1_short.config import SCHEDULE_GROWTH_STAGE, SCHEDULE_GROWTH_STEPS, TrainingStage, scaled_steps

# The lr cooldown decays linearly to this absolute multiplier (record #360's code; its README says 0.20).
LR_FLOOR = 0.30

class TrainingSchedule:
    """
    Training schedule initialized via TRAINING_STAGES
        1. Multi Token Prediction schedule of [1, 0.5, 0.25->0] -> [1, 0.5->0] -> [1] @varunneal
        2. Prefix-token prediction weight 0.25 -> 0.20/0.15/0.10/0.05 (quarters of stage 1) -> 0
        3. Sliding Attention window schedule of [1,3] -> [3,7] -> [5,11] -> [5,11] -> [6,13]
        4. YaRN updates to RoPE on window changes
        5. Split embed and lm head at the start of the extension stage
        6. Batch size schedule of 8 -> 16 -> 24 -> 20 (terminal taper) -> 8 (extension)
        7. Post training extension of long windows from 13 to 20
        8. Seq len updates 896 -> 2048 -> 3072 at the first two stage boundaries
    """

    def __init__(self, stages: list[TrainingStage], scheduled_iterations: int, extension_iterations: int,
                 device: torch.device, cooldown_frac: float, split_embed_stage: int, ws_post_yarn_ext: int):
        self.stages = stages
        # The main stages run the base count plus the growth steps; the lr cooldown spans all of it.
        self.scheduled_iterations = scheduled_iterations + SCHEDULE_GROWTH_STEPS
        self.cooldown_frac = cooldown_frac
        # increase final validation ws, used for YaRN extension and short window size @classiclarryd
        self.ws_post_yarn_ext = ws_post_yarn_ext

        self.total_steps = self.scheduled_iterations + extension_iterations

        # Stage ends from the durations over the base count, then every end after the growth stage
        # shifts by the growth steps (the last is extension stage, ending at total_steps).
        ends = [0, *[round(c * scheduled_iterations) for c in accumulate(s.duration for s in stages[:-1])], self.total_steps]
        ends[SCHEDULE_GROWTH_STAGE + 1:-1] = [e + SCHEDULE_GROWTH_STEPS for e in ends[SCHEDULE_GROWTH_STAGE + 1:-1]]
        assert self.scheduled_iterations == ends[-2]
        self.boundaries = list(pairwise(ends))

        # Split embed at specified stage (ensure odd step for Adam)
        self.split_step = self.boundaries[split_embed_stage][0] | 1

        # Precompute MTP weights and prefix-prediction weights for all steps
        self.mtp_weights = []
        self.prefix_weights = []
        for step in range(self.total_steps + 1):
            stage, t = self.lookup(step)
            w = [a + (b - a) * t for a, b in zip(stage.mtp_weights_start, stage.mtp_weights_end)]
            self.mtp_weights.append(torch.tensor(w, device=device))
            self.prefix_weights.append(torch.tensor([self.prefix_weight(step)], device=device))

    def lookup(self, step: int) -> tuple[TrainingStage, float]:
        # Returns stage and % of the way through that stage
        for i, (start, end) in enumerate(self.boundaries):
            if step < end:
                t = (step - start) / (end - start)
                return self.stages[i], t
        return self.stages[-1], 1.0

    def prefix_weight(self, step: int) -> float:
        stage, t = self.lookup(step)
        n = len(stage.prefix_weights)
        return stage.prefix_weights[min(n - 1, int(t * n))]

    def get_lr(self, step: int) -> float:
        # learning rate schedule: tied to batch size schedule, with cooldown at the end
        stage, _ = self.lookup(step)
        lr = stage.lr_mul
        cd_start = int(self.scheduled_iterations * (1 - self.cooldown_frac))
        if step >= cd_start:
            t = min(1.0, (step - cd_start) / (self.scheduled_iterations - cd_start))
            lr = lr * (1 - t) + LR_FLOOR * t
        return lr


def get_rail_beta(step: int, total_steps: int, beta_warmup_steps=scaled_steps(240), beta_cooldown_steps=scaled_steps(50), beta_min=0.85, beta_max=0.93):
    """ANVIL's fast-rail beta, also its Nesterov lookahead: linear warmup from beta_min to beta_max,
    flat for the bulk of the run, linear cooldown back over the last beta_cooldown_steps."""
    beta_cd_start = total_steps - beta_cooldown_steps
    if step < beta_warmup_steps:
        frac = step / beta_warmup_steps
        beta = beta_min + frac * (beta_max - beta_min)
    elif step > beta_cd_start:
        frac = (step - beta_cd_start) / beta_cooldown_steps
        beta = beta_max - frac * (beta_max - beta_min)
    else:
        beta = beta_max
    return beta
