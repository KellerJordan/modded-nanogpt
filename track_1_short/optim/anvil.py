"""ANVIL (for projection matrices) + Adam (for everything else), with explicit comms scheduling."""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist
from torch import Tensor, nn

from track_1_short.perf.bank_scalars import BankScalarStaging
from track_1_short.perf.kernels.polar_express import XTX, XXT, ba_plus_cAA
from track_1_short.perf.kernels.transpose import transpose_add, transpose_copy
from track_1_short.perf.replicated_adam import ReplicatedAdam

# The row-sparse updates run at value_embeds' place in scatter_order and work_order (value_embeds is one
# of them; step() never sees its gradient).
SPARSE_UPDATE_LABEL = "value_embeds"


class SparseUpdate(Protocol):
    """Row-sparse updates outside the dense reduce/gather path (the n-gram table, value_embeds), slotted
    into step()'s comms schedule so their exchanges overlap the dense ones."""
    def launch(self) -> None:
        """At SPARSE_UPDATE_LABEL's place in scatter_order: start the gradient exchange."""
    def update(self) -> None:
        """At SPARSE_UPDATE_LABEL's place in work_order: wait for the exchange, update, and start serving
        the next cycle's rows (queued on the NCCL stream ahead of the gathers launched after it)."""
    def finish(self) -> None:
        """After the lm_head gather: land what the next steps read before any deferred gather is waited."""

# -----------------------------------------------------------------------------
# ANVIL: twin-rail momentum + whitening cascade (record #360)

# Six quintic spectral maps applied to the velocity Gram, re-derived for this run rather than
# taken from the Polar Express reference (https://arxiv.org/pdf/2505.16932).
ANVIL_MAPS = [
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
]

# Twin-rail velocity. Rail 0 is fast (beta scheduled by get_rail_beta until RAIL_ENGAGE_STEP, then
# RAIL_FAST_BETA); rail 1 is slow (RAIL_SLOW_BETA) and accumulates from step 0. Until the engage
# step the update reads the fast rail only; after it, RAIL_FAST_WEIGHT * fast + (1 - w) * slow.
RAIL_FAST_BETA, RAIL_SLOW_BETA, RAIL_FAST_WEIGHT, RAIL_ENGAGE_STEP = 0.85, 0.98, 0.4385, 514

# The cascade's input is divided by FROBENIUS_MARGIN * ||X||_F + FROBENIUS_EPS, which puts every
# singular value safely below 1 where the maps converge (the margin and eps of record #360).
FROBENIUS_MARGIN, FROBENIUS_EPS = 1.05, 1e-6
# Banks whose matrices have more rows than this run the a*X + X@B step as two kernels instead of one
# baddbmm (see anvil_cascade): true for mlp_bank (2816 rows), false for qk_bank (256) and vo_bank (768).
SPLIT_BADDBMM_MIN_ROWS = 1024


@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def anvil_cascade(grad_chunk: torch.Tensor, velocity: torch.Tensor, momentum_t: torch.Tensor,
                  split_baddbmm: bool, fast_beta_t: torch.Tensor, fast_weight_t: torch.Tensor):
    """Twin-rail Nesterov momentum, then the ANVIL cascade that drives every singular value to ~1.

    velocity is one fp32 [2, *chunk] tensor: rail 0 fast, rail 1 slow. Their blend gets a Nesterov
    lookahead (momentum_t), is cast to bf16, normalized by its Frobenius norm, and whitened by
    ANVIL_MAPS. momentum_t, fast_beta_t and fast_weight_t are 0-D device tensors (AnvilBank), so
    per-step changes never recompile and a captured graph reads their current values.
    """
    grad_chunk = grad_chunk.float()
    momentum = momentum_t.to(grad_chunk.dtype)
    velocity[0].lerp_(grad_chunk, 1 - fast_beta_t.to(grad_chunk.dtype))
    velocity[1].lerp_(grad_chunk, 1 - RAIL_SLOW_BETA)
    w = fast_weight_t.to(grad_chunk.dtype)
    blend = w * velocity[0] + (1 - w) * velocity[1]
    g = grad_chunk.lerp_(blend, momentum)

    X = g.bfloat16().contiguous()
    is_tall = g.size(-2) > g.size(-1)

    # The first Gram is taken on the unnormalized X: its trace is ||X||_F^2, which gives the
    # normalization for free; X and the Gram are then rescaled instead of recomputing.
    if is_tall:
        # Tall: use Triton kernels with X^T @ X (small) and right multiplication
        A = torch.empty((*X.shape[:-2], X.size(-1), X.size(-1)), device=X.device, dtype=X.dtype)
        XTX(X, out=A)  # A = X.T @ X
        tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
        d = tr.sqrt() * FROBENIUS_MARGIN + FROBENIUS_EPS
        X = (X.float() / d).bfloat16()
        A = (A.float() / d.square()).bfloat16()
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            XB_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_XB = torch.baddbmm if X.ndim > 2 else torch.addmm

        for k, (a, b, c) in enumerate(ANVIL_MAPS):
            if k > 0:
                XTX(X, out=A)  # A = X.T @ X
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b*A + c*(A@A)

            # Referencing X twice causes pytorch to make a defensive copy,
            # resulting in a cudaMemcpyAsync in baddbmm.
            # For large matrices (i.e., the mlp weights), it's faster to split
            # the operation into two kernels to avoid this.
            if split_baddbmm:
                XB_matmul(X, B, out=C)  # C = X @ B
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_XB(X, X, B, beta=a, out=C)  # C = a * X + X @ B

            X, C = C, X  # Swap references to avoid unnecessary copies
    else:
        # Wide: use Triton kernels with X @ X^T (small) and left multiplication
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        XXT(X, out=A)  # A = X @ X.mT
        tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
        d = tr.sqrt() * FROBENIUS_MARGIN + FROBENIUS_EPS
        X = (X.float() / d).bfloat16()
        A = (A.float() / d.square()).bfloat16()
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

        for k, (a, b, c) in enumerate(ANVIL_MAPS):
            if k > 0:
                XXT(X, out=A)  # A = X @ X.mT
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A

            if split_baddbmm:
                BX_matmul(B, X, out=C)  # C = B @ X
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X

            X, C = C, X  # Swap references to avoid unnecessary copies

    return X

# -----------------------------------------------------------------------------
# Combined ANVIL + Adam Optimizer

@dataclass(slots=True)
class ParamConfig:
    """Per-parameter configuration for AnvilAndAdam."""
    label: str
    optim: str  # "adam" or "anvil"
    comms: str  # "replicated" or "sharded"
    adam_betas: tuple[float, float] | None
    lr_mul: float
    wd_mul: float
    lr: float
    initial_lr: float
    weight_decay: float
    # Adam-specific
    eps: float | None = None
    # ANVIL-specific
    reshape: tuple | None = None
    chunk_size: int | None = None
    momentum: float | None = None
    beta2: float | None = None
    per_matrix_lr_mul: list[float] | None = None


@dataclass(slots=True)
class AnvilBank:
    """One ANVIL bank's update on this rank: every tensor anvil_bank_update reads or writes.

    The optimizer-tail CUDA graphs (perf/cuda_graphs/optimizer_graphs.py) bake these addresses, so each is
    allocated once and only ever written in place: the reduce-scatter lands in `grad`, and the per-step
    scalars are device tensors refreshed by one H2D at the top of every step() (perf/bank_scalars.py).
    """
    label: str
    grad: Tensor          # [chunk, rows, cols] bf16: this rank's reduce-scattered gradient
    velocity: Tensor      # [2, chunk, rows, cols] fp32: fast and slow rail
    lane_energy: Tensor   # the equalizer's per-lane EMA
    mantissa: Tensor      # uint16: low half of the fp32 master weights
    p_slice: Tensor       # this rank's slice of the bank (a bf16 view of the parameter)
    momentum: Tensor      # 0-D fp32: the Nesterov lookahead (the scheduled rail beta)
    eff_wd: Tensor        # 0-D fp32: wd_mul * weight_decay * lr
    fast_beta: Tensor     # 0-D fp32: the fast rail's beta
    fast_weight: Tensor   # 0-D fp32: the fast rail's weight in the blend
    eff_lr: Tensor        # [chunk, 1, 1] fp32: lr_mul * per-matrix multiplier * lr
    beta2: float          # the equalizer's EMA decay
    split_baddbmm: bool
    red_dim: int

    def mutated(self) -> dict[str, Tensor]:
        """What the update writes."""
        return {"velocity": self.velocity, "lane_energy": self.lane_energy, "mantissa": self.mantissa,
                "p_slice": self.p_slice}


class AnvilAndAdam:
    """
    Combined optimizer that handles both ANVIL (for projection matrices) and
    Adam (for embeddings/scalars/gate weights).

    ANVIL (record #360), differences from standard Muon (https://kellerjordan.github.io/posts/muon/):
    - Twin-rail momentum: a fast and a slow velocity EMA, blended after RAIL_ENGAGE_STEP
    - Newton-Schulz is replaced with the ANVIL cascade: six re-derived quintic maps after a
      Frobenius normalization (successor of Polar Express)
    - Per-lane energy equalizer, the low-rank variance estimator from NorMuon
      (https://arxiv.org/pdf/2510.05491)
    - Cautious weight decay gated on the slow rail's sign
    - Mantissa tracking for precision

    Adam (for embeddings/scalars/gates):
    - Standard Adam with bias correction
    - Cautious weight decay

    Configuration:
    Unlike torch.optim.Optimizer, this class uses per-parameter configs from a `param_table` dict
    and does not include parameter "groups". All parameters require a .label attribute, and a
    corresponding entry in the param_table to specify their hyperparameters (lr_mul, wd_mul, adam_betas, etc.).

    Communication and ordering:
    Gradient communication is explicitly scheduled rather than hook-driven.
    Reductions are launched in `scatter_order`, while update math and final
    gathers are executed in `work_order`. These orders are independent and
    must each contain every parameter label exactly once.

    Two communication modes are supported per parameter:
    - 'replicated': Gradients are all-reduced and each rank computes the full update.
    - 'sharded': Gradients are reduce-scattered, each rank updates its shard,
      and results are all-gathered.

    Adam parameters may be freely sharded. ANVIL operates on full matrices; sharding is
    supported by grouping matrices into parameter banks. ANVIL parameters must have a
    `.reshape` attribute that reshapes the bank so that the leading dimension is divisible
    by world_size.

    # Contributors include @YouJiacheng, @KonstantinWilleke, @alexrgilbert, @adricarda,
    # @tuttyfrutyee, @vdlad, @ryanyang0, @vagrawal, @varunneal, @chrisjmccormick
    """
    def __init__(self, named_params, param_table: dict, scatter_order: list, work_order: list,
                 adam_defaults: dict, anvil_defaults: dict, bank_update: Callable[[AnvilBank], None]):
        """bank_update(bank) runs one bank's ANVIL update: anvil_bank_update, or its captured graph."""
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Store defaults for each optimizer type
        self.adam_defaults = adam_defaults
        self.anvil_defaults = anvil_defaults
        self.param_table = param_table
        self.scatter_order = scatter_order
        self.work_order = work_order

        # Collect params by label and build config
        self.param_cfgs: dict[nn.Parameter, ParamConfig] = {}
        self.param_states: dict[nn.Parameter, dict] = {}
        self._param_by_label: dict[str, nn.Parameter] = {}
        for name, param in named_params:
            label = getattr(param, "label", None)
            assert label is not None and label in param_table  # all params must have valid label
            assert label not in self._param_by_label  # exactly one param per label
            self._param_by_label[label] = param
            self._build_param_cfg(param, label)

        # Assert scatter_order and work_order match present labels exactly
        present = self._param_by_label.keys()
        assert set(scatter_order) == present and set(work_order) == present

        # The replicated Adam params are reduced and updated as flat buffers (perf/replicated_adam.py).
        replicated = [p for p, c in self.param_cfgs.items() if c.optim == "adam" and c.comms == "replicated"]
        self.replicated_labels = {self.param_cfgs[p].label for p in replicated}

        # Initialize state for all params
        self._init_state()
        self.replicated = ReplicatedAdam(replicated, self.param_cfgs, self.param_states, device=replicated[0].device)

        # Adam's per-parameter scalars: 0-D CPU tensors to avoid recompilation
        self._step_size_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # ANVIL's twin-rail scalars for this step (set_rails)
        self.fast_beta, self.fast_weight = 0.0, 1.0
        self._init_banks()
        self.bank_update = bank_update

        # Track async operations
        self._reduce_futures: dict[nn.Parameter, tuple] = {}

        # Embed/lm_head tying state
        self.split_embed = False
        self._lm_head_param = self._param_by_label.get("lm_head")
        self._embed_param = self._param_by_label.get("embed")

    def _build_param_cfg(self, param: nn.Parameter, label: str):
        """Build config for a single parameter from param_table."""
        table_entry = self.param_table[label]
        optim = table_entry["optim"]
        comms = table_entry["comms"]
        adam_betas = table_entry.get("adam_betas")
        lr_mul = table_entry.get("lr_mul", 1.0)
        wd_mul = table_entry.get("wd_mul", 1.0)

        if optim == "adam":
            chunk_size = param.shape[0] // self.world_size if comms == "sharded" else None
            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.adam_defaults["lr"],
                initial_lr=self.adam_defaults["lr"],
                weight_decay=self.adam_defaults["weight_decay"],
                eps=self.adam_defaults["eps"],
                chunk_size=chunk_size,
            )
        elif optim == "anvil":
            reshape = getattr(param, "reshape", None)
            if reshape is None:
                raise ValueError(f"ANVIL param {label} must have .reshape attribute")
            if reshape[0] % self.world_size != 0:
                raise ValueError(f"reshape[0]={reshape[0]} must be divisible by world_size")

            chunk_size = reshape[0] // self.world_size
            chunk_shape = (chunk_size, *reshape[1:])
            # Shape-based LR multiplier for ANVIL
            shape_mult = max(1.0, chunk_shape[-2] / chunk_shape[-1]) ** 0.5 if len(chunk_shape) >= 2 else 1.0
            lr_mul = shape_mult * lr_mul

            # Per-matrix LR multipliers for MLP c_proj (2x LR on odd indices). Matrices the model
            # marks as frozen (`param.frozen_matrices`, e.g. a removed layer's MLP) get LR 0, which
            # also zeroes their decay term: they do not move at all.
            per_matrix_lr_mul = None
            if label == "mlp_bank":
                start_idx = self.rank * chunk_size
                frozen = getattr(param, "frozen_matrices", frozenset())
                per_matrix_lr_mul = []
                for i in range(chunk_size):
                    global_idx = start_idx + i
                    is_c_proj = (global_idx % 2 == 1)
                    per_matrix_lr_mul.append(0.0 if global_idx in frozen else 2.0 if is_c_proj else 1.0)

            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.anvil_defaults["lr"],
                initial_lr=self.anvil_defaults["lr"],
                weight_decay=self.anvil_defaults["weight_decay"],
                reshape=reshape,
                chunk_size=chunk_size,
                momentum=self.anvil_defaults["momentum"],
                beta2=self.anvil_defaults["beta2"],
                per_matrix_lr_mul=per_matrix_lr_mul,
            )
        else:
            raise ValueError(f"Unknown optim type: {optim}")

        self.param_cfgs[param] = p_cfg

    def _init_state(self):
        """Initialize optimizer state for all parameters."""
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam":
                # Sharded params use chunk state, replicated use full state
                if p_cfg.comms == "sharded":
                    chunk = param[:p_cfg.chunk_size]
                else:
                    chunk = param
                exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=param.device)
                self.param_states[param] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg))

            elif p_cfg.optim == "anvil":
                chunk_shape = (p_cfg.chunk_size, *p_cfg.reshape[1:])

                # Twin-rail velocity (FP32 for precision): [0] fast rail, [1] slow rail
                velocity = torch.zeros(
                    (2, *chunk_shape), dtype=torch.float32, device=param.device
                )

                # Per-lane update energy for the equalizer - reduced along the longer dimension
                if chunk_shape[-2] >= chunk_shape[-1]:
                    lane_shape = (*chunk_shape[:-1], 1)
                else:
                    lane_shape = (*chunk_shape[:-2], 1, chunk_shape[-1])
                lane_energy = torch.zeros(
                    lane_shape, dtype=torch.float32, device=param.device
                )

                # Mantissa buffer for precision tracking
                mantissa = torch.zeros(
                    chunk_shape, dtype=torch.uint16, device=param.device
                )

                self.param_states[param] = dict(
                    velocity=velocity,
                    lane_energy=lane_energy,
                    mantissa=mantissa,
                )

    # -----------------------------------
    # ANVIL banks: fixed buffers and device scalars

    def _init_banks(self):
        """One AnvilBank per ANVIL parameter, with its persistent gradient buffer and its per-step
        scalars: views into one device buffer that one H2D per step refreshes (perf/bank_scalars.py)."""
        anvil = [(p, c) for p, c in self.param_cfgs.items() if c.optim == "anvil"]
        device = anvil[0][0].device
        self.bank_scalars = BankScalarStaging([c.chunk_size for _, c in anvil], device)
        self.banks: dict[str, AnvilBank] = {}
        for i, (param, cfg) in enumerate(anvil):
            # bf16, so anvil_cascade's grad.float() is a copy and its in-place lerp never writes `grad`.
            assert param.dtype == torch.bfloat16, f"{cfg.label} must be bf16 before the optimizer is built"
            chunk_shape = (cfg.chunk_size, *cfg.reshape[1:])
            self.banks[cfg.label] = AnvilBank(
                label=cfg.label,
                grad=torch.empty(chunk_shape, dtype=param.dtype, device=device),
                **self.live_bank_state(cfg.label),
                **self.bank_scalars.scalars(i),
                eff_lr=self.bank_scalars.eff_lr(i),
                beta2=cfg.beta2,
                split_baddbmm=chunk_shape[-2] > SPLIT_BADDBMM_MIN_ROWS,
                red_dim=-1 if chunk_shape[-2] >= chunk_shape[-1] else -2,
            )

    def live_bank_state(self, label: str) -> dict[str, Tensor]:
        """The bank's state tensors as the live optimizer and parameter hold them now (the one definition
        both the bank and the graphs' address check use)."""
        param = self._param_by_label[label]
        cfg, state = self.param_cfgs[param], self.param_states[param]
        lo = self.rank * cfg.chunk_size
        return dict(velocity=state["velocity"], lane_energy=state["lane_energy"], mantissa=state["mantissa"],
                    p_slice=param.data.view(cfg.reshape)[lo:lo + cfg.chunk_size])

    def stage_bank_scalars(self):
        """Upload this step's ANVIL scalars (from the current ParamConfigs and rails) in one H2D."""
        fields, eff_lrs = [], []
        for label in self.banks:
            cfg = self.param_cfgs[self._param_by_label[label]]
            # The products keep the eager code's order (lr_mul * matrix multiplier * lr).
            fields.append((cfg.momentum, cfg.wd_mul * cfg.weight_decay * cfg.lr, self.fast_beta, self.fast_weight))
            per_matrix = cfg.per_matrix_lr_mul or [1.0] * cfg.chunk_size
            eff_lrs.append([cfg.lr_mul * m * cfg.lr for m in per_matrix])
        self.bank_scalars.upload(fields, eff_lrs)

    # -----------------------------------
    # Reduce/Gather operations

    def _launch_reduce(self, param: nn.Parameter, grad: Tensor):
        """Launch async reduce for a parameter based on its comms policy."""
        p_cfg = self.param_cfgs[param]
        if p_cfg.comms == "sharded":
            if p_cfg.optim == "anvil":
                # ANVIL: reshape before reduce_scatter, into the bank's persistent gradient buffer
                grad_chunk = self.banks[p_cfg.label].grad
                future = dist.reduce_scatter_tensor(
                    grad_chunk, grad.view(p_cfg.reshape).contiguous(), op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                self._reduce_futures[param] = (future, grad_chunk)
            else:
                # Adam: simple reduce_scatter
                grad_chunk = torch.empty_like(grad[:p_cfg.chunk_size])
                future = dist.reduce_scatter_tensor(
                    grad_chunk, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                self._reduce_futures[param] = (future, grad_chunk)

    def _launch_gather(self, param: nn.Parameter, p_slice: Tensor) -> "torch.futures.Future":
        """Launch async all_gather for a sharded parameter."""
        p_cfg = self.param_cfgs[param]
        if p_cfg.optim == "anvil":
            full_param = param.data.view(p_cfg.reshape)
            assert full_param.is_contiguous()
            return dist.all_gather_into_tensor(
                full_param, p_slice.contiguous(), async_op=True
            ).get_future()
        else:
            return dist.all_gather_into_tensor(
                param, p_slice.contiguous(), async_op=True
            ).get_future()

    # -----------------------------------
    # State management

    def reset(self):
        """Reset ANVIL velocity/lane state and split_embed state (called on training reset)."""
        self.split_embed = False
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "anvil":
                p_state = self.param_states[param]
                p_state["velocity"].zero_()
                p_state["mantissa"].zero_()
                p_state["lane_energy"].zero_()

    def copy_lm_state_to_embed(self):
        """
        Copy the optimizer state from the lm_head to the embed at the untie point.
        This requires an all-gather + reshard because of different sharding:
        - lm_head (768, 50304) is sharded to (96, 50304) per rank (along model_dim)
        - embed (50304, 768) is sharded to (6288, 768) per rank (along vocab_size)

        We all-gather the lm_head momentum, transpose it, then each rank takes their
        embed shard to get the correct momentum state.
        """
        lm_head = self._lm_head_param
        embed = self._embed_param
        lm_state = self.param_states[lm_head]
        embed_state = self.param_states[embed]
        embed_cfg = self.param_cfgs[embed]

        embed_state['step'] = lm_state['step'] # Preserve step count for bias correction

        # Copy optimizer state with all-gather + transpose + reshard
        embed_chunk_size = embed_cfg.chunk_size  # 6288
        # All-gather lm_head momentum to get full (768, 50304) tensor
        for key in ["exp_avg", "exp_avg_sq"]:
            lm_chunk = lm_state[key]  # (96, 50304)
            full_lm = torch.empty(lm_head.shape[0], lm_head.shape[1], dtype=lm_chunk.dtype, device=lm_chunk.device)
            dist.all_gather_into_tensor(full_lm, lm_chunk.contiguous())
            embed_state[key].copy_(full_lm.T[self.rank * embed_chunk_size:(self.rank + 1) * embed_chunk_size])

        # Mark as split
        self.split_embed = True

    def state_dict(self):
        """Return the optimizer state as a dict."""
        return {
            "param_states": {id(p): s for p, s in self.param_states.items()},
            "param_cfgs": {id(p): s for p, s in self.param_cfgs.items()},
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state from a dict. Tensors are copied in place: the replicated params' moments
        are views into ReplicatedAdam's flat buffers and must never be rebound."""
        # Build id->param mapping
        id_to_param = {id(p): p for p in self.param_cfgs}

        for param_id, saved_p_state in state_dict["param_states"].items():
            if param_id in id_to_param:
                param = id_to_param[param_id]
                p_state = self.param_states[param]
                for k, v in saved_p_state.items():
                    if isinstance(v, torch.Tensor) and k in p_state:
                        p_state[k].copy_(v)
                    else:
                        p_state[k] = v

    # -----------------------------------
    # Unified optimizer step with explicit ordering

    @torch.no_grad()
    def step(self, do_adam: bool, sparse_update: SparseUpdate | None,
             deferred_labels: frozenset[str]) -> dict[str, torch.futures.Future]:
        """
        Combined optimizer step with explicit ordering.

        Args:
            do_adam: If True, update Adam params. ANVIL params always updated.
            sparse_update: this step's row-sparse updates, if any (their hooks run at the points below).
            deferred_labels: sharded labels whose all-gather this step launches but does not wait for.

        Returns:
            label -> gather future for each deferred label that gathered. The caller must wait each one
            before the first reader of that parameter's full replica (perf/deferred_gathers.py).

        Flow (NCCL runs one stream in enqueue order and future.wait() is a stream wait, so the order
        in which collectives are launched is the comms schedule):
        0. Adam steps: the replicated params' flat all_reduce (perf/replicated_adam.py)
        1. Scatter phase: Launch reduces in scatter_order (sparse_update.launch at SPARSE_UPDATE_LABEL)
        2. Work phase: Process updates in work_order
           - Wait for reduce, compute update, launch gather
           - the first replicated label runs every replicated param's update at once
           - sparse_update.update at SPARSE_UPDATE_LABEL
        3. Finalize phase: wait for the lm_head gather, sparse_update.finish, wait for the gathers
           that are not deferred

        While the embeddings are tied:
        - Comms and update math are only done on lm_head.
        - We add embed.grad.T into lm_head.grad before comms.
        - After lm_head gather, we copy lm_head.data.T --> embed.data
        """
        lm_param, embed_param = self._lm_head_param, self._embed_param
        # Outside any graph: the ANVIL scalars the bank updates (or their replays) read this step.
        self.stage_bank_scalars()

        # ===== Phase 0: one flat all_reduce per dtype for the replicated params =====
        if do_adam:
            self.replicated.launch_reduce()

        # ===== Phase 1: Launch reduces in scatter_order =====
        for label in self.scatter_order:
            if label == SPARSE_UPDATE_LABEL and sparse_update is not None:
                sparse_update.launch()
            param = self._param_by_label[label]
            p_cfg = self.param_cfgs[param]

            if p_cfg.optim == "adam" and not do_adam:
                continue
            if param.grad is None or label in self.replicated_labels:
                continue

            # lm_head when tied: aggregate embed.grad.T (tiled Triton transpose-add)
            if label == "lm_head" and do_adam and not self.split_embed:
                if embed_param is not None and embed_param.grad is not None:
                    transpose_add(embed_param.grad, param.grad)

            # Skip embed when tied (copied from lm_head after gather)
            if label == "embed" and not self.split_embed:
                continue

            self._launch_reduce(param, param.grad)

        # ===== Phase 2: Process updates in work_order =====
        gather_futures = []
        deferred_futures = {}
        lm_head_gather_future = None
        replicated_done = False

        for label in self.work_order:
            if label == SPARSE_UPDATE_LABEL and sparse_update is not None:
                sparse_update.update()
            if label in self.replicated_labels:
                # The replicated params are independent, so one fused pass at the first of them equals
                # updating each in turn; they launch no gather.
                if do_adam and not replicated_done:
                    self.replicated.update()
                    replicated_done = True
                continue
            param = self._param_by_label[label]
            if param not in self._reduce_futures:
                continue

            p_cfg = self.param_cfgs[param]
            if p_cfg.optim == "adam" and not do_adam:
                continue
            # Wait for reduce
            future, grad_chunk = self._reduce_futures[param]
            future.wait()

            # Apply update based on optim type
            if p_cfg.optim == "adam":
                p_slice = self._adam_update(param, grad_chunk, p_cfg)
            else:
                p_slice = self._anvil_update(p_cfg, grad_chunk)
            # Launch gather for sharded params
            if p_cfg.comms == "sharded":
                gather_fut = self._launch_gather(param, p_slice)
                if label in deferred_labels:
                    deferred_futures[label] = gather_fut
                elif label == "lm_head":
                    lm_head_gather_future = gather_fut
                else:
                    gather_futures.append(gather_fut)

        # ===== Phase 3: Wait for gathers, sync embed if tied =====
        # Wait for lm_head gather first so we can copy to embed while other gathers complete
        if lm_head_gather_future is not None:
            lm_head_gather_future.wait()

        # When tied: copy lm_head.T to embed (tiled Triton transpose for coalesced writes)
        if do_adam and not self.split_embed and embed_param is not None and lm_param is not None:
            transpose_copy(lm_param.data, embed_param.data)

        if sparse_update is not None:
            sparse_update.finish()

        # Wait for the remaining gathers that are not deferred
        for fut in gather_futures:
            fut.wait()

        self._reduce_futures.clear()

        # Clear grads for updated params
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam" and not do_adam:
                continue  # Don't clear Adam grads on even steps
            param.grad = None
        return deferred_futures

    # -----------------------------------
    # Adam update

    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig) -> Tensor:
        """Apply Adam update to a parameter. Returns the updated p_slice."""
        beta1, beta2 = p_cfg.adam_betas
        lr = p_cfg.lr * p_cfg.lr_mul

        # Get parameter slice
        if p_cfg.comms == "sharded":
            p_slice = param[self.rank * p_cfg.chunk_size:(self.rank + 1) * p_cfg.chunk_size]
        else:
            p_slice = param

        p_state = self.param_states[param]
        p_state["step"] += 1
        t = p_state["step"]

        bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))
        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)

        AnvilAndAdam._adam_update_step(
            p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],
            beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
        )

        return p_slice

    @torch.no_grad()
    def adam_update_shard(self, param: nn.Parameter, grad_chunk: Tensor):
        """Adam on this rank's shard of a sharded parameter whose gradient was reduced outside step()
        (value_embeds: perf/value_embed_pull.py). Same update, state and config as in step()."""
        p_cfg = self.param_cfgs[param]
        assert p_cfg.optim == "adam" and p_cfg.comms == "sharded" and param.grad is None
        self._adam_update(param, grad_chunk, p_cfg)

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _adam_update_step(p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
        """Compiled Adam update step."""
        exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)
        # Cautious weight decay
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)
        p_slice.add_(other=update, alpha=-1.0)

    # -----------------------------------
    # ANVIL update

    def set_rails(self, fast_beta: float, fast_weight: float):
        """Set the fast rail's beta and its weight in the twin-rail blend for this step."""
        self.fast_beta, self.fast_weight = fast_beta, fast_weight

    def _anvil_update(self, p_cfg: ParamConfig, grad_chunk: Tensor) -> Tensor:
        """Apply the ANVIL update to this rank's slice of a bank. Returns the updated p_slice."""
        bank = self.banks[p_cfg.label]
        assert grad_chunk is bank.grad
        self.bank_update(bank)
        return bank.p_slice

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _sign_aligned_decay_update(p, mantissa, update, wd_tensor, lr_tensor, gate_src):
        """
        Sign-aligned (cautious) weight decay + parameter update: decay applies only where the
        parameter and `gate_src` (the slow rail, a denoised gradient estimate) agree in sign.
        wd_tensor is a 0-D device tensor, lr_tensor a [matrices, 1, 1] device tensor (per-matrix lr).
        Mantissa is tracked to enable higher precision updates on bfloat16 parameters.
        bfloat16 format: 1 sign bit + 8 exponent bits + 7 mantissa bits = 16 bits total
        float32 format: 1 sign bit + 8 exponent bits + 23 mantissa bits = 32 bits total
        """
        assert p.dtype == mantissa.dtype == torch.uint16
        update = update.float()
        wd_factor = wd_tensor.to(torch.float32)
        lr_factor = lr_tensor.to(torch.float32)
        p_precise_raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
        p_precise = p_precise_raw.view(torch.float32)
        aligned = (gate_src.float() * p_precise) >= 0
        p_precise.copy_(p_precise - (p_precise * aligned * wd_factor * lr_factor) - (update * lr_factor))
        p.copy_((p_precise_raw >> 16).to(torch.uint16))
        mantissa.copy_(p_precise_raw.to(torch.uint16))

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _rail_equalizer(v_chunk, lane_energy, beta2, red_dim):
        """Equalize lanes (reduced over red_dim, the longer matrix dimension): each lane's update is
        rescaled by the inverse root of an EMA of its mean squared update, then the whole matrix is
        rescaled back to its pre-equalization norm. Low-rank, Adafactor-like variance estimate from
        NorMuon (https://arxiv.org/pdf/2510.05491)."""
        lane_power = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
        lane_len = v_chunk.size(red_dim)
        pre_norm = lane_power.sum(dim=(-2, -1), keepdim=True).mul_(lane_len).sqrt_()
        lane_energy.lerp_(lane_power.to(dtype=lane_energy.dtype), 1 - beta2)
        lane_gain = lane_energy.clamp_min(1e-10).rsqrt_()
        post_power = (lane_power * lane_len) * lane_gain.float().square()
        post_norm = post_power.sum(dim=(-2, -1), keepdim=True).sqrt_()
        eq_scale = lane_gain * (pre_norm / post_norm.clamp_min_(1e-10))
        return v_chunk.mul_(eq_scale.type_as(v_chunk))


def anvil_bank_update(bank: AnvilBank):
    """One bank's ANVIL update, from its reduce-scattered gradient (the body the optimizer-tail graphs capture)."""
    # 1. Twin-rail Nesterov momentum + ANVIL whitening cascade
    v_chunk = anvil_cascade(
        bank.grad, bank.velocity, bank.momentum,
        split_baddbmm=bank.split_baddbmm,
        fast_beta_t=bank.fast_beta, fast_weight_t=bank.fast_weight,
    )
    # 2. Equalize per-lane update energy
    v_chunk = AnvilAndAdam._rail_equalizer(v_chunk, bank.lane_energy, bank.beta2, bank.red_dim)
    # 3. Update the parameter in place, with weight decay gated on the slow rail's sign. eff_lr is per
    #    matrix: MLP c_proj gets 2x, frozen matrices 0.
    AnvilAndAdam._sign_aligned_decay_update(
        bank.p_slice.view(torch.uint16), bank.mantissa, v_chunk, bank.eff_wd, bank.eff_lr, bank.velocity[1],
    )
