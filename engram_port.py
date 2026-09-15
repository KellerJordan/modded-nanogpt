"""Engram for modded-nanogpt: hashed n-gram rows from a host-resident table, gated into one
attention head's values at the layers in ENGRAM_SITES. Each token's trailing 2- and 3-grams (two
hashes each) index a table of ENGRAM_ROWS rows; a row holds 32 bf16 values per hash slot per
site, 128 per site (the width of an attention head). The per-site sigmoid gate is a model
parameter; the rows get a row-wise RMSprop update here.
"""
import os, math
import torch, torch.distributed as dist, torch.nn.functional as F

from engram_host import EngramHost
from engram_triton import row_gather

GATE_CHANNELS = 12          # residual channels the gate reads
ORDERS, WEIGHTS = (2, 3), (2, 2)   # n-gram orders and hash slots per order
SITE_WIDTH = 128            # values per site in a row (= head width)
ROW_LR, ROW_DECAY, ACCUM_BETA = 0.2, 0.001, 0.99
PF_DEPTH = 2                # steps of prefetch


def sites():
    return [int(s) for s in os.environ.get("ENGRAM_SITES", "1,2,4,5,7,8,9,10").split(",") if s]


def _largest_prime_at_most(n):
    def is_prime(k):
        if k < 2 or k % 2 == 0:
            return k == 2
        return all(k % d for d in range(3, int(math.isqrt(k)) + 1, 2))
    while not is_prime(n):
        n -= 1
    return n


class EngramPort:
    def __init__(self, rows, rank, world_size, max_tokens):
        self.sites = sites()
        self.per_order = SITE_WIDTH // sum(WEIGHTS)
        self.rows = _largest_prime_at_most(rows)
        self.rank, self.world_size = rank, world_size
        self.path = f"/dev/shm/modded_engram_{self.rows}.bin" if world_size > 1 else None
        self.host = EngramHost(self.rows, len(self.sites), self.per_order, ROW_DECAY, ACCUM_BETA, ORDERS, WEIGHTS,
                               self.path, rank, pf_depth=PF_DEPTH, ring=PF_DEPTH + 2, max_tokens=max_tokens)
        self._leaf = None
        if self.path is not None:
            dist.barrier()

    def describe(self):
        return (f"Engram: {self.rows:,} rows x {len(self.sites)} sites x {SITE_WIDTH} ({self.host.weight.numel() * 2 / 1e9:.1f} GB host"
                f"{', shared mmap' if self.path else ''}) | sites {self.sites}")

    def lookup(self, inputs, training):
        """(T, n_sites, SITE_WIDTH) bf16 rows for the token ids, grad-tracked into the host table."""
        T = inputs.numel()
        if training and self.host._pending:
            uniq, uniq_cpu, inv, rows, slot = self.host.take()
        else:
            uniq, uniq_cpu, inv, rows, slot = self.host.gather_sync(inputs)
        leaf = rows.requires_grad_(True)
        if training:
            assert self._leaf is None, "engram: two training lookups without a step()"
            self._leaf = (uniq, uniq_cpu, leaf, slot)
        return row_gather(leaf, inv.contiguous(), T, len(self.sites), self.host.G, self.per_order)

    def attach(self, cfg, inputs, training):
        cfg.engram_out = self.lookup(inputs, training)
        return cfg

    def prefetch(self, batch):
        self.host.prefetch(batch[5])   # the loader's CPU copy of the ids

    def step(self):
        if self._leaf is not None:
            uniq, uniq_cpu, leaf, slot = self._leaf
            self._leaf = None
            self.host.step(uniq, uniq_cpu, leaf, ROW_LR, slot)

    @torch.no_grad()
    def reset(self):
        self._leaf = None
        self.host.reset()
        if self.world_size > 1:
            dist.barrier()

    def shutdown(self):
        self.host.shutdown(unlink_shared=(self.rank == 0 and self.path is not None))


def gated_values(x_normed, engram_rows, gate_bank, gate_bias, site_idx):
    """(1, T, SITE_WIDTH): this site's slice of the rows, scaled by its gate."""
    e = engram_rows[:, site_idx]
    gin = x_normed[0, :, :GATE_CHANNELS]
    z = F.linear(gin.to(e.dtype), gate_bank[site_idx].to(e.dtype)) / GATE_CHANNELS + gate_bias[site_idx].to(e.dtype)
    return (torch.sigmoid(z) * e)[None]
