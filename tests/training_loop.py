"""Paired CPU language-model training with real Gloo gradient reductions.

Uses production BOS planning, n-gram hashing, candidates and GPT-2 prefix mapping. The reduced
decoder and CPU optimizer equations are a training regression fixture, not the FP8/FA3 record.
Run: python -m tests.training_loop --output work/parity [--shard path/to/FineWeb.bin]
"""
import argparse
import copy
from datetime import timedelta
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

from tests.reference_sampled_candidates import BufferedCandidateBuilder
from tests.reference_training_prep import LegacyShard, legacy_ngram_row_ids
from track_1_short.data import BOS_ID, Shard
from track_1_short.ngram_hash import ngram_row_ids, prime_ngram_hash
from track_1_short.perf.sampled_candidates import CandidateBuilder
from track_1_short.schedule import get_rail_beta
from track_1_short.optim.spectral_maps import ANVIL_MAPS as MAPS


def geometry(step, steps):
    return ((64, 32, 256), (96, 64, 512), (128, 96, 1024), (64, 64, 0))[min(3, step * 4 // steps)]


def ready_shard(cls, tokens, world):
    shard = cls(tokens, world)
    shard._loader_thread.join()
    shard._maybe_switch()
    return shard


def batch_from(shard, rank, budget, cap, hash_fn):
    starts, ends = shard.next_batch(budget, cap)
    starts, ends = torch.tensor(starts[rank]), torch.tensor(ends[rank])
    packed = torch.cat([shard.tokens[a:b] for a, b in zip(starts, ends)]).to(torch.int64)
    ends[-1] -= 1
    return packed[:-1].to(torch.int32), packed[1:], (ends-starts).cumsum(0), hash_fn(packed[:-1])


class Decoder(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.embed = nn.Embedding(50304, 16)
        self.ngram = nn.Embedding(rows, 8)
        nn.init.zeros_(self.ngram.weight)
        self.qkv = nn.Linear(16, 48, bias=False)
        self.proj = nn.Linear(16, 16, bias=False)
        self.up = nn.Linear(16, 64, bias=False)
        self.down = nn.Parameter(torch.randn(64, 16) * .02)
        self.head = nn.Linear(16, 50304, bias=False)
        self.gate = nn.Parameter(torch.tensor(.1))
        self.register_buffer("sign_pool", torch.randn(8192, 8).sign())

    def forward(self, inputs, ends, slots, candidates):
        n = len(inputs)
        x = self.embed(inputs.long())
        a = ((48271*inputs) ^ (30011*inputs.roll(1))) & 8191
        b = ((58699*inputs) ^ (39779*inputs.roll(1)) ^ (26801*inputs.roll(2))) & 8191
        a[0], b[:2] = 0, 0
        ng = self.ngram(slots[:n]) * self.sign_pool[a.long()]
        ng = ng + self.ngram(slots[n:]) * self.sign_pool[b.long()]
        x = x + F.pad(self.gate*ng, (0, 8))
        q, k, v = self.qkv(F.rms_norm(x, (16,))).view(n, 3, 2, 8).unbind(1)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        q, k = F.rms_norm(q, (8,)), F.rms_norm(k, (8,))
        docs = torch.bucketize(torch.arange(n), ends[:-1], right=True)
        mask = (docs[:, None] == docs[None, :]) & (torch.arange(n)[:, None] >= torch.arange(n)[None, :])
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(attn.transpose(0, 1).reshape(n, 16))
        x = x + F.relu(self.up(F.rms_norm(x, (16,)))).square() @ self.down
        w = self.head.weight if candidates is None else self.head.weight[candidates]
        logits = F.linear(F.rms_norm(x, (16,)), w)
        return 23 * torch.sigmoid((logits + 5) / 7.5)


def reference_cascade(g):
    x = g.bfloat16().float()
    wide = x.shape[0] < x.shape[1]
    if wide:
        x = x.T.contiguous()
    A = (x.T @ x).bfloat16().float()
    d = A.diagonal().sum().sqrt() * 1.05 + 1e-6
    x, A = (x/d).bfloat16().float(), (A/d.square()).bfloat16().float()
    for i, (a, b, c) in enumerate(MAPS):
        if i:
            A = (x.T @ x).bfloat16().float()
        B = (b*A + c*(A@A)).bfloat16().float()
        x = (a*x + x@B).bfloat16().float()
    return x.T.contiguous() if wide else x


class ReferenceOptimizers:
    """ANVIL twin rails/equalizer and row-RMS Adam, with shortened transition points for the fixture.

    FP32 master parameters and moments are compared completely, not through checksums.
    """
    def __init__(self, model):
        self.model = model
        self.projections = {n: p for n, p in model.named_parameters() if n in
                            ("qkv.weight", "proj.weight", "up.weight", "down")}
        excluded = {id(p) for p in self.projections.values()} | {id(model.ngram.weight)}
        self.adam = torch.optim.AdamW([p for p in model.parameters() if id(p) not in excluded], lr=.008)
        self.rails = {n: {"fast": torch.zeros_like(p), "slow": torch.zeros_like(p),
                          "energy": torch.zeros((p.shape[0], 1) if p.shape[0] >= p.shape[1] else (1, p.shape[1]))}
                      for n, p in self.projections.items()}
        self.row_energy = torch.zeros(model.ngram.num_embeddings)
        self.row_event = 0
        self.adam_updates = 0

    @staticmethod
    def row_update(step, steps):
        return step % (2 if step < steps//2 else 4) == (1 if step < steps//2 else 3)

    @torch.no_grad()
    def step(self, step, steps):
        cooldown = 1 - .7 * step / max(1, steps-1)
        lr = .023 * cooldown
        beta = get_rail_beta(step, steps, beta_warmup_steps=4, beta_cooldown_steps=4)
        for name, p in self.projections.items():
            state = self.rails[name]
            state["fast"].lerp_(p.grad, 1-(beta if step < steps//2 else .85))
            state["slow"].lerp_(p.grad, .02)
            w = 1. if step < steps//2 else .4385
            v = reference_cascade(p.grad.lerp(w*state["fast"]+(1-w)*state["slow"], beta))
            red = -1 if p.shape[0] >= p.shape[1] else -2
            power = v.square().mean(red, keepdim=True)
            before = (power.sum()*v.shape[red]).sqrt()
            state["energy"].lerp_(power, .1)
            gain = state["energy"].clamp_min(1e-10).rsqrt()
            after = ((power*v.shape[red])*gain.square()).sum().sqrt()
            v = (v*(gain*before/after.clamp_min(1e-10)).bfloat16().float()).bfloat16().float()
            eff_lr = lr * max(1., p.shape[0]/p.shape[1])**.5 * (2 if name == "down" else 1)
            p.sub_(p*((state["slow"]*p)>=0)*(2.25*lr)*eff_lr + v*eff_lr)
            p.grad = None
        if step % 2:
            self.adam_updates += 1
            for group in self.adam.param_groups:
                group["lr"] = .008*cooldown
            self.adam.step()
            self.adam.zero_grad(set_to_none=True)
        if self.row_update(step, steps):
            self.row_event += 1
            beta2 = .95 if step < steps//2 else .95**2
            self.row_energy.mul_(beta2)
            p, g = self.model.ngram.weight, self.model.ngram.weight.grad
            rows = g.abs().sum(1) > 0
            grad = g[rows]
            self.row_energy[rows] += (1-beta2)*grad.square().mean(1)
            update = grad/(self.row_energy[rows, None].sqrt()+1e-10)
            update *= .008*cooldown*70*(1-beta2**self.row_event)**.5
            update += torch.where(update*p[rows]>0, p[rows]*.0001, 0)
            p[rows] -= update
            p.grad = None


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        if not torch.isfinite(a).all():
            raise AssertionError("non-finite training state")
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def worker(rank, world, steps, seed, shard_path, output, fault):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{Path(output).resolve() / 'gloo'}",
                            rank=rank, world_size=world, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(seed)
        if shard_path:
            raw = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=1024)
            tokens = torch.from_numpy(np.array(raw[:1_000_000], dtype=np.int32))
        from track_1_short.model.prefix_prediction import build_prefix_table_bucket
        prefix = build_prefix_table_bucket(50304, 0, 1).numpy()
        if not shard_path:
            children = list(range(1000, 1016))
            parents = [int(prefix[i]) for i in children if prefix[i] >= 0]
            # Include real GPT-2 prefix parents as targets, so the auxiliary loss is exercised.
            tokens = torch.tensor(([BOS_ID] + children + parents) * 1600, dtype=torch.int32)
        preview = ready_shard(LegacyShard, tokens, world)
        row_sets = []
        for step in range(steps+1):
            budget, cap, _ = geometry(min(step, steps-1), steps)
            for_rank = preview.next_batch(budget, cap)
            for r in range(world):
                packed = torch.cat([tokens[a:b] for a, b in zip(for_rank[0][r], for_rank[1][r])])
                row_sets.append(legacy_ngram_row_ids(packed[:-1]).numpy())
        rows = np.unique(np.concatenate(row_sets))
        models = [Decoder(len(rows))]
        for p in models[0].parameters():
            dist.broadcast(p.detach(), 0)
        models.append(copy.deepcopy(models[0]))
        initial = {n: p.detach().clone() for n, p in models[0].named_parameters()}
        optimizers = [ReferenceOptimizers(m) for m in models]
        planners = [ready_shard(cls, tokens, world) for cls in (LegacyShard, Shard)]
        builders = [cls(50304, 1024, 128) for cls in (BufferedCandidateBuilder, CandidateBuilder)]
        for b in builders:
            b.reset(rank, world)
        prime_ngram_hash([64, 96, 128])
        trace = []
        tail = [{name: None for name in optimizer.projections} for optimizer in optimizers]
        for step in range(steps):
            budget, cap, P = geometry(step, steps)
            batches = [batch_from(s, rank, budget, cap, h) for s, h in
                       zip(planners, (legacy_ngram_row_ids, ngram_row_ids))]
            for a, b in zip(*batches):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            losses = []
            active_prefixes = 0
            for branch, (model, builder, batch) in enumerate(zip(models, builders, batches)):
                inputs, targets, ends, ids = batch
                slots = torch.from_numpy(np.searchsorted(rows, ids.numpy())).long()
                if P:
                    candidates, target_pos, prefix_pos = builder.build(P, targets.numpy(), prefix)
                    candidates = torch.from_numpy(candidates.copy())
                    labels, prefixes = torch.from_numpy(target_pos.copy()), torch.from_numpy(prefix_pos.copy())
                    if fault and branch == 1 and step == 0:
                        labels[0] = (labels[0]+1) % P
                else:
                    candidates, labels = None, targets
                    prefixes = torch.from_numpy(prefix[targets.numpy()].copy())
                logits = model(inputs, ends, slots, candidates)
                loss = F.cross_entropy(logits, labels)
                if step < steps//2:
                    if branch == 0:
                        active_prefixes += int((prefixes >= 0).sum())
                    loss = loss + (.5*(1-step/(steps//2)))*F.cross_entropy(logits[:-1], labels[1:])
                    loss = loss + .2*F.cross_entropy(logits, prefixes, ignore_index=-1, reduction="sum")/budget
                loss.backward()
                losses.append(float(loss.detach()))
            assert losses[0] == losses[1] and math.isfinite(losses[0]), "forward-loss mismatch"
            for model, optimizer in zip(models, optimizers):
                for name, p in model.named_parameters():
                    reduce = name in optimizer.projections or (optimizer.row_update(step, steps)
                              if name == "ngram.weight" else step % 2 == 1)
                    if reduce and p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad.div_(world)
            gradient_norm = 0.
            for (_, a), (_, b) in zip(models[0].named_parameters(), models[1].named_parameters()):
                assert_tree_equal(a.grad, b.grad)
                if a.grad is not None:
                    gradient_norm += float(a.grad.square().sum())
            for optimizer in optimizers:
                optimizer.step(step, steps)
            assert_tree_equal(models[0].state_dict(), models[1].state_dict())
            for attribute in ("rails", "row_energy", "row_event"):
                assert_tree_equal(getattr(optimizers[0], attribute), getattr(optimizers[1], attribute))
            assert_tree_equal(optimizers[0].adam.state_dict(), optimizers[1].adam.state_dict())
            if step >= steps-8 and step % 4 == 3:
                for averages, optimizer in zip(tail, optimizers):
                    for name, p in optimizer.projections.items():
                        if averages[name] is None:
                            averages[name] = p.detach().clone()
                        else:
                            averages[name].lerp_(p.detach(), 4./53.)
            if step == steps//2-1:
                # Real serialization after an update boundary; destroy then restore candidate state.
                model, optimizer = models[1], optimizers[1]
                buffer = io.BytesIO()
                torch.save(dict(model=model.state_dict(), adam=optimizer.adam.state_dict(), rails=optimizer.rails,
                                row_energy=optimizer.row_energy, row_event=optimizer.row_event,
                                planner=planners[1].i, sweep=builders[1].sweep_offset), buffer)
                with torch.no_grad():
                    for p in model.parameters():
                        p.zero_()
                optimizer.adam.state.clear()
                optimizer.rails.clear()
                optimizer.row_energy.zero_()
                planners[1].i = builders[1].sweep_offset = 0
                buffer.seek(0)
                restored = torch.load(buffer, weights_only=True)
                model.load_state_dict(restored["model"])
                optimizer.adam.load_state_dict(restored["adam"])
                optimizer.rails = restored["rails"]
                optimizer.row_energy.copy_(restored["row_energy"])
                optimizer.row_event = restored["row_event"]
                planners[1].i, builders[1].sweep_offset = restored["planner"], restored["sweep"]
                assert_tree_equal(models[0].state_dict(), model.state_dict())
            if step % 8 == 7:
                for p in models[0].parameters():
                    replica = p.detach().clone()
                    dist.broadcast(replica, 0)
                    torch.testing.assert_close(p, replica, rtol=0, atol=0)
            trace.append(dict(step=step, tokens=budget, candidates=P, loss=losses[0],
                              gradient_norm=gradient_norm**.5, prefix_targets=active_prefixes,
                              adam_update=bool(step % 2), row_update=optimizers[0].row_update(step, steps),
                              slow_rail_engaged=step >= steps//2, max_loss_error=0., max_state_error=0.))
        changed = sum(not torch.equal(p, initial[n]) for n, p in models[0].named_parameters())
        assert changed > 0 and any(float(s["fast"].abs().sum()) > 0 for s in optimizers[0].rails.values())
        assert sum(t["prefix_targets"] for t in trace) > 0
        for optimizer in optimizers:
            assert optimizer.adam_updates == steps//2 and optimizer.row_event == 3*steps//8
            assert (optimizer.row_energy > 0).any()
        # One end-of-run average/decontraction cycle, with both complete parameter paths compared.
        with torch.no_grad():
            for averages, optimizer in zip(tail, optimizers):
                for name, p in optimizer.projections.items():
                    before = p.norm()
                    p.copy_(averages[name])
                    p.mul_(before / p.norm().clamp_min(1e-10))
        assert_tree_equal(models[0].state_dict(), models[1].state_dict())
        # A held-out full-softmax forward after the last real optimizer update.
        validation = []
        for model, planner, h in zip(models, planners, (legacy_ngram_row_ids, ngram_row_ids)):
            model.eval()
            inputs, targets, ends, ids = batch_from(planner, rank, 64, 64, h)
            slots = torch.from_numpy(np.searchsorted(rows, ids.numpy())).long()
            with torch.no_grad():
                validation.append(float(F.cross_entropy(model(inputs, ends, slots, None), targets)))
        assert validation[0] == validation[1] and math.isfinite(validation[0])
        report = dict(rank=rank, world=world, seed=seed, corpus="FineWeb" if shard_path else "synthetic",
                      steps=steps, changed_parameters=changed, checkpoint_roundtrip=True, tail_averaged=True,
                      validation_loss=validation[0], trace=trace)
        Path(output, f"rank-{rank}.json").write_text(json.dumps(report, indent=2))
    finally:
        dist.destroy_process_group()


def run_training_parity(output, *, world=2, steps=24, seed=387, shard=None, fault=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mp.spawn(worker, args=(world, steps, seed, None if shard is None else str(Path(shard).resolve()), str(output), fault),
             nprocs=world, join=True)
    return [json.loads(Path(output, f"rank-{r}.json").read_text()) for r in range(world)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 8 or args.steps % 8 or args.ranks < 1:
        parser.error("steps must be a positive multiple of eight (at least eight); ranks must be positive")
    results = run_training_parity(args.output, world=args.ranks, steps=args.steps, shard=args.shard)
    print(json.dumps({"passed": True, "ranks": len(results), "steps_per_rank": args.steps,
                      "corpus": results[0]["corpus"], "validation_loss_by_rank": [r["validation_loss"] for r in results],
                      "max_loss_error": 0., "max_gradient_error": 0., "max_state_error": 0.}, indent=2))


if __name__ == "__main__":
    main()
