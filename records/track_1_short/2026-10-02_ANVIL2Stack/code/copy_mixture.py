"""Document-local copy distribution, adapted from modded-nanogpt PR376.

Calibration is a caller-owned, timed operation on consumed training examples.
This module never opens a data source or inspects an environment variable.
"""
from collections import deque
import math

import torch
import torch.distributed as dist


BOS_ID = 50256
COPY_SPLITS = ((1, 0, 0), (2, 1, 1), (3, 2, 1), (4, 2, 2),
               (6, 4, 2), (8, 4, 4), (12, 8, 4), (16, 8, 8),
               (24, 16, 8), (32, 16, 16))
COPY_DISTS = (768, 2560)
COPY_NUM_BUCKETS = 31
COPY_FIT_BATCHES = 8


def _reject(bad, message, *, collective=False):
    flag = bad.to(torch.int64)
    if collective and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if bool(flag):
        raise ValueError(message)


def _rank_and_prev(key):
    sorted_key, order = torch.sort(key, stable=True)
    first = torch.ones_like(sorted_key, dtype=torch.bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    rank, prev = torch.empty_like(order), torch.empty_like(order)
    rank[order] = torch.cumsum(first, 0) - 1
    prev[order] = torch.where(first, -1, order.roll(1))
    return rank, prev


@torch.no_grad()
def copy_candidates(inputs):
    """Return exact suffix-copy candidates and buckets using only input prefixes."""
    if inputs.ndim != 1 or inputs.dtype not in (torch.int32, torch.int64):
        raise ValueError('Inputs must be an int32/int64 token vector')
    _reject(((inputs < 0) | (inputs > BOS_ID)).any(), 'Input token is outside the GPT-2 vocabulary')
    count = inputs.numel()
    x = inputs.long()
    ar = torch.arange(count, device=x.device)
    is_bos = x == BOS_ID
    pos_in_doc = ar - torch.cummax(ar * is_bos, 0).values
    ranks = {}
    src = torch.full_like(ar, -1)
    len_idx = torch.zeros_like(ar)
    for k, (length, left, right) in enumerate(COPY_SPLITS):
        if length == 1:
            key = torch.cumsum(is_bos, 0) * 65536 + x
        else:
            key = ranks[left].roll(right) * count + ranks[right]
            key = torch.where(pos_in_doc >= length - 1, key, -1 - ar)
        ranks[length], prev = _rank_and_prev(key)
        found = prev >= 0
        src = torch.where(found, prev, src)
        len_idx = torch.where(found, k, len_idx)
    cand = x[(src + 1).clamp(min=0)]
    dist_idx = torch.bucketize(ar - src, torch.tensor(COPY_DISTS, device=x.device))
    bucket = torch.where(src >= 0, 1 + len_idx * 3 + dist_idx, 0)
    return cand, bucket


def copy_mix_loss(loss, hit, lam):
    """NLL of a normalized full-vocabulary mixture; target use stays in the loss."""
    if loss.shape != hit.shape or loss.shape != lam.shape or hit.dtype != torch.bool:
        raise ValueError('Loss, hit and lambda must be matching vectors')
    _reject((~torch.isfinite(lam) | (lam < 0) | (lam >= 1)).any(), 'Invalid mixture weight')
    keep = torch.log1p(-lam)
    both = -torch.logaddexp(keep - loss, torch.log(lam))
    mixed = torch.where(hit & (lam > 0), both, loss - keep)
    return torch.where(lam == 0, loss, mixed)


def copy_log_probabilities(model_log_probabilities, candidates, buckets, weights):
    """Target-free complete distribution, equivalent to copy_mix_loss."""
    if model_log_probabilities.ndim != 2 or candidates.shape != buckets.shape:
        raise ValueError('Expected [tokens,vocabulary] log probabilities and vector features')
    if candidates.shape != (model_log_probabilities.shape[0],):
        raise ValueError('Each row needs exactly one copy feature')
    if weights.shape != (COPY_NUM_BUCKETS,) or weights.dtype != torch.float32:
        raise ValueError('The checkpointed mixture has exactly 31 float32 parameters')
    _reject((~torch.isfinite(weights) | (weights < 0) | (weights >= 1)).any() | (weights[0] != 0),
            'Invalid mixture checkpoint')
    _reject(((buckets < 0) | (buckets >= COPY_NUM_BUCKETS)).any()
            | ((candidates < 0) | (candidates >= model_log_probabilities.shape[1])).any(),
            'Invalid copy feature')
    lam = weights[buckets]
    mixed = model_log_probabilities + torch.log1p(-lam)[:, None]
    indices = candidates[:, None]
    copied = torch.logaddexp(mixed.gather(1, indices), torch.log(lam)[:, None])
    return mixed.scatter(1, indices, copied)


@torch.no_grad()
def copy_fit(loss, hit, bucket, *, iters=40, lam_max=0.98, prior_miss=5.0):
    """Fit FP32 parameters with FP64 arithmetic on supplied training NLLs."""
    tensors = all(isinstance(value, torch.Tensor) for value in (loss, hit, bucket))
    valid = tensors and (loss.ndim == 1 and loss.shape == hit.shape == bucket.shape
                         and loss.is_floating_point() and hit.dtype == torch.bool
                         and bucket.dtype == torch.int64 and loss.device == hit.device == bucket.device)
    bounds = (type(iters) is int and iters >= 1 and type(lam_max) in (int, float)
              and math.isfinite(lam_max) and 0 < lam_max < 1
              and type(prior_miss) in (int, float) and math.isfinite(prior_miss) and prior_miss >= 0)
    descriptor = ((iters, float(lam_max), float(prior_miss), str(loss.dtype), loss.device.type)
                  if valid and bounds else None)
    descriptors = [descriptor]
    if dist.is_initialized():
        descriptors = [None] * dist.get_world_size()
        dist.all_gather_object(descriptors, descriptor)
    if any(value is None or value != descriptors[0] for value in descriptors):
        raise ValueError('Calibration geometry or fit configuration disagrees across ranks')
    _reject((~torch.isfinite(loss) | (loss < 0)).any()
            | ((bucket < 0) | (bucket >= COPY_NUM_BUCKETS)).any(),
            'Invalid calibration value on at least one rank', collective=True)
    p = torch.exp(-loss.double())
    n_miss = torch.bincount(bucket[~hit], minlength=COPY_NUM_BUCKETS).double()
    if dist.is_initialized():
        dist.all_reduce(n_miss)
    n_miss += prior_miss
    hit_bucket, hit_p = bucket[hit], p[hit]
    lo, hi = torch.zeros_like(n_miss), torch.full_like(n_miss, lam_max)
    for _ in range(iters):
        mid = (lo + hi) / 2
        grad = torch.zeros_like(n_miss).index_add_(
            0, hit_bucket, (1 - hit_p) / (hit_p + mid[hit_bucket] * (1 - hit_p)))
        if dist.is_initialized():
            dist.all_reduce(grad)
        up = grad > n_miss / (1 - mid)
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    lo[0] = 0
    return lo.float()


class ConsumedTrainingBatches:
    """Retain a fixed, contiguous pre-extension window after actual batch consumption."""
    def __init__(self, scheduled_steps, count=COPY_FIT_BATCHES):
        if type(scheduled_steps) is not int or type(count) is not int or not 0 < count <= scheduled_steps:
            raise ValueError('Invalid calibration capture window')
        self.start = scheduled_steps - count
        self.stop = scheduled_steps
        self.count = count
        self.next_step = 0
        self.batches = deque()
        self.sealed = False

    @torch.no_grad()
    def after_backward(self, step, batch):
        if self.sealed or step != self.next_step:
            raise RuntimeError('Consumed batches must arrive once in training order')
        if len(batch) != 4:
            raise ValueError('Preserve exactly inputs, targets, cumulative lengths and n-gram IDs')
        inputs, targets, cu, ngrams = batch
        if (inputs.ndim != 1 or inputs.dtype != torch.int32 or targets.shape != inputs.shape
                or targets.dtype != torch.int64 or cu.ndim != 1 or cu.dtype != torch.int32
                or ngrams.shape != (2 * inputs.numel(),) or ngrams.dtype != torch.int32
                or any(t.device != inputs.device for t in batch)):
            raise ValueError('Invalid original training batch geometry')
        if self.start <= step < self.stop:
            self.batches.append(tuple(t.detach().clone() for t in batch))
        self.next_step += 1

    def take(self, total_steps):
        if self.sealed or self.next_step != total_steps or len(self.batches) != self.count:
            raise RuntimeError('Calibration needs the completed run and its exact capture window')
        invalid = []
        for inputs, targets, cu, _ in self.batches:
            if cu.numel() < 2:
                raise ValueError('Missing cumulative sequence boundaries')
            invalid.append((cu[0] != 0) | (cu[-1] != inputs.numel()) | (cu[1:] < cu[:-1]).any()
                           | (cu < 0).any() | (cu > inputs.numel()).any()
                           | ((inputs < 0) | (inputs > BOS_ID)).any()
                           | ((targets < 0) | (targets > BOS_ID)).any())
        _reject(torch.stack(invalid).any(), 'Invalid retained original training batch', collective=True)
        self.sealed = True
        batches = tuple(self.batches)
        self.batches.clear()
        return batches
