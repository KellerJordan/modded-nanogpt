"""CPU checks of the stratified sampled softmax (track_1_short/stratified_softmax.py and its wiring).

No GPU, no triton: the kernels are replaced by small torch references, so these tests cover the set
builder, the schedule, the upload/gather bookkeeping and the chunked loss's plumbing (which rows, which
set, which gradient goes where). They do not run the CUDA kernels or the captured step graphs.

    python -m unittest discover -s tests -v
"""
import ast
import importlib
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from track_1_short import stratified_softmax as strat  # noqa: E402  (numpy only)

V = 50304
REAL_V = 50257


def zipf_targets(rng, n):
    """Token ids with a long-tailed frequency, like text."""
    return np.minimum(rng.zipf(1.3, size=n) - 1, REAL_V - 1).astype(np.int64)


def kernel_source():
    tree = ast.parse((ROOT / "track_1_short/perf/kernels/cross_entropy.py").read_text(encoding="utf-8"))
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "CE_KERNEL_SOURCE")


class BuilderTests(unittest.TestCase):
    def build(self, chunks=3, seed=0, rank=0, prefix=True, warm=4):
        rng = np.random.default_rng(seed)
        builder = strat.StratifiedBuilder(V, 3 * strat.CHUNK_ROWS, rank)
        for _ in range(warm):
            builder.observe(zipf_targets(rng, strat.CHUNK_ROWS))
        targets = zipf_targets(rng, chunks * strat.CHUNK_ROWS)
        prefix_table = None
        if prefix:
            prefix_table = np.full(V, -1, dtype=np.int64)
            ids = rng.choice(REAL_V, size=20000, replace=False)
            prefix_table[ids] = rng.integers(0, REAL_V, size=ids.size)
        counts = builder.counts.copy()
        out = builder.build(targets, prefix_table)
        return builder, targets, prefix_table, counts, out

    def test_every_chunk_gets_a_full_ascending_set_holding_its_targets(self):
        builder, targets, _, _, (candidates, target_pos, _, col_logw) = self.build()
        self.assertEqual(candidates.shape, (3 * strat.WIDTH,))
        self.assertEqual(col_logw.dtype, np.float32)
        for k in range(3):
            cand = candidates[k * strat.WIDTH:(k + 1) * strat.WIDTH]
            rows = slice(k * strat.CHUNK_ROWS, (k + 1) * strat.CHUNK_ROWS)
            self.assertTrue(np.all(np.diff(cand) > 0))  # ascending, distinct
            self.assertTrue(0 <= cand[0] and cand[-1] < V)
            np.testing.assert_array_equal(cand[target_pos[rows]], targets[rows])  # every target, at its position

    def test_exact_part_is_targets_plus_most_frequent_and_tail_is_weighted(self):
        builder, targets, _, counts, (candidates, target_pos, _, col_logw) = self.build()
        order = np.argsort(-counts, kind="stable")
        for k in range(3):
            cols = slice(k * strat.WIDTH, (k + 1) * strat.WIDTH)
            cand, logw = candidates[cols], col_logw[cols]
            chunk = np.unique(targets[k * strat.CHUNK_ROWS:(k + 1) * strat.CHUNK_ROWS])
            exact, tail = cand[logw == 0], cand[logw != 0]
            self.assertEqual(exact.size, strat.EXACT_IDS)
            self.assertEqual(tail.size, strat.TAIL_IDS)
            self.assertTrue(np.isin(chunk, exact).all())  # no target is a weighted column
            fill = np.setdiff1d(exact, chunk)
            expected = order[~np.isin(order, chunk)][:strat.EXACT_IDS - chunk.size]
            np.testing.assert_array_equal(np.sort(fill), np.sort(expected))
            self.assertTrue(np.allclose(logw[logw != 0], math.log((V - strat.EXACT_IDS) / strat.TAIL_IDS), rtol=1e-6))
        self.assertEqual(builder.grown_chunks, 0)

    def test_prefix_targets_keep_only_exact_columns(self):
        _, targets, prefix_table, _, (candidates, _, prefix_pos, col_logw) = self.build()
        for k in range(3):
            cols = slice(k * strat.WIDTH, (k + 1) * strat.WIDTH)
            rows = slice(k * strat.CHUNK_ROWS, (k + 1) * strat.CHUNK_ROWS)
            cand, logw, pos = candidates[cols], col_logw[cols], prefix_pos[rows]
            wanted = prefix_table[targets[rows]]
            exact = set(cand[logw == 0].tolist())
            keep = np.array([w in exact for w in wanted.tolist()])
            self.assertTrue((pos[~keep] == -1).all())  # no prefix, outside the set, or only a tail column
            np.testing.assert_array_equal(cand[pos[keep]], wanted[keep])
            self.assertGreater(keep.sum(), 0)

    def test_no_prefix_table_means_no_prefix_positions(self):
        *_, (_, _, prefix_pos, _) = self.build(prefix=False)
        self.assertIsNone(prefix_pos)

    def test_reset_repeats_the_draws_and_ranks_draw_differently(self):
        rng = np.random.default_rng(5)
        targets = zipf_targets(rng, strat.CHUNK_ROWS)
        a = strat.StratifiedBuilder(V, strat.CHUNK_ROWS, rank=0)
        first = a.build(targets, None)[0].copy()
        a.observe(targets)
        a.reset()
        self.assertEqual(int(a.counts.sum()), 0)
        np.testing.assert_array_equal(a.build(targets, None)[0], first)
        b = strat.StratifiedBuilder(V, strat.CHUNK_ROWS, rank=1)
        self.assertFalse(np.array_equal(b.build(targets, None)[0], first))

    def test_many_distinct_targets_grow_the_exact_part_and_shrink_the_tail(self):
        builder = strat.StratifiedBuilder(V, strat.CHUNK_ROWS, rank=0)
        targets = np.arange(strat.CHUNK_ROWS, dtype=np.int64) % 7000  # 7000 distinct > EXACT_IDS
        candidates, target_pos, _, col_logw = builder.build(targets, None)
        self.assertEqual(builder.grown_chunks, 1)
        self.assertEqual(int((col_logw == 0).sum()), 7000)
        tail = strat.WIDTH - 7000
        self.assertTrue(np.allclose(col_logw[col_logw != 0], math.log((V - 7000) / tail), rtol=1e-6))
        np.testing.assert_array_equal(candidates[target_pos], targets)
        with self.assertRaises(AssertionError):  # a chunk that leaves no tail fails loudly
            builder.build(np.arange(strat.CHUNK_ROWS, dtype=np.int64) % 8100, None)

    def test_weighted_normalizer_is_unbiased(self):
        rng = np.random.default_rng(7)
        builder = strat.StratifiedBuilder(V, strat.CHUNK_ROWS, rank=0)
        builder.observe(zipf_targets(rng, 8 * strat.CHUNK_ROWS))
        targets = zipf_targets(rng, strat.CHUNK_ROWS)
        mass = np.exp(rng.normal(0.0, 1.5, size=V))  # exp(z) of one row over the whole vocabulary
        estimates = []
        for _ in range(300):
            candidates, _, _, col_logw = builder.build(targets, None)
            estimates.append(float((mass[candidates] * np.exp(col_logw.astype(np.float64))).sum()))
        self.assertAlmostEqual(np.mean(estimates) / mass.sum(), 1.0, delta=0.01)


class ScheduleAndKernelTests(unittest.TestCase):
    def test_only_the_widest_record_steps_are_replaced(self):
        record = [10240] * 5 + [14336] * 4 + [24576] * 3 + [0] * 2
        self.assertEqual(strat.stratified_counts(record), [10240] * 5 + [14336] * 4 + [strat.WIDTH] * 3 + [0] * 2)
        self.assertEqual(strat.WIDTH % (256 * 8), 0)  # the CE kernel's vector width

    def test_weighted_kernel_is_the_record_kernel_plus_column_weights(self):
        source = kernel_source()
        weighted = strat.weighted_kernel_source(source)
        self.assertEqual(weighted.count("col_logw"), 4)  # the parameter and three exp() sites
        self.assertEqual(weighted.replace("    const float* __restrict__ col_logw,\n", "")
                         .replace(" + col_logw[idx + k]", "").replace(" + col_logw[idx + j + t]", "")
                         .replace(" + col_logw[target]", ""), source)
        with self.assertRaises(AssertionError):
            strat.weighted_kernel_source(source.replace("thread_sum += __expf(tmp - A);", ""))


def import_trainer_modules():
    """The loss and candidate modules with triton and the CUDA kernel compile stubbed out."""
    if "triton" not in sys.modules:
        triton = types.ModuleType("triton")
        triton.jit = lambda fn: fn
        triton.cdiv = lambda a, b: -(-a // b)
        triton.language = types.ModuleType("triton.language")
        triton.language.constexpr = int
        triton.__version__ = "stub"
        sys.modules["triton"], sys.modules["triton.language"] = triton, triton.language

    class FakeKernel:
        def set_shared_memory_config(self, _):
            pass

    torch.cuda._compile_kernel = lambda *args, **kwargs: FakeKernel()
    softmax = importlib.import_module("track_1_short.sampled_softmax")
    loss = importlib.import_module("track_1_short.perf.kernels.sampled_cross_entropy")
    return softmax, loss


class FakeStaging:
    """CandidateStaging without pinned memory, streams or threads: every copy is immediate."""
    def __init__(self, max_candidates, max_rows, device):
        self.max_candidates, self.pending = max_candidates, None

    def reset(self):
        pass

    def fill_slot(self, candidates, target_pos, prefix_pos, col_logw=None):
        assert candidates.shape[0] <= self.max_candidates
        return tuple(None if a is None else torch.from_numpy(a.copy()) for a in (candidates, target_pos, prefix_pos, col_logw))

    def submit(self, step, build, *args):
        self.pending = (step, build(*args))

    def take(self, step):
        pending, self.pending = self.pending, None
        return None if pending is None else pending[1]

    def upload(self, staged, candidates, target_pos, prefix_pos, col_logw=None):
        for dst, src in zip((candidates, target_pos, prefix_pos, col_logw), staged):
            if src is not None:
                dst[:src.numel()].copy_(src)

    def wait_uploaded(self, staged):
        pass

    def mark_readers_done(self):
        pass


class WiringTests(unittest.TestCase):
    """SampledSoftmax on the CPU with a fake staging: counts, uploads and row gathers per chunk."""

    @classmethod
    def setUpClass(cls):
        cls.softmax, cls.loss = import_trainer_modules()
        from track_1_short.config import LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT
        from track_1_short.schedule import TrainingSchedule
        cls.schedule = TrainingSchedule(TRAINING_STAGES, 1122, 20, device="cpu", cooldown_frac=LR_COOLDOWN_FRAC,
                                        split_embed_stage=SPLIT_EMBED_STAGE, ws_post_yarn_ext=WS_POST_YARN_EXT)
        cls.softmax.CandidateStaging = FakeStaging
        cls.softmax.transpose_copy = lambda src, dst: dst.copy_(src.T)

    def make(self, dim=16):
        return self.softmax.SampledSoftmax(self.schedule, V, model_dim=dim, max_rows=49152, rank=3, world_size=8,
                                           device=torch.device("cpu"))

    def test_steps_965_to_1106_are_stratified_in_three_chunks(self):
        sampled = self.make()
        self.assertEqual(sorted(sampled.stratified_steps), list(range(965, 1107)))
        self.assertEqual(sampled.chunks, 3)
        self.assertEqual({sampled.counts[s] for s in range(965, 1107)}, {strat.WIDTH})
        self.assertEqual([sampled.counts[s] for s in (0, 680, 681, 964, 1107, 1193)], [10240, 10240, 14336, 14336, 0, 0])
        self.assertEqual(sampled.count_change_steps(), [681, 965, 1107])
        self.assertIsInstance(sampled.loss_inputs[strat.WIDTH], self.softmax.ChunkedSampledLoss)
        self.assertIsInstance(sampled.loss_inputs[14336], self.softmax.SampledLoss)
        self.assertEqual(sampled.loss_inputs[strat.WIDTH].width, strat.WIDTH)
        self.assertEqual(sampled.loss_inputs[14336].width, 14336)
        names = [name for name, _ in sampled.loss_inputs[strat.WIDTH].named_tensors()]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("steps 965-1106", sampled.describe())

    def test_a_stratified_step_uploads_and_gathers_one_set_per_chunk(self):
        dim = 16
        sampled = self.make(dim)
        rng = np.random.default_rng(11)
        prefix_table = np.full(V, -1, dtype=np.int64)
        prefix_table[:REAL_V] = rng.integers(-1, REAL_V, size=REAL_V)
        sampled.set_prefix_table(prefix_table)
        e4m3 = torch.float8_e4m3fn
        cache = torch.empty_strided((dim, V), (1, dim), dtype=e4m3)
        cache.view(torch.uint8).copy_(torch.from_numpy(rng.integers(0, 120, size=(dim, V)).astype(np.uint8)))
        head_rows = cache.T.view(torch.uint8)  # (V, dim)

        # A record step first (prefetched), then a stratified one: the ranking has counted the first.
        early = torch.from_numpy(zipf_targets(rng, 16384))
        sampled.prefetch(0, early)
        sampled.upload(0, early)
        record_inputs = sampled.gather(0, cache)
        self.assertIsInstance(record_inputs, self.softmax.SampledLoss)
        self.assertEqual(int(sampled.stratified.counts.sum()), 16384)
        np.testing.assert_array_equal(sampled.candidates[:10240][record_inputs.target_pos[:16384]].numpy(), early.numpy())

        targets = torch.from_numpy(zipf_targets(rng, 49152))
        sampled.upload(965, targets)
        inputs = sampled.gather(965, cache)
        self.assertIs(inputs, sampled.loss_inputs[strat.WIDTH])
        self.assertEqual(inputs.chunks, 3)
        self.assertEqual(int(sampled.stratified.counts.sum()), 16384 + 49152)
        for k in range(3):
            cand = sampled.candidates[k * strat.WIDTH:(k + 1) * strat.WIDTH]
            rows, rows_t, vocab_pos = inputs.slabs[3 * k:3 * k + 3]
            chunk = slice(k * strat.CHUNK_ROWS, (k + 1) * strat.CHUNK_ROWS)
            self.assertTrue(torch.equal(cand[inputs.target_pos[chunk]], targets[chunk]))
            self.assertTrue(torch.equal(rows.view(torch.uint8), head_rows[cand]))
            self.assertTrue(torch.equal(rows_t.view(torch.uint8), head_rows[cand].T))
            self.assertTrue(torch.equal(vocab_pos[cand].long(), torch.arange(strat.WIDTH)))
            self.assertEqual(int((vocab_pos >= 0).sum()), strat.WIDTH)
            logw = inputs.col_logw[k * strat.WIDTH:(k + 1) * strat.WIDTH]
            self.assertEqual(int((logw > 0).sum()), strat.TAIL_IDS)
            prefix = inputs.prefix_pos[chunk]
            kept = prefix >= 0
            self.assertTrue(torch.equal(cand[prefix[kept]], torch.from_numpy(prefix_table)[targets[chunk]][kept]))
            self.assertTrue((logw[prefix[kept]] == 0).all())
        # Different chunks, different sets.
        self.assertFalse(torch.equal(sampled.candidates[:strat.WIDTH], sampled.candidates[strat.WIDTH:2 * strat.WIDTH]))

    def test_reset_clears_the_ranking(self):
        sampled = self.make()
        targets = torch.from_numpy(zipf_targets(np.random.default_rng(2), 16384))
        sampled.upload(0, targets)
        sampled.gather(0, torch.zeros((16, V), dtype=torch.float8_e4m3fn).T.contiguous().T)
        sampled.reset()
        self.assertEqual(int(sampled.stratified.counts.sum()), 0)


class ChunkedLossPlumbingTests(unittest.TestCase):
    """StratifiedSoftcappedCrossEntropy with the GEMMs, CE kernel and densify replaced by exact fp64
    references: its loss and both gradients must equal autograd on the weighted softcapped CE written
    out directly. Catches a wrong row slice, a wrong set, a wrong saved tensor or a misplaced gradient."""

    @classmethod
    def setUpClass(cls):
        cls.softmax, cls.loss = import_trainer_modules()

    def test_loss_and_gradients_match_a_direct_reference(self):
        loss_module = self.loss
        torch.manual_seed(0)
        chunks, R, P, D, vocab = 3, 8, 16, 5, 64
        x_s, w_s, grad_s = 0.25, 0.125, 0.5
        A, B, C = loss_module.SOFTCAP_A, loss_module.SOFTCAP_B, loss_module.SOFTCAP_C
        exact_grads = {}

        def logit_gemm(x_f8, w_col, xs, ws):
            return (x_f8.double() * xs) @ (w_col.double() * ws)

        def ce(logits, target_pos, mtp_weights, prefix_pos, prefix_weight, col_logw, losses, grad_input,
               n_rows, n_predict, a, b, c, gs):
            with torch.enable_grad():  # a Function's forward runs with grad off
                exact(logits, target_pos, mtp_weights, prefix_pos, prefix_weight, col_logw, losses, grad_input,
                      n_rows, n_predict, a, b, c)

        def exact(logits, target_pos, mtp_weights, prefix_pos, prefix_weight, col_logw, losses, grad_input,
                  n_rows, n_predict, a, b, c):
            logits = logits.detach().clone().requires_grad_(True)
            z = a * torch.sigmoid((logits + b) / c)
            lse = torch.logsumexp(z + col_logw.double(), dim=1)
            total = torch.zeros(n_rows, dtype=torch.float64)
            for i in range(n_rows):
                for k in range(n_predict):
                    if i + k < n_rows:
                        total[i] = total[i] + mtp_weights[k] * (lse[i] - z[i, target_pos[i + k]])
                if prefix_pos[i] >= 0:
                    total[i] = total[i] + prefix_weight[0] * (lse[i] - z[i, prefix_pos[i]])
            losses.copy_(total.detach())
            exact_grads[id(grad_input)] = torch.autograd.grad(total.sum(), logits)[0]

        def backward_gemms(grad_input, rows_t, x_f8, xs, ws, gs):
            grad = exact_grads[id(grad_input)]
            return grad @ (rows_t.double() * ws).T, (x_f8.double() * xs).T @ grad

        def densify(src, pos, dst):
            dst.copy_(torch.where(pos >= 0, src[:, pos.clamp(min=0).long()], torch.zeros((), dtype=src.dtype)))

        originals = {name: getattr(loss_module, name) for name in
                     ("_ce_logit_gemm", "stratified_ce_fwd_bwd", "_ce_backward_gemms", "sampled_densify")}
        loss_module._ce_logit_gemm, loss_module.stratified_ce_fwd_bwd = logit_gemm, ce
        loss_module._ce_backward_gemms, loss_module.sampled_densify = backward_gemms, densify
        try:
            e4m3 = torch.float8_e4m3fn
            head = torch.randn(vocab, D).div(w_s).to(e4m3)  # (V, D) fp8 lm_head rows
            x = torch.randn(chunks * R, D, dtype=torch.float32, requires_grad=True)
            lm_head_weight = torch.zeros(D, vocab, dtype=torch.float64, requires_grad=True)
            mtp_weights = torch.tensor([1.0, 0.5], dtype=torch.float32)
            prefix_weight = torch.tensor([0.25])
            slabs, sets = [], []
            target_pos = torch.randint(0, P, (chunks * R,))
            prefix_pos = torch.randint(-1, P, (chunks * R,))
            col_logw = torch.zeros(chunks * P)
            for k in range(chunks):
                cand = torch.randperm(vocab)[:P].sort().values
                sets.append(cand)
                vocab_pos = torch.full((vocab,), -1, dtype=torch.int32)
                vocab_pos[cand] = torch.arange(P, dtype=torch.int32)
                rows = head.view(torch.uint8)[cand].view(e4m3)  # fp8 has no CPU index kernel: move the bytes
                slabs += [rows, rows.view(torch.uint8).T.contiguous().view(e4m3), vocab_pos]
                col_logw[k * P + P // 2:(k + 1) * P] = 1.0 + k  # each chunk its own tail weight

            losses = loss_module.StratifiedSoftcappedCrossEntropy.apply(
                x, mtp_weights, target_pos, prefix_pos, prefix_weight, lm_head_weight, col_logw,
                x_s, w_s, grad_s, *slabs)
            grad_x, grad_w = torch.autograd.grad(losses.sum(), (x, lm_head_weight))

            # The same loss, written directly on the quantized operands the function reads.
            xq = (x.detach().div(x_s).to(e4m3).double() * x_s).requires_grad_(True)
            wq = (head.double() * w_s).requires_grad_(True)  # (V, D)
            reference = []
            for k in range(chunks):
                rows = slice(k * R, (k + 1) * R)
                z = A * torch.sigmoid((xq[rows] @ wq[sets[k]].T + B) / C)
                lse = torch.logsumexp(z + col_logw[k * P:(k + 1) * P].double(), dim=1)
                tp, pp = target_pos[rows], prefix_pos[rows]
                for i in range(R):
                    term = sum(mtp_weights[j].double() * (lse[i] - z[i, tp[i + j]]) for j in range(2) if i + j < R)
                    if pp[i] >= 0:
                        term = term + prefix_weight[0].double() * (lse[i] - z[i, pp[i]])
                    reference.append(term)
            reference = torch.stack(reference)
            ref_grad_x, ref_grad_w = torch.autograd.grad(reference.sum(), (xq, wq))

            self.assertEqual(losses.shape, (chunks * R,))
            self.assertTrue(torch.allclose(losses.double(), reference.detach(), rtol=1e-5, atol=1e-5))
            self.assertTrue(torch.allclose(grad_x.double(), ref_grad_x, rtol=1e-6, atol=1e-8))
            self.assertEqual(grad_w.shape, (D, vocab))
            self.assertTrue(torch.allclose(grad_w.double(), ref_grad_w.T, rtol=1e-6, atol=1e-8))
        finally:
            for name, fn in originals.items():
                setattr(loss_module, name, fn)


if __name__ == "__main__":
    unittest.main()
