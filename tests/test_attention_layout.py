import itertools
import unittest

import torch

from tests.attention_cpu import case, forward, load_attention


class AttentionLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.original = load_attention(legacy=True)
        cls.updated = load_attention()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_forward_and_every_input_gradient_match_for_record_widths_and_extras(self):
        configs = list(itertools.product((64, 128), (64, 128), (False, True), (False, True), (False, True)))
        for qk, value, paired, aux, gate in configs:
            offsets = (False, True) if qk == 128 and not paired else (False,)
            for key_offset, fp8, dtype in itertools.product(offsets, (False, True), (torch.float64, torch.bfloat16)):
                with self.subTest(qk=qk, value=value, paired=paired, aux=aux, gate=gate, key_offset=key_offset, fp8=fp8, dtype=dtype):
                    kwargs = dict(qk=qk, value=value, paired=paired, aux=aux, gate=gate, key_offset=key_offset, fp8=fp8, dtype=dtype)
                    old, new = (case(m, **kwargs) for m in (self.original, self.updated))
                    a, b = forward(old), forward(new)
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(a.shape, (1, 32, 24))
                    # A nonuniform upstream gradient catches reshapes that preserve a sum but reorder heads.
                    upstream = torch.linspace(-1, 1, a.numel(), dtype=dtype).reshape_as(a)
                    (a*upstream).sum().backward()
                    (b*upstream).sum().backward()
                    for name in old[2]:
                        torch.testing.assert_close(old[2][name].grad, new[2][name].grad, rtol=0, atol=0)

    def test_strided_inputs_and_zero_auxiliary_values_preserve_backward(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                old, new = (case(m, paired=paired, aux=True, gate=True, strided=True) for m in (self.original, self.updated))
                for bundle in (old, new):
                    self.assertFalse(bundle[2]["x"].is_contiguous())
                    self.assertFalse(bundle[2]["aux"].is_contiguous())
                    with torch.no_grad():
                        bundle[2]["aux"].zero_()
                        if "alpha" in bundle[2]:
                            bundle[2]["alpha"].zero_()
                a, b = forward(old), forward(new)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                a.square().sum().backward()
                b.square().sum().backward()
                for name in old[2]:
                    torch.testing.assert_close(old[2][name].grad, new[2][name].grad, rtol=0, atol=0)

    def test_training_removes_the_three_fa3_select_backward_materializations(self):
        counts = []
        for module in (self.original, self.updated):
            bundle = case(module)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, acc_events=True) as profile:
                forward(bundle).square().sum().backward()
            # The old leading-axis selections pass 3-D dq/dk/dv into aten::select_backward.
            events = [e for e in profile.events() if e.name == "aten::select_backward"
                      and e.input_shapes and len(e.input_shapes[0]) == 3]
            counts.append(len(events))
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 0, "FA3 output gradients still materialize through leading-axis selects")

    def test_zero_values_and_no_gain_are_finite_at_the_minimum_token_multiple(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                bundles = [case(m, paired=paired, gate=True, tokens=16) for m in (self.original, self.updated)]
                for bundle in bundles:
                    bundle[1].o_gain = None
                    with torch.no_grad():
                        bundle[2]["v"].zero_()  # XSA must take the zero-norm clamp branch
                a, b = (forward(bundle) for bundle in bundles)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                a.sum().backward()
                b.sum().backward()
                for name in bundles[0][2]:
                    ga, gb = (bundle[2][name].grad for bundle in bundles)
                    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
                    if ga is not None:
                        self.assertTrue(torch.isfinite(ga).all())

    def test_actual_no_grad_validation_path_matches(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                with torch.no_grad():
                    outputs = [forward(case(m, paired=paired, aux=True, gate=True, fp8=False,
                                            qk=128, value=64)) for m in (self.original, self.updated)]
                torch.testing.assert_close(*outputs, rtol=0, atol=0)
                self.assertFalse(outputs[1].requires_grad)

    def test_repeated_training_updates_match_weights_gradients_and_adam_moments(self):
        # Actual production attention forward bodies, real autograd and Adam updates. GPU-only
        # operators use the documented CPU references; this is not production FP8/FA3 validation.
        from tests.training_loop import assert_tree_equal
        branches = [case(m, aux=True, gate=True, qk=128, value=64, key_offset=True, dtype=torch.float64)
                    for m in (self.original, self.updated)]
        optimizers = [torch.optim.AdamW(list(b[2].values()), lr=.001) for b in branches]
        initial = branches[0][2]["qk"].detach().clone()
        for step in range(32):
            losses = []
            for bundle, optimizer in zip(branches, optimizers):
                optimizer.zero_grad(set_to_none=True)
                logits = forward(bundle)
                labels = (torch.arange(32) + step) % 24
                loss = torch.nn.functional.cross_entropy(logits[0], labels)
                loss.backward()
                losses.append(loss.detach())
            assert_tree_equal(losses[0], losses[1])
            for name in branches[0][2]:
                assert_tree_equal(branches[0][2][name].grad, branches[1][2][name].grad)
            for optimizer in optimizers:
                optimizer.step()
            assert_tree_equal(branches[0][2], branches[1][2])
            assert_tree_equal(optimizers[0].state_dict(), optimizers[1].state_dict())
        self.assertFalse(torch.equal(initial, branches[0][2]["qk"]))


if __name__ == "__main__":
    unittest.main()
