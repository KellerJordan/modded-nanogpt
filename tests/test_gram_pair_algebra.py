import unittest

import torch

from experiments.anvil_gram_pairs import transform


class GramPairAlgebraTests(unittest.TestCase):
    def test_exact_arithmetic_reassociation_for_rectangular_and_degenerate_inputs(self):
        torch.manual_seed(387)
        for shape in ((2, 127, 32), (2, 32, 127), (1, 32, 32)):
            x = torch.randn(shape, dtype=torch.float64)
            for value in (x, torch.zeros_like(x), x[..., :1].expand_as(x).clone()):
                with self.subTest(shape=shape):
                    torch.testing.assert_close(transform(value, paired=True), transform(value), rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
