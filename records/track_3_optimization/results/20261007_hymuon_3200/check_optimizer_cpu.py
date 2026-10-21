#!/usr/bin/env python3
"""Small CPU-only checks of the actual archived/released optimizer definitions."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    path = Path(__file__).with_name("train_gpt_hymuon.py")
    tree = ast.parse(path.read_text())
    names = {"zeropower_via_newtonschulz", "muon_update", "nested_muon_update", "Muon"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    # Run the same function bodies eagerly on CPU; do not invoke torch.compile.
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    scope = dict(torch=torch, Tensor=torch.Tensor,
                 dist=SimpleNamespace(get_world_size=lambda: 1, get_rank=lambda: 0,
                                      all_gather=lambda *args: None))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    update, ns = scope["nested_muon_update"], scope["zeropower_via_newtonschulz"]
    torch.manual_seed(17)
    mu, alpha = 0.95, 0.05
    for shape in ((3, 5), (5, 3), (4, 4)):
        gradients = torch.randn(3200, *shape)
        fast = torch.zeros(shape)
        slow = torch.zeros(shape)
        mass = 0.0
        for step, grad in enumerate(gradients, 1):
            mass = mu * mass + (1 - mu)
            actual = update(grad.clone(), fast, slow, mass, step, alpha=alpha)
            if step not in (1, 2, 5, 19, 40, 3200):
                continue
            history = gradients[:step].double()
            indices = torch.arange(1, step + 1, dtype=torch.float64)
            expected_slow = torch.einsum("i,ijk->jk", 2 * indices / (step * (step + 1)), history).float()
            expected_fast = torch.einsum("i,ijk->jk", (1 - mu) * mu ** (step - indices), history).float()
            torch.testing.assert_close(slow, expected_slow, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(fast, expected_fast, rtol=2e-5, atol=2e-6)
            assert abs(mass - (1 - mu ** step)) < 2e-14
            expected_direction = ((1 - mu) * grad + mu * ((1 - alpha) * expected_fast + alpha * mass * expected_slow))
            expected = ns(expected_direction, 12) * max(1, shape[0] / shape[1]) ** 0.5
            torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.02)
            assert torch.isfinite(actual).all()
        # Constant gradients expose incorrect initialization-mass matching.
        constant = torch.randn(shape)
        fast, slow, mass = torch.zeros(shape), torch.zeros(shape), 0.0
        for step in range(1, 41):
            mass = mu * mass + (1 - mu)
            update(constant.clone(), fast, slow, mass, step, alpha=alpha)
            torch.testing.assert_close(slow, constant)
            torch.testing.assert_close(fast, constant * mass)
    # Two matrices ensure the global clock advances once per optimizer step.
    params = [torch.nn.Parameter(torch.randn(3, 5)), torch.nn.Parameter(torch.randn(4, 4))]
    optimizer = scope["Muon"](params, lr=0.03, weight_decay=0.05, nested_alpha=alpha)
    clone_params = clone_optimizer = None
    for step in range(1, 41):
        grads = [torch.randn_like(p) for p in params]
        for p, grad in zip(params, grads):
            p.grad = grad.clone()
        optimizer.step()
        assert optimizer.param_groups[0]["nested_global_step"] == step
        if step == 11:
            clone_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
            clone_optimizer = scope["Muon"](clone_params, lr=0.03, weight_decay=0.05, nested_alpha=alpha)
            clone_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        elif step > 11:
            for p, grad in zip(clone_params, grads):
                p.grad = grad.clone()
            clone_optimizer.step()
            for a, b in zip(params, clone_params):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert clone_optimizer.param_groups[0]["nested_global_step"] == step
    print(f"PASS: CPU torch {torch.__version__}; triangular closed form through 3200 toy updates; "
          "fast EMA/mass; tall, wide, square NS12; constant-gradient initialization; "
          "two-matrix clock and exact optimizer state round trip. No model training or GPU use.")


if __name__ == "__main__":
    main()
