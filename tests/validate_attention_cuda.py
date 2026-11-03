"""Real pinned FA3/FP8 attention parity on one existing H100; no CPU substitution.

python -m tests.validate_attention_cuda --output work/attention-cuda.json [--compile] [--tokens 16384]
Run this before full-record A/B timing. This diagnostic itself makes no speed claim.
"""
import argparse
import itertools
import json
from pathlib import Path
import types

import torch


def make_case(module, width, value, paired, extras, tokens):
    torch.manual_seed(396)
    heads, dim = 6, 768
    device = torch.device("cuda")
    def leaf(shape, scale):
        return (torch.randn(shape, device=device, dtype=torch.bfloat16)*scale).requires_grad_()
    leaves = dict(x=leaf((1, tokens, dim), .2), qk=leaf((2*heads*width, dim), .02),
                  v=leaf((heads*value, dim), .02), o=leaf((dim, heads*value), .02),
                  scale=torch.tensor([1., .5], device=device, requires_grad=True))
    if extras:
        leaves.update(aux=leaf((1, tokens, heads*128), .1), gate=leaf((1, tokens, heads), .3),
                      gain=torch.tensor(.9, device=device, requires_grad=True))
        if not paired:
            leaves["alpha"] = leaf((1, tokens, heads), .1)
    model = module.CausalSelfAttention(heads, 128, width, value, tokens, paired).cuda()
    yarn = module.Yarn(width, tokens, paired, attn_scale=.12, device=device)
    args = module.AttnArgs(sa_lambdas=leaves["scale"], seqlens=torch.tensor([0, tokens//2, tokens], device=device, dtype=torch.int32),
                          bm_size=128, yarn=yarn, key_offset=False, attn_gate_w=leaves.get("gate"),
                          aux_v=leaves.get("aux"), xsa_alpha=leaves.get("alpha"), train_max_seq_len=tokens,
                          o_gain=leaves.get("gain"))
    xs, ws, gs = (torch.tensor(v, device=device) for v in (.01, .001, .01))
    with torch.no_grad():
        weight = (torch.cat((leaves["qk"], leaves["v"]))/ws).to(torch.float8_e4m3fn).contiguous()
        activation = (leaves["x"][0]/xs).to(torch.float8_e4m3fn).contiguous()
    cache = weight, weight.T.contiguous(), ws, xs, gs, activation, activation.T.contiguous()
    return model, args, leaves, cache


def run(tokens, compiled=False):
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("real attention validation requires an existing CUDA H100 GPU")
    if torch.__version__.split("+")[0] != "2.10.0":
        raise RuntimeError("use the record's Torch 2.10.0 environment")
    from track_1_short.model import attention as module
    from tests.reference_attention import legacy_forward
    # Match train_gpt.py's existing limit; no additional Inductor/compile flags.
    torch._dynamo.config.recompile_limit = 64
    legacy = types.FunctionType(legacy_forward.__code__, vars(module), "legacy_forward")
    interface = module.flash_attn_interface
    actual_fa3 = interface.flash_attn_varlen_func
    def deterministic_fa3(*args, **kwargs):
        return actual_fa3(*args, **kwargs, deterministic=True)
    module.flash_attn_interface = types.SimpleNamespace(flash_attn_varlen_func=deterministic_fa3)
    rows = []
    try:
        for width, value, paired, extras in itertools.product((64, 128), (64, 128), (False, True), (False, True)):
            offsets = (False, True) if width == 128 and not paired else (False,)
            for key_offset in offsets:
                for training in (False, True):
                    results = []
                    for old in (True, False):
                        model, args, leaves, cache = make_case(module, width, value, paired, extras, tokens)
                        args.key_offset = key_offset
                        model.train(training)
                        if old:
                            model.forward = types.MethodType(legacy, model)
                        call = torch.compile(model, dynamic=False, fullgraph=True) if compiled else model
                        with torch.set_grad_enabled(training):
                            output = call(leaves["x"], args, leaves["qk"], leaves["v"], leaves["o"], cache if training else None)
                            if training:
                                upstream = torch.linspace(-.01, .01, output.numel(), device="cuda").reshape_as(output)
                                (output.float()*upstream).sum().backward()
                        results.append((output.detach(), {n: p.grad for n, p in leaves.items()}))
                    errors = []
                    for a, b in [(results[0][0], results[1][0]), *zip(results[0][1].values(), results[1][1].values())]:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                        if a is not None:
                            if not torch.isfinite(a).all():
                                raise AssertionError("non-finite attention output or gradient")
                            errors.append(float((a.float()-b.float()).abs().max()))
                    row = dict(qk=width, value=value, paired=paired, extras=extras, key_offset=key_offset,
                               training=training, max_error=max(errors))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        module.flash_attn_interface = interface
    return dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), tokens=tokens, compiled=compiled,
                deterministic_fa3=True, passed=True, cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.tokens < 16 or args.tokens % 16:
        parser.error("tokens must be a positive multiple of 16")
    report = run(args.tokens, args.compile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
