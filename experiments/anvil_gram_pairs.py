"""ANVIL paired-Gram experiment. Not enabled in the trainer; BF16 stability is unresolved.

python -m experiments.anvil_gram_pairs --output work/gram-pairs.json
"""
import argparse
import json
from pathlib import Path

import torch

from track_1_short.optim.spectral_maps import ANVIL_MAPS


def transform(matrix, *, paired=False, dtype=torch.float64):
    """Compose pairs in small Gram space, retaining scalar identity terms in working precision.

    This models BF16 storage boundaries with CPU matmuls; it is not a timed CUDA implementation.
    """
    wide = matrix.shape[-2] < matrix.shape[-1]
    x = (matrix.mT if wide else matrix).to(dtype)
    work = torch.float64 if dtype == torch.float64 else torch.float32
    def mm(a, b):
        return a.to(work) @ b.to(work)
    A = mm(x.mT, x).to(dtype)
    d = A.to(work).diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].sqrt()*1.05+1e-6
    x, A = (x.to(work)/d).to(dtype), (A.to(work)/d.square()).to(dtype)
    for i in range(0, len(ANVIL_MAPS), 2 if paired else 1):
        if i:
            A = mm(x.mT, x).to(dtype)
        a, b, c = ANVIL_MAPS[i]
        B = (b*A.to(work)+c*mm(A, A)).to(dtype)
        if paired:
            # A1=(aI+B) A (aI+B). Keep a outside BF16 diagonal storage.
            D = mm(A, B)
            H = a*a*A.to(work)+a*(D+D.mT)+mm(B, D.to(dtype))
            A1 = ((H+H.mT)*.5).to(dtype)
            a1, b1, c1 = ANVIL_MAPS[i+1]
            B1 = (b1*A1.to(work)+c1*mm(A1, A1)).to(dtype)
            # (aI+B)(a1I+B1)=aa1 I+a1 B+a B1+B B1. Do not mirror the last product.
            combined = (a1*B.to(work)+a*B1.to(work)+mm(B, B1)).to(dtype)
            x = (a*a1*x.to(work)+mm(x, combined)).to(dtype)
        else:
            x = (a*x.to(work)+mm(x, B)).to(dtype)
    return x.mT.contiguous() if wide else x


def diagnostics():
    torch.manual_seed(387)
    u = torch.linalg.qr(torch.randn(128, 32, dtype=torch.float64)).Q
    v = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64)).Q
    cases = {"random": torch.randn(128, 32, dtype=torch.float64),
             "rank-one": torch.randn(128, 1, dtype=torch.float64) @ torch.randn(1, 32, dtype=torch.float64),
             "rotated-ill-conditioned": (u*torch.logspace(0, -8, 32, dtype=torch.float64)) @ v.T,
             "zero": torch.zeros(128, 32, dtype=torch.float64)}
    rows = []
    for name, x in cases.items():
        exact, candidate = transform(x), transform(x, paired=True)
        fp64_error = float((exact-candidate).norm()/exact.norm().clamp_min(1e-30))
        ref, paired = transform(x, dtype=torch.bfloat16).float(), transform(x, paired=True, dtype=torch.bfloat16).float()
        rows.append(dict(case=name, fp64_relative_error=fp64_error,
                         bf16_relative_error=float((ref-paired).norm()/ref.norm().clamp_min(1e-30)),
                         baseline_spectral_norm=float(torch.linalg.svdvals(ref.double()).max()),
                         paired_spectral_norm=float(torch.linalg.svdvals(paired.double()).max())))
    return {"enabled_in_trainer": False, "cuda_timing_measured": False, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = diagnostics()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
