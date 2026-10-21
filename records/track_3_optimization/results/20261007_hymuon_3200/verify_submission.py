#!/usr/bin/env python3
"""Verify all eight recorded endpoints, source provenance, and baseline invariants.

Standard library only; does not initialize a model, load data, or use a GPU.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

from reproduce_log import extract, record

HERE = Path(__file__).resolve().parent


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def definitions(text):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(text).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def main():
    provenance = json.loads((HERE / "provenance.json").read_text())
    summary = json.loads((HERE / "summary.json").read_text())
    release = (HERE / "train_gpt_hymuon.py").read_text()
    manifest = json.loads((HERE / "release_manifest.json").read_text())
    assert sha(release) == provenance["release_trainer_sha256"] == manifest["trainer_sha256"]
    assert manifest["steps"] == [3200]
    assert manifest["confirmation_seeds"] == list(range(636, 644))
    assert manifest["configs"] == [{"steps": 3200, "alpha": 0.05, "lr": "lr030", "wd": "wd050"}]
    assert provenance["no_retry"] and provenance["development_seeds_excluded"] == [658, 659]
    assert [row["seed"] for row in summary] == list(range(636, 644))
    assert [row["seed"] for row in provenance["guards"]] == list(range(636, 644))
    assert all(g["returncode"] == g["child_returncode"] == 0 and not g["interference"]
               and not g["violations"] for g in provenance["guards"])
    assert sorted(path.name for path in (HERE / "logs").iterdir()) == [f"seed{s}.txt" for s in range(636, 644)]
    values, init_hashes, batch_hashes = [], set(), set()
    source = None
    for seed, row, receipt in zip(range(636, 644), summary, provenance["logs"]):
        text = (HERE / f"logs/seed{seed}.txt").read_text()
        assert receipt["seed"] == seed and sha(text) == receipt["published_log_sha256"]
        archived = extract(text, "ARCHIVED_TRAINER")
        assert sha(archived) == provenance["archived_trainer_sha256"]
        assert source is None or source == archived
        source = archived
        assert sha(extract(text, "OBSERVATIONAL_WRAPPER")) == provenance["observational_wrapper_sha256"]
        console = extract(text, "RECORDED_CONSOLE")
        assert not re.search(r"/(?:DATA|Users|home|root)/", console)
        final = record(console, "FINAL_METRICS_JSON ")
        runtime = record(console, "ACTUAL_RUNTIME_JSON ")
        profile = record(console, "OBSERVATIONAL_RUNTIME_PROFILE_JSON ")
        launch = record(text, "REPRODUCTION_ARGS_JSON ")
        old_manifest = record(text, "REPRODUCTION_MANIFEST_JSON ")
        assert old_manifest["trainer_sha256"] == provenance["archived_trainer_sha256"]
        expected_args = ["--formal", "--arm", "muon12", "--train-steps", "3200", "--seed", str(seed),
                         "--nested-alpha", ".05", "--lr-config", "lr030", "--wd-config", "wd050", "--run-id",
                         f"hymcoreconfirm-seed{seed}-s3200-tri-a050-lr030-wd050"]
        assert launch == expected_args
        assert final["seed"] == runtime["seed"] == seed
        assert final["arm"] == "muon12" and final["nested_alpha"] == 0.05
        assert final["lr_config"] == "lr030" and final["wd_config"] == "wd050"
        assert final["matrix_peak_weight_decay"] == 0.05 and final["slow_decay"] is None
        assert final["step"] == final["updates"] == final["configured_schedule_horizon"] == 3200
        assert not final["integration_gate"] and not final["integration_checkpoint_verified"]
        assert final["primary_readout"] == row["readout"] == "raw"
        assert final["val_loss_x"] is None and final["eval_restore_verified"]
        assert final["val_loss_raw"] == final["primary_val_loss"] == row["val_loss_raw"]
        assert final["trainer_sha256"] == runtime["source_SHA256"] == sha(archived)
        assert profile["source_sha256"] == sha(archived) and profile["status"] == "PASS"
        assert runtime["model_parameters"] == profile["parameter_count"] == 162354816
        assert runtime["world_size"] == 1 and runtime["torch"] == "2.11.0"
        assert not runtime["matmul_allow_tf32"] and runtime["float32_matmul_precision"] == "highest"
        steps = [int(s) for s in re.findall(r"^step:(\d+)/3200 train_time:", console, re.M)]
        assert steps == list(range(1, 3201)), (seed, len(steps))
        vals = [int(s) for s in re.findall(r"^step:(\d+)/3200 val_loss:", console, re.M)]
        expected_vals = [s for s in range(3201) if s == 3200 or s % (125 if s / 3200 < 0.9 else 25) == 0]
        assert vals == expected_vals
        init_hashes.add(record(console, "FULL_INIT_RNG_HASH_JSON ")["model_sha256"])
        first = record(console, "FIRST_TRAIN_BATCH_HASH_JSON ")
        assert first["global_tokens"] == first["local_tokens"] == 524288 and first["seed_independent"]
        batch_hashes.add(first["sha256"])
        values.append(final["val_loss_raw"])
    assert len(init_hashes) == 8 and len(batch_hashes) == 1
    mean = math.fsum(values) / len(values)
    statistic = math.sqrt(len(values)) * (3.28 - mean)
    assert mean == 3.2783215045928955 and statistic == 0.004747501938215692
    assert statistic >= 0.004
    # Every original function/class is unchanged, except manifest discovery.
    archived_defs, release_defs = definitions(source), definitions(release)
    assert archived_defs.keys() == release_defs.keys()
    assert [name for name in archived_defs if archived_defs[name] != release_defs[name]] == ["validate_run_admission"]
    baseline = (HERE.parents[1] / "train_gpt_simple.py").read_text()
    baseline_defs = definitions(baseline)
    fixed = ["_load_data_shard", "distributed_data_generator", "RMSNorm", "Linear", "Rotary",
             "CausalSelfAttention", "MLP", "Block", "GPT"]
    assert all(release_defs[name] == baseline_defs[name] for name in fixed)
    # Resolve the public defaults/admission without importing torch or running setup.
    tree = ast.parse(release)
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "ARM_CONFIGS" for t in n.targets))
    stop = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "device" for t in n.targets))
    import argparse
    import os
    import sys
    from unittest.mock import patch
    scope = dict(argparse=argparse, os=os, sys=sys, Path=Path, hashlib=hashlib,
                 json=json, code=release, __file__=str(HERE / "train_gpt_hymuon.py"))
    with patch.object(sys, "argv", ["train_gpt_hymuon.py"]), patch.dict(os.environ, {}, clear=True):
        exec(compile(ast.Module(body=tree.body[start:stop], type_ignores=[]), "<public setup>", "exec"), scope)
    args = scope["args"]
    assert args.seed == 636 and args.train_steps == scope["run_updates"] == 3200
    assert args.arm == "muon12" and args.nested_alpha == 0.05 and args.formal
    assert args.lr_config == "lr030" and args.wd_config == "wd050"
    assert scope["matrix_peak_lr"] == 0.03 and scope["recipe_config"]["matrix_weight_decay"] == 0.05
    print(json.dumps({"status": "PASS", "runs": 8, "updates_per_run": 3200,
                      "mean_raw": mean, "statistic": statistic, "threshold": 0.004,
                      "all_3200_update_records_present": True,
                      "baseline_model_and_data_AST_equal": True,
                      "release_functions_unchanged_except_manifest_discovery": True,
                      "public_defaults_and_admission": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
