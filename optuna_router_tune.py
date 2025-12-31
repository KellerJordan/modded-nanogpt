#!/usr/bin/env python3
import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import optuna
import torch
import wandb

import loss_adjustment as loss_adj
import train_switch_bank as tsb
from switch_bank.model.gpt import _second_expert_step

RUN_WANDB_PROJECT = "switch-bank-router-tune4"
OVERVIEW_WANDB_PROJECT = "switch-bank-router-tune-overview4"
OVERVIEW_RUN_NAME = "optuna_router_tune4"
RUN_NAME_TEMPLATE = "trial_{trial:04d}"
OVERVIEW_STEP_OFFSET = 1
FORCE_WANDB_RUN_ID = True
WANDB_RUN_ID_PREFIX = "optuna-router"
WANDB_RESUME_MODE = "never"

DEFAULT_TRIALS = 52
DEFAULT_STUDY_NAME = "router_tune4_rebuild"
DEFAULT_RESULTS_DIR = "optuna_results"
DEFAULT_STORAGE_TEMPLATE = "sqlite:///{results_dir}/optuna_study.db"
LOSS_CALIBRATION_PATH = "optuna_results/loss_adjustment.json"
DEFAULT_SEED = 1337

LOGIT_OBJECTIVE_WEIGHT = 0.35
BEST_SCORE_LOGIT_WEIGHT = 0.2

EARLY_STOP_EXTRA_STEPS = 35
EARLY_STOP_VAL_MULTIPLIER = 1

NSGA2_POPULATION_SIZE = 13

LOSS_REFERENCE_STEP = 750
LOSS_REFERENCE_VALUE = 4.15
LOSS_POWER_FLOOR = 3.205
LOSS_POWER_P = 0.575
LOSS_POWER_A = (LOSS_REFERENCE_VALUE - LOSS_POWER_FLOOR) * (LOSS_REFERENCE_STEP ** LOSS_POWER_P)

TEMP_INIT_RANGE = (1.25, 2.2)
TEMP_FINAL_RANGE = (0.67, 2.0)
TEMP_FINAL_GAP = 0.1
TEMP_ANCHOR_STEPS_RANGE = (650, 1000)

CAP_INIT_RANGE = (0.6, 1.5)
CAP_FINAL_RANGE = (8.0, 20.0)
CAP_FINAL_MIN_GAP = 0.5
CAP_DELTA_RANGE = (575, 850)

TUNED_PARAM_KEYS = (
    "router_temp_init",
    "router_temp_final",
    "router_temp_anchor_delta_steps",
    "router_logit_cap_initial",
    "router_logit_cap_final",
    "router_logit_cap_delta_steps",
)

EXTRA_OVERRIDES = {
    "use_wandb": True,
    "enable_extra_logging": False,
    "enable_extra_wandb_logging": True,
}


def _loss_curve(step: int) -> float:
    step_val = max(int(step), 1)
    return LOSS_POWER_FLOOR + LOSS_POWER_A * (step_val ** (-LOSS_POWER_P))


def _fallback_adjust_loss(val_loss: float, step: int) -> float:
    return float(val_loss) + (_loss_curve(LOSS_REFERENCE_STEP) - _loss_curve(step))


def _suggest_overrides(trial: optuna.Trial) -> dict:
    temp_init = trial.suggest_float("router_temp_init", *TEMP_INIT_RANGE)
    temp_final_max = min(TEMP_FINAL_RANGE[1], temp_init - TEMP_FINAL_GAP)
    temp_final_min = TEMP_FINAL_RANGE[0]
    if temp_final_max < temp_final_min:
        temp_final_max = temp_final_min
    temp_final = trial.suggest_float("router_temp_final", temp_final_min, temp_final_max)
    temp_anchor_steps = trial.suggest_int("router_temp_anchor_delta_steps", *TEMP_ANCHOR_STEPS_RANGE)

    cap_init = trial.suggest_float("router_logit_cap_initial", *CAP_INIT_RANGE)
    cap_final_low = max(CAP_FINAL_RANGE[0], cap_init + CAP_FINAL_MIN_GAP)
    cap_final = trial.suggest_float("router_logit_cap_final", cap_final_low, CAP_FINAL_RANGE[1])
    cap_delta = trial.suggest_int("router_logit_cap_delta_steps", *CAP_DELTA_RANGE)

    overrides = {
        "router_temp_init": temp_init,
        "router_temp_final": temp_final,
        "router_temp_anchor_delta_steps": temp_anchor_steps,
        "router_logit_cap_initial": cap_init,
        "router_logit_cap_final": cap_final,
        "router_logit_cap_delta_steps": cap_delta,
    }
    overrides.update(EXTRA_OVERRIDES)
    overrides["wandb_project"] = RUN_WANDB_PROJECT
    overrides["wandb_run_name"] = RUN_NAME_TEMPLATE.format(trial=trial.number)
    return overrides


def _compute_stop_step(overrides: dict) -> int:
    base_args = tsb.Hyperparameters()
    cap_start = _second_expert_step(tuple(base_args.expert_activation_schedule))
    delta = int(overrides["router_logit_cap_delta_steps"])
    return cap_start + delta + EARLY_STOP_EXTRA_STEPS


def _make_run_id(prefix: str, suffix: str | None = None) -> str:
    token = uuid.uuid4().hex[:8]
    return f"{prefix}-{suffix}-{token}" if suffix else f"{prefix}-{token}"


def _set_env(key: str, value: str | None) -> str | None:
    previous = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return previous


def _force_wandb_run_id(run_id: str) -> tuple[str | None, str | None]:
    old_id = _set_env("WANDB_RUN_ID", run_id)
    old_resume = _set_env("WANDB_RESUME", WANDB_RESUME_MODE)
    return old_id, old_resume


def _restore_wandb_env(old_id: str | None, old_resume: str | None) -> None:
    _set_env("WANDB_RUN_ID", old_id)
    _set_env("WANDB_RESUME", old_resume)


def _attach_trial_params(path: Path, overrides: dict, trial_number: int, study_name: str) -> None:
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except Exception:
        return
    if not isinstance(data, dict):
        return
    data["params"] = {key: overrides[key] for key in TUNED_PARAM_KEYS if key in overrides}
    data["trial_number"] = trial_number
    data["study_name"] = study_name
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _init_overview_run(run_id: str, config: dict | None = None) -> wandb.sdk.wandb_run.Run:
    run = wandb.init(
        project=OVERVIEW_WANDB_PROJECT,
        name=OVERVIEW_RUN_NAME,
        reinit=True,
        id=run_id,
        resume="allow",
        config=config,
    )
    run.define_metric("trial/step")
    run.define_metric("trial/*", step_metric="trial/step")
    run.define_metric("best/*", step_metric="trial/step")
    run.define_metric("overview/*", step_metric="trial/step")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna router temp/logit-cap tuner (single GPU).")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if torch.cuda.device_count() < 1:
        raise SystemExit("CUDA is required for this tuner.")

    os.environ.setdefault("WANDB_REINIT", "1")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    calibration = loss_adj.load_calibration(LOSS_CALIBRATION_PATH)
    reference_step = int(
        calibration.get("reference_step", LOSS_REFERENCE_STEP) if calibration else LOSS_REFERENCE_STEP
    )
    fallback_cfg = {
        "reference_step": LOSS_REFERENCE_STEP,
        "floor": LOSS_POWER_FLOOR,
        "A": LOSS_POWER_A,
        "p": LOSS_POWER_P,
    }

    reuse_state: dict = {}
    overview_config = {
        "study_name": args.study_name,
        "trials": args.trials,
        "seed": args.seed,
        "logit_objective_weight": LOGIT_OBJECTIVE_WEIGHT,
        "best_score_logit_weight": BEST_SCORE_LOGIT_WEIGHT,
        "loss_adjustment": {
            "calibration_path": LOSS_CALIBRATION_PATH,
            "calibration_loaded": bool(calibration),
            "reference_step": reference_step,
            "fallback_reference_step": LOSS_REFERENCE_STEP,
            "fallback_reference_value": LOSS_REFERENCE_VALUE,
            "fallback_floor": LOSS_POWER_FLOOR,
            "fallback_a": LOSS_POWER_A,
            "fallback_p": LOSS_POWER_P,
        },
        "search_space": {
            "router_temp_init": TEMP_INIT_RANGE,
            "router_temp_final": TEMP_FINAL_RANGE,
            "router_temp_anchor_delta_steps": TEMP_ANCHOR_STEPS_RANGE,
            "router_logit_cap_initial": CAP_INIT_RANGE,
            "router_logit_cap_final": CAP_FINAL_RANGE,
            "router_logit_cap_delta_steps": CAP_DELTA_RANGE,
        },
    }

    overview_id = _make_run_id(WANDB_RUN_ID_PREFIX, "overview") if FORCE_WANDB_RUN_ID else None
    overview_id = _make_run_id(WANDB_RUN_ID_PREFIX, "overview") if FORCE_WANDB_RUN_ID else None
    if overview_id:
        overview_run = _init_overview_run(overview_id, config=overview_config)
        overview_run.log({"overview/init": 1, "trial/step": 0, "trial/number": -1}, step=0)
        overview_run.finish()
    best_state: dict[str, Any] = {"score": float("inf")}

    def objective(trial: optuna.Trial):
        overrides = _suggest_overrides(trial)
        stop_step = _compute_stop_step(overrides)
        result_path = results_dir / f"trial_{trial.number:04d}.json"
        old_wandb_id = None
        old_wandb_resume = None
        if FORCE_WANDB_RUN_ID:
            trial_id = _make_run_id(WANDB_RUN_ID_PREFIX, f"trial{trial.number:04d}")
            old_wandb_id, old_wandb_resume = _force_wandb_run_id(trial_id)
        result = tsb.run_training(
            overrides=overrides,
            single_gpu=True,
            early_stop_step=stop_step,
            early_stop_val_multiplier=EARLY_STOP_VAL_MULTIPLIER,
            reuse_state=reuse_state,
            results_path=str(result_path),
            destroy_process_group=False,
            seed=args.seed,
        )
        if FORCE_WANDB_RUN_ID:
            _restore_wandb_env(old_wandb_id, old_wandb_resume)
        _attach_trial_params(result_path, overrides, trial.number, args.study_name)
        val_loss = float("inf")
        val_loss_adj = float("inf")
        logit_score = 0.0
        step_for_loss = stop_step
        if isinstance(result, dict) and not result.get("aborted"):
            val_loss = result.get("val_loss", float("inf"))
            logit_score = result.get("logit_score", 0.0)
            step_for_loss = int(result.get("stop_step", stop_step))
        if val_loss is None or (isinstance(val_loss, float) and math.isnan(val_loss)):
            val_loss = float("inf")
        if logit_score is None or (isinstance(logit_score, float) and math.isnan(logit_score)):
            logit_score = 0.0
        if not math.isinf(val_loss):
            trial_key = RUN_NAME_TEMPLATE.format(trial=trial.number)
            val_loss_adj = loss_adj.adjust_loss(
                float(val_loss),
                int(step_for_loss),
                calibration=calibration,
                trial_key=trial_key,
                fallback=fallback_cfg,
            )
        logit_objective = (1.0 - float(logit_score)) * LOGIT_OBJECTIVE_WEIGHT
        weighted_score = float(val_loss_adj) + BEST_SCORE_LOGIT_WEIGHT * (1.0 - float(logit_score))

        trial.set_user_attr("val_loss", float(val_loss))
        trial.set_user_attr("val_loss_adjusted", float(val_loss_adj))
        trial.set_user_attr("loss_reference_step", reference_step)
        trial.set_user_attr("loss_adjustment_source", "calibration" if calibration else "fallback")
        trial.set_user_attr("loss_step", int(step_for_loss))
        trial.set_user_attr("logit_score", float(logit_score))
        trial.set_user_attr("logit_objective", float(logit_objective))
        trial.set_user_attr("weighted_score", float(weighted_score))
        for key in ("logit_cap_hit_rate", "logit_cap_ratio_mean", "logit_cap_ratio_max", "logit_headroom_mean"):
            if isinstance(result, dict) and key in result:
                trial.set_user_attr(key, float(result[key]))

        if weighted_score < float(best_state["score"]):
            best_state["score"] = weighted_score
            best_state["trial"] = trial.number
            best_state["val_loss"] = float(val_loss)
            best_state["val_loss_adjusted"] = float(val_loss_adj)
            best_state["loss_step"] = int(step_for_loss)
            best_state["logit_score"] = float(logit_score)
            best_state["params"] = dict(overrides)
            best_state["logit_objective"] = float(logit_objective)

        log_payload = {
            "trial/step": trial.number + OVERVIEW_STEP_OFFSET,
            "trial/number": trial.number,
            "trial/val_loss": float(val_loss),
            "trial/val_loss_adjusted": float(val_loss_adj),
            "trial/loss_step": int(step_for_loss),
            "trial/loss_reference_step": reference_step,
            "trial/logit_score": float(logit_score),
            "trial/logit_objective": float(logit_objective),
            "trial/weighted_score": float(weighted_score),
        }
        for key, value in overrides.items():
            log_payload[f"trial/params/{key}"] = value
        log_payload["trial/params_json"] = json.dumps(
            {key: overrides[key] for key in TUNED_PARAM_KEYS if key in overrides},
            sort_keys=True,
        )
        if best_state.get("trial") is not None:
            log_payload["best/score"] = float(best_state["score"])
            log_payload["best/trial"] = int(best_state["trial"])
            log_payload["best/val_loss"] = float(best_state["val_loss"])
            log_payload["best/val_loss_adjusted"] = float(best_state["val_loss_adjusted"])
            log_payload["best/loss_step"] = int(best_state["loss_step"])
            log_payload["best/logit_score"] = float(best_state["logit_score"])
            log_payload["best/logit_objective"] = float(best_state["logit_objective"])
            for key, value in best_state.get("params", {}).items():
                log_payload[f"best/params/{key}"] = value
            if best_state.get("params"):
                log_payload["best/params_json"] = json.dumps(best_state["params"], sort_keys=True)
        if overview_id:
            overview_run = _init_overview_run(overview_id)
            overview_run.log(log_payload, step=trial.number + OVERVIEW_STEP_OFFSET)
            overview_run.finish()

        if not isinstance(result, dict) or result.get("aborted"):
            return float("inf"), float("inf")
        return float(val_loss_adj), float(logit_objective)

    storage = args.storage if args.storage else DEFAULT_STORAGE_TEMPLATE.format(results_dir=results_dir.resolve())
    sampler = optuna.samplers.NSGAIISampler(seed=args.seed, population_size=NSGA2_POPULATION_SIZE)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=args.resume,
        directions=["minimize", "minimize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.trials, n_jobs=1)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    best_path = results_dir / "best.json"
    best_payload = dict(best_state)
    if best_state.get("trial") is None and study.best_trials:
        fallback = study.best_trials[0]
        best_payload = {
            "trial": fallback.number,
            "values": fallback.values,
            "params": fallback.params,
            "user_attrs": fallback.user_attrs,
        }
    best_path.write_text(json.dumps(best_payload, indent=2))
    if overview_id:
        overview_run = _init_overview_run(overview_id)
        overview_run.log(
            {"overview/complete": 1, "trial/step": args.trials + OVERVIEW_STEP_OFFSET},
            step=args.trials + OVERVIEW_STEP_OFFSET,
        )
        overview_run.finish()


if __name__ == "__main__":
    main()
