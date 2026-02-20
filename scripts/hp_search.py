"""Optuna hyperparameter search for Deep Compressor.

Usage:
  # Stage 1 (NTP) — minimize loss
  python scripts/hp_search.py --n_trials 50 --stage 1 \\
      --data_path data/ntp_tiny.jsonl

  # Stage 2 (QA) — maximize F1 (internally minimizes -F1)
  python scripts/hp_search.py --n_trials 30 --stage 2 \\
      --data_path data/qa_tiny_train.json \\
      --eval_data_path data/qa_tiny_dev.json

  # With persistent storage (resumable)
  python scripts/hp_search.py --n_trials 50 --stage 1 \\
      --data_path data/ntp_tiny.jsonl \\
      --storage sqlite:///outputs/hp_search/study.db
"""

import argparse
import gc
import logging

import optuna
import torch

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.train import _run_training

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def suggest_hyperparams(trial: optuna.Trial, stage: int) -> dict:
    """Define the search space. Returns a dict of suggested values."""
    hp = {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True),
        "warmup_steps": trial.suggest_int(
            "warmup_steps", 10, 100, step=10),
        "weight_decay": trial.suggest_float(
            "weight_decay", 0.001, 0.1, log=True),
        "perceiver_dropout": trial.suggest_float(
            "perceiver_dropout", 0.0, 0.3, step=0.05),
        "projection_dropout": trial.suggest_float(
            "projection_dropout", 0.0, 0.3, step=0.05),
    }
    if stage == 2:
        hp.update({
            "kl_temperature": trial.suggest_float(
                "kl_temperature", 1.0, 4.0, step=0.5),
            "kl_weight": trial.suggest_float(
                "kl_weight", 0.1, 2.0, step=0.1),
            "hidden_mse_weight": trial.suggest_float(
                "hidden_mse_weight", 0.1, 2.0, step=0.1),
        })
    return hp


def apply_hyperparams(config: DeepCompressorConfig, hp: dict) -> None:
    """Apply suggested hyperparameters to a config in-place."""
    config.training.learning_rate = hp["learning_rate"]
    config.training.warmup_steps = hp["warmup_steps"]
    config.training.weight_decay = hp["weight_decay"]
    config.perceiver.dropout = hp["perceiver_dropout"]
    config.projection.dropout = hp["projection_dropout"]

    if "kl_temperature" in hp:
        config.loss.kl_temperature = hp["kl_temperature"]
    if "kl_weight" in hp:
        config.loss.kl_weight = hp["kl_weight"]
    if "hidden_mse_weight" in hp:
        config.loss.hidden_mse_weight = hp["hidden_mse_weight"]


def objective(trial: optuna.Trial, args) -> float:
    """Single trial: load config, apply HPs, train, return metric."""
    config = DeepCompressorConfig.from_yaml(args.config)
    config.training.stage = args.stage

    hp = suggest_hyperparams(trial, args.stage)
    apply_hyperparams(config, hp)

    # Use a unique output dir per trial
    config.training.output_dir = f"outputs/hp_search/trial_{trial.number}"

    # Optuna pruning callback
    def _optuna_callback(step: int, metrics: dict):
        if args.stage == 1:
            value = metrics.get("loss", float("inf"))
        else:
            value = -metrics.get("f1", 0.0)  # minimize -F1

        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # wandb disabled for HP search trials (too noisy)
    from types import SimpleNamespace
    wandb_conf = SimpleNamespace(enabled=False)

    try:
        last_metrics = _run_training(
            config=config,
            data_path=args.data_path,
            eval_data_path=args.eval_data_path,
            max_eval_samples=args.max_eval_samples,
            max_train_samples=args.max_train_samples,
            wandb_conf=wandb_conf,
            optuna_callback=_optuna_callback,
        )
    except optuna.TrialPruned:
        raise
    finally:
        # Clean up GPU memory between trials
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if last_metrics is None:
        # No eval was run — return a large loss as penalty
        return float("inf")

    if args.stage == 1:
        return last_metrics.get("loss", float("inf"))
    else:
        return -last_metrics.get("f1", 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for Deep Compressor")
    parser.add_argument("--config", type=str,
                        default="configs/tiny_subset.yaml",
                        help="Base config YAML (defaults to tiny_subset)")
    parser.add_argument("--stage", type=int, required=True,
                        help="Training stage (1 or 2)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///hp.db)")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optuna study name (auto-generated if None)")
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    args = parser.parse_args()

    study_name = args.study_name or f"dc-stage{args.stage}"
    direction = "minimize"  # stage1: loss, stage2: -f1

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
    )

    # ── report results ──
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
