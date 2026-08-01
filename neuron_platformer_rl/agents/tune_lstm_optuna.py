"""Optuna search for RecurrentPPO (LSTM) hyperparameters.

The v3 pixel agent is reactive: 4 stacked frames, no memory, and its hard
ceiling (53%) motivates recurrence. This study tunes RecurrentPPO with a
CnnLstmPolicy on SINGLE frames (no frame stack - velocity must be carried
by the LSTM), short 800k-step trials on easy, objective = deterministic
success rate on the 30 held-out seeds. Median-pruned, resumable (sqlite in
logs/). Best params land in logs/optuna_lstm_best.json for the final run.

Run:  .venv/Scripts/python.exe -m neuron_platformer_rl.agents.tune_lstm_optuna [n_trials]
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless pygame

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from neuron_platformer_rl.agents.train_ppo_pixels import (
    N_EVAL_ENVS, keep_windows_awake, make_env, make_eval_env)

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

N_ENVS = 24
# Sized to the measured RecurrentPPO throughput on this laptop (~150-250
# fps, 6-8x slower than plain PPO): 600k-step trials, one prune point.
TRIAL_STEPS = 600_000
EVAL_EVERY = 300_000


def held_out_success(model, eval_env) -> float:
    """Deterministic success rate over one full pass of the 30 held-out
    seeds (evaluate_policy threads LSTM states through predict)."""
    successes = []

    def on_step(locals_, globals_):
        for i, done in enumerate(locals_["dones"]):
            if done:
                successes.append(bool(locals_["infos"][i].get("is_success")))

    evaluate_policy(model, eval_env, n_eval_episodes=30, deterministic=True,
                    callback=on_step, warn=False)
    return float(np.mean(successes)) if successes else 0.0


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("learning_rate", 8e-5, 4e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.003, 0.03, log=True)
    n_steps = trial.suggest_categorical("n_steps", [128, 256])
    lstm_hidden = trial.suggest_categorical("lstm_hidden_size", [128, 256])
    n_epochs = trial.suggest_categorical("n_epochs", [4, 8])

    env = DummyVecEnv([make_env() for _ in range(N_ENVS)])          # no frame stack
    eval_env = DummyVecEnv([make_eval_env(r) for r in range(N_EVAL_ENVS)])
    model = RecurrentPPO(
        "CnnLstmPolicy", env, verbose=0, seed=1000 + trial.number,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=n_steps * N_ENVS // 8,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        # One shared LSTM for actor and critic: the separate critic LSTM
        # (sb3-contrib default) nearly doubles recurrent compute, and this
        # hardware is the constraint.
        policy_kwargs=dict(lstm_hidden_size=lstm_hidden, shared_lstm=True,
                           enable_critic_lstm=False),
    )
    try:
        for chunk_end in range(EVAL_EVERY, TRIAL_STEPS + 1, EVAL_EVERY):
            # total_timesteps is a TOTAL target, not an increment: with
            # reset_num_timesteps=False the counter carries over, so each
            # chunk must aim at its own cumulative end.
            model.learn(total_timesteps=chunk_end, reset_num_timesteps=False)
            score = held_out_success(model, eval_env)
            trial.report(score, chunk_end)
            print(f"[trial {trial.number}] {chunk_end:,} steps: success {score:.1%}",
                  flush=True)
            if trial.should_prune():
                raise optuna.TrialPruned()
    finally:
        env.close()
        eval_env.close()
    return score


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    keep_windows_awake()
    print("device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    study = optuna.create_study(
        study_name="rppo_lstm_easy",
        direction="maximize",
        storage=f"sqlite:///{(LOG_DIR / 'optuna_lstm.db').as_posix()}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=EVAL_EVERY - 1),
    )
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params, best_value=study.best_value)
    out = LOG_DIR / "optuna_lstm_best.json"
    out.write_text(json.dumps(best, indent=2))
    print("best:", json.dumps(best), "->", out)
