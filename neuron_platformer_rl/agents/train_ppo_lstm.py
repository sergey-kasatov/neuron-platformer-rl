"""RecurrentPPO (LSTM) final run: the recurrence ceiling experiment (v4).

The v3 pixel agent is reactive (4 stacked frames, no memory); this run
replaces the frame stack with an LSTM on SINGLE frames - velocity and
short-term context must be carried by the recurrent state. Hyperparameters
come from the Optuna study (logs/optuna_lstm_best.json, written by
tune_lstm_optuna.py); the difficulty ladder is the v3 curriculum grown
from scratch, since LSTM weights cannot start from the CNN-only v2/v3.

Run:  .venv/Scripts/python.exe -m neuron_platformer_rl.agents.train_ppo_lstm [total_steps]
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless pygame

import json
import sys
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from neuron_platformer_rl.agents.train_ppo_curriculum import DifficultyScheduleCallback
from neuron_platformer_rl.agents.train_ppo_curriculum import make_eval_env as make_diff_eval_env
from neuron_platformer_rl.agents.train_ppo_pixels import (
    N_ENVS, N_EVAL_ENVS, keep_windows_awake, make_env)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

DEFAULTS = dict(learning_rate=2.5e-4, ent_coef=0.01, n_steps=256,
                lstm_hidden_size=256, n_epochs=4)

# From scratch (no v2/v3 warm start), so the ladder begins on easy.
# 8M total: sized to RecurrentPPO's measured ~4-6x-slower-than-PPO
# throughput; the README must state the budget difference vs v3.
STAGES = [
    (0,         [("easy", 24)]),
    (2_500_000, [("demo", 24)]),
    (4_000_000, [("medium", 16), ("demo", 8)]),
    (6_000_000, [("hard", 16), ("medium", 8)]),
]


if __name__ == "__main__":
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 8_000_000
    keep_windows_awake()
    # TRAIN_DEVICE=cpu forces a CPU run (used for pipeline sanity checks
    # while the GPU is busy with another training process).
    device = os.environ.get("TRAIN_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    best_file = LOG_DIR / "optuna_lstm_best.json"
    params = dict(DEFAULTS)
    if best_file.exists():
        loaded = json.loads(best_file.read_text())
        params.update({k: loaded[k] for k in DEFAULTS if k in loaded})
        print("hyperparameters from", best_file, "->", params)
    else:
        print("no Optuna result found, using defaults:", params)

    env = DummyVecEnv([make_env() for _ in range(N_ENVS)])    # single frames, no stack

    callbacks = [DifficultyScheduleCallback(STAGES)]
    for difficulty in ("medium", "hard"):
        eval_env = DummyVecEnv(
            [make_diff_eval_env(difficulty, r) for r in range(N_EVAL_ENVS)])
        callbacks.append(EvalCallback(
            eval_env,
            n_eval_episodes=30,
            eval_freq=500_000 // N_ENVS,
            best_model_save_path=str(MODEL_DIR / f"lstm_{difficulty}_best"),
            log_path=str(LOG_DIR / f"eval_lstm_{difficulty}"),
            deterministic=True,
            verbose=1,
        ))
    callbacks.append(CheckpointCallback(
        save_freq=1_000_000 // N_ENVS,
        save_path=str(LOG_DIR / "checkpoints_lstm"),
        name_prefix="ppo_lstm",
    ))

    peak_lr = params["learning_rate"]
    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=lambda progress: progress * peak_lr,   # linear decay
        n_steps=params["n_steps"],
        batch_size=params["n_steps"] * N_ENVS // 8,
        n_epochs=params["n_epochs"],
        ent_coef=params["ent_coef"],
        # Shared LSTM, matching the tuning study (separate critic LSTM
        # nearly doubles recurrent compute on this hardware).
        policy_kwargs=dict(lstm_hidden_size=params["lstm_hidden_size"],
                           shared_lstm=True, enable_critic_lstm=False),
        tensorboard_log=str(LOG_DIR / "tensorboard"),
    )
    model.learn(total_timesteps=total_steps, callback=callbacks,
                tb_log_name="PPO_lstm")
    model.save(MODEL_DIR / "ppo_neuron_platformer_v4_lstm")
    print("Saved last model to:", MODEL_DIR / "ppo_neuron_platformer_v4_lstm.zip")
    print("Best-on-eval models in:", MODEL_DIR / "lstm_medium_best",
          "and", MODEL_DIR / "lstm_hard_best")
