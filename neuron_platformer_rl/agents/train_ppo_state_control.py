"""State-vector PPO at the pixel agent's budget: the control run behind v1 vs v2.

The published comparison gives the state baseline 2M steps (train_ppo.py) and
the pixel agent 10M (train_ppo_pixels.py), so "pixels beat hand-crafted
features" is confounded with training budget. This script re-runs the v1
configuration unchanged (MlpPolicy, 8 envs, the same PPO hyperparameters) for
a pixel-sized budget and adds only what v2 already had: an EvalCallback on the
same 30 held-out seeds every 250k steps (a best-on-eval snapshot plus the
learning curve, which is what answers "had v1 plateaued at 2M"), a checkpoint,
and the keep-awake call. Every output goes to its own path; the committed v1
model is not touched.

Run:  python -m neuron_platformer_rl.agents.train_ppo_state_control
      [--steps 10000000] [--eval-every 250000] [--tag state_control]
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless pygame

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from neuron_platformer_rl.agents.train_ppo_pixels import (
    EVAL_SEEDS, N_EVAL_ENVS, SeedCycleEnv, keep_windows_awake)
from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Same as train_ppo.py: the control changes the budget and nothing else.
N_ENVS = 8


def make_env():
    def _init():
        return Monitor(NeuronPlatformerEnv(render_mode=None, difficulty="easy",
                                           observation_mode="state"))
    return _init


def make_eval_env(rank):
    def _init():
        env = NeuronPlatformerEnv(render_mode=None, difficulty="easy",
                                  observation_mode="state")
        return Monitor(SeedCycleEnv(env, EVAL_SEEDS[rank::N_EVAL_ENVS]))
    return _init


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--tag", default="state_control")
    args = ap.parse_args()
    keep_windows_awake()

    env = DummyVecEnv([make_env() for _ in range(N_ENVS)])
    eval_env = DummyVecEnv([make_eval_env(r) for r in range(N_EVAL_ENVS)])

    # Callback frequencies count per-env steps, hence the division.
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=30,
        eval_freq=args.eval_every // N_ENVS,
        best_model_save_path=str(MODEL_DIR / f"{args.tag}_best"),
        log_path=str(LOG_DIR / f"eval_{args.tag}"),
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=1_000_000 // N_ENVS,
        save_path=str(LOG_DIR / f"checkpoints_{args.tag}"),
        name_prefix="ppo_state",
    )

    # Hyperparameters copied verbatim from train_ppo.py. Device pinned to CPU:
    # that is where v1 trained (the venv had no CUDA torch yet), an MLP on 8
    # envs gains nothing from the GPU, and the GPU stays free.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log=str(LOG_DIR / "tensorboard"),
    )
    model.learn(total_timesteps=args.steps, callback=[eval_cb, ckpt_cb],
                tb_log_name=f"PPO_{args.tag}")
    model.save(MODEL_DIR / f"ppo_neuron_platformer_{args.tag}_final")
    print("Saved last model to:", MODEL_DIR / f"ppo_neuron_platformer_{args.tag}_final.zip")
    print("Best-on-eval model in:", MODEL_DIR / f"{args.tag}_best")
