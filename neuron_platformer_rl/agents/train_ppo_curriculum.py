"""Curriculum over difficulties for the pixel agent (roadmap v1.6).

Continues from the v2 pixel model (trained on easy) and walks the
difficulty ladder with mixed env pools, so earlier tiers are rehearsed
while a new tier is learned:

    0.0M - 2.0M   demo x24            (bridge: wider levels, more air)
    2.0M - 5.5M   medium x16, demo x8
    5.5M - 9.0M   hard x16, medium x8

Difficulty switches go through env_method("set_difficulty", ...) and take
effect on each env's next reset. Two EvalCallbacks track medium and hard
on the same 30 held-out seeds (10000-10029) throughout the whole run.

Run:  .venv/Scripts/python.exe -m neuron_platformer_rl.agents.train_ppo_curriculum [total_steps]
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless pygame

import ctypes
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from neuron_platformer_rl.agents.train_ppo_pixels import (
    EVAL_SEEDS, N_ENVS, N_EVAL_ENVS, SeedCycleEnv, keep_windows_awake, make_env)
from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

START_MODEL = MODEL_DIR / "ppo_neuron_platformer_v2_pixels.zip"

# (from_step, [(difficulty, env_count), ...] summing to N_ENVS)
STAGES = [
    (0,         [("demo", 24)]),
    (2_000_000, [("medium", 16), ("demo", 8)]),
    (5_500_000, [("hard", 16), ("medium", 8)]),
]


class DifficultyScheduleCallback(BaseCallback):
    """Apply the stage table to the live envs as training crosses stage
    boundaries. A switch only changes what the next reset generates, so
    in-flight episodes finish on their old difficulty."""

    def __init__(self, stages):
        super().__init__()
        self.stages = list(stages)
        self.applied = -1

    def _apply(self, idx):
        mix = self.stages[idx][1]
        rank = 0
        for difficulty, count in mix:
            self.training_env.env_method("set_difficulty", difficulty,
                                         indices=range(rank, rank + count))
            rank += count
        self.applied = idx
        desc = ", ".join(f"{d} x{c}" for d, c in mix)
        print(f"[curriculum] step {self.num_timesteps:,}: {desc}", flush=True)

    def _on_step(self):
        due = max(i for i, (start, _) in enumerate(self.stages)
                  if self.num_timesteps >= start)
        if due != self.applied:
            self._apply(due)
        return True


def make_eval_env(difficulty, rank):
    def _init():
        env = NeuronPlatformerEnv(render_mode=None, difficulty=difficulty,
                                  observation_mode="rgb")
        return Monitor(SeedCycleEnv(env, EVAL_SEEDS[rank::N_EVAL_ENVS]))
    return _init


if __name__ == "__main__":
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 9_000_000
    keep_windows_awake()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    env = VecFrameStack(DummyVecEnv([make_env() for _ in range(N_ENVS)]), 4)

    callbacks = [DifficultyScheduleCallback(STAGES)]
    for difficulty in ("medium", "hard"):
        eval_env = VecFrameStack(
            DummyVecEnv([make_eval_env(difficulty, r) for r in range(N_EVAL_ENVS)]), 4)
        # 500k, not the pixel phase's 250k: each point now costs two 30-episode
        # evals and medium/hard episodes run up to 2000/2400 steps.
        callbacks.append(EvalCallback(
            eval_env,
            n_eval_episodes=30,
            eval_freq=500_000 // N_ENVS,
            best_model_save_path=str(MODEL_DIR / f"pixels_{difficulty}_best"),
            log_path=str(LOG_DIR / f"eval_pixels_{difficulty}"),
            deterministic=True,
            verbose=1,
        ))
    callbacks.append(CheckpointCallback(
        save_freq=1_000_000 // N_ENVS,
        save_path=str(LOG_DIR / "checkpoints_curriculum"),
        name_prefix="ppo_curriculum",
    ))

    # The saved schedules resume where the 10M run ended (lr ~0), so both
    # get replaced: a lower fresh peak for fine-tuning, same linear decay.
    model = PPO.load(
        START_MODEL, env=env, device=device,
        custom_objects={"learning_rate": lambda progress: progress * 1.5e-4},
    )
    model.learn(total_timesteps=total_steps, callback=callbacks,
                tb_log_name="PPO_curriculum", reset_num_timesteps=True)
    model.save(MODEL_DIR / "ppo_neuron_platformer_v3_curriculum")
    print("Saved last model to:", MODEL_DIR / "ppo_neuron_platformer_v3_curriculum.zip")
    print("Best-on-eval models in:", MODEL_DIR / "pixels_medium_best",
          "and", MODEL_DIR / "pixels_hard_best")
