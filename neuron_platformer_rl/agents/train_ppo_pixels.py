"""PPO on raw pixels (roadmap v1.4): VecFrameStack(4) + CnnPolicy.

The state-vector agent (train_ppo.py) proves the task is learnable; this
phase is the computer-vision story: the agent must read platforms, pits,
enemies and the portal from the clean 84x84 frames alone. Colour channels
are kept (no grayscale) because entity classes are colour-coded.

Run:  .venv/Scripts/python.exe -m neuron_platformer_rl.agents.train_ppo_pixels [total_steps]
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless pygame

import ctypes
import sys
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# DummyVecEnv on purpose: after the 84x84 direct-draw optimization one env
# step costs ~0.2 ms, so SubprocVecEnv pays more for Windows pipe round
# trips than the envs cost to step. Measured on the RTX 3070 laptop:
# subproc x8 ~350 fps, dummy x8 ~550, dummy x24 ~1000 fps steady.
N_ENVS = 24
N_EVAL_ENVS = 6
# Held out: training resets draw random seeds, evaluation always plays these.
# Same set as scripts in evaluation/, so numbers are comparable across phases.
EVAL_SEEDS = list(range(10_000, 10_030))


class SeedCycleEnv(gym.Wrapper):
    """Cycle a fixed seed list, one seed per reset. Each eval env gets a
    distinct slice of the 30 held-out seeds; 6 envs x 5 episodes means every
    evaluation covers exactly the same 30 levels, so eval curves compare."""

    def __init__(self, env, seeds):
        super().__init__(env)
        self.seeds = list(seeds)
        self.i = 0

    def reset(self, **kwargs):
        kwargs["seed"] = self.seeds[self.i % len(self.seeds)]
        self.i += 1
        return self.env.reset(**kwargs)


def make_env():
    def _init():
        return Monitor(NeuronPlatformerEnv(render_mode=None, difficulty="easy",
                                           observation_mode="rgb"))
    return _init


def make_eval_env(rank):
    def _init():
        env = NeuronPlatformerEnv(render_mode=None, difficulty="easy",
                                  observation_mode="rgb")
        return Monitor(SeedCycleEnv(env, EVAL_SEEDS[rank::N_EVAL_ENVS]))
    return _init


def keep_windows_awake():
    # Run 3 of the state phase died overnight at 412k steps when the laptop
    # went to sleep (fps 549 -> 12). ES_SYSTEM_REQUIRED blocks idle sleep
    # while this process lives and clears when it exits. Lid close still
    # sleeps the machine, so leave the lid open.
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


if __name__ == "__main__":
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    keep_windows_awake()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    env = VecFrameStack(DummyVecEnv([make_env() for _ in range(N_ENVS)]), 4)
    eval_env = VecFrameStack(DummyVecEnv([make_eval_env(r) for r in range(N_EVAL_ENVS)]), 4)

    # Callback frequencies count per-env steps, hence the division.
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=30,
        eval_freq=250_000 // N_ENVS,
        best_model_save_path=str(MODEL_DIR / "pixels_best"),
        log_path=str(LOG_DIR / "eval_pixels"),
        deterministic=True,
    )
    # The state phase lost a whole overnight run to an interruption; a
    # checkpoint every 1M steps caps any future loss at minutes, not hours.
    ckpt_cb = CheckpointCallback(
        save_freq=1_000_000 // N_ENVS,
        save_path=str(LOG_DIR / "checkpoints_pixels"),
        name_prefix="ppo_pixels",
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        # Linear decay is the standard schedule for pixel PPO; the state
        # phase used a constant rate, but 10M steps benefit from annealing.
        learning_rate=lambda progress: progress * 2.5e-4,
        # 341 x 24 envs = 8184 per rollout, close to the state phase's 8 x
        # 2048. Per-env fragments are shorter than an episode here, but GAE
        # bootstraps fragment ends with V(s), and each rollout still sees
        # ~10 complete episodes across the 24 envs.
        n_steps=341,
        batch_size=264,   # 8184 / 264 = 31 minibatches exactly, no truncated tail
        # Atari-style: fewer passes over a much larger rollout than the
        # 64-sample minibatches of the state phase.
        n_epochs=4,
        # Entropy-collapse fix carried over from the state phase.
        ent_coef=0.01,
        tensorboard_log=str(LOG_DIR / "tensorboard"),
    )
    model.learn(total_timesteps=total_steps, callback=[eval_cb, ckpt_cb],
                tb_log_name="PPO_pixels")
    model.save(MODEL_DIR / "ppo_neuron_platformer_v2_pixels")
    print("Saved last model to:", MODEL_DIR / "ppo_neuron_platformer_v2_pixels.zip")
    print("Best-on-eval model in:", MODEL_DIR / "pixels_best")
