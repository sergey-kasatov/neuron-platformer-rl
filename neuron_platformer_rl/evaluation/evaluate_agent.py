"""Evaluate a trained agent on held-out seeds (10000+).

Defaults reproduce the historical v1 numbers: state model, easy, 30
episodes, deterministic. The rgb mode rebuilds the training-time view of
the pixel agent (VecFrameStack of 4 clean 84x84 frames); --recurrent
rebuilds the v4 LSTM view instead (single frames, context in the hidden
state).

Run:  python -m neuron_platformer_rl.evaluation.evaluate_agent
      [--model models/pixels_best/best_model.zip --obs rgb]
      [--model models/lstm_hard_best/best_model.zip --obs rgb --recurrent]
      [--difficulty easy] [--episodes 30] [--stochastic] [--device cpu]
"""
import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv

ROOT = Path(__file__).resolve().parents[2]


def run_episode_state(model, difficulty, seed, deterministic):
    env = NeuronPlatformerEnv(render_mode=None, difficulty=difficulty, seed=seed,
                              observation_mode="state")
    obs, info = env.reset()
    done, total = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        done = terminated or truncated
    env.close()
    return info, total


def run_episode_rgb(model, difficulty, seed, deterministic):
    make = lambda: NeuronPlatformerEnv(render_mode=None, difficulty=difficulty,
                                       seed=seed, observation_mode="rgb")
    venv = VecFrameStack(DummyVecEnv([make]), 4)
    obs = venv.reset()
    done, total = False, 0.0
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)
        total += float(rewards[0])
        done = bool(dones[0])
        info = infos[0]   # on done this is the terminal step's info
    venv.close()
    return info, total


def run_episode_lstm(model, difficulty, seed, deterministic):
    # No frame stack: the LSTM carries velocity and short-term context, so the
    # policy sees one raw frame and the hidden state has to be threaded by hand.
    # episode_start=True only on the first step, which is what resets the state.
    env = NeuronPlatformerEnv(render_mode=None, difficulty=difficulty, seed=seed,
                              observation_mode="rgb")
    obs, info = env.reset()
    state, done, total = None, False, 0.0
    episode_start = np.ones((1,), dtype=bool)
    while not done:
        action, state = model.predict(obs, state=state, episode_start=episode_start,
                                      deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        episode_start = np.zeros((1,), dtype=bool)
        total += reward
        done = terminated or truncated
    env.close()
    return info, total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "ppo_neuron_platformer_v1_state.zip"))
    ap.add_argument("--obs", choices=("state", "rgb"), default="state")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of argmax")
    ap.add_argument("--recurrent", action="store_true",
                    help="load a RecurrentPPO (LSTM) model: single frames, no stack")
    ap.add_argument("--device", default="auto",
                    help="torch device; pass cpu to leave the GPU to a training run")
    args = ap.parse_args()
    if args.recurrent and args.obs != "rgb":
        ap.error("--recurrent implies --obs rgb (the LSTM policy is a CNN policy)")

    if args.recurrent:
        model = RecurrentPPO.load(args.model, device=args.device)
        run_episode = run_episode_lstm
    else:
        model = PPO.load(args.model, device=args.device)
        run_episode = run_episode_state if args.obs == "state" else run_episode_rgb
    results = []
    for i in range(args.episodes):
        info, total = run_episode(model, args.difficulty, 10_000 + i,
                                  not args.stochastic)
        results.append((info["success"], total, info["chips_collected"],
                        info["enemy_kills"], info["x"]))
    n = len(results)
    print("Evaluation on unseen seeds")
    print(f"Model: {args.model}")
    print(f"Mode: {args.obs}{' + lstm' if args.recurrent else ''}, "
          f"difficulty: {args.difficulty}, "
          f"{'stochastic' if args.stochastic else 'deterministic'}")
    print(f"Episodes: {n}")
    print(f"Success rate: {sum(r[0] for r in results) / n:.2%}")
    print(f"Average reward: {sum(r[1] for r in results) / n:.2f}")
    print(f"Average chips: {sum(r[2] for r in results) / n:.2f}")
    print(f"Average enemy kills: {sum(r[3] for r in results) / n:.2f}")
    print(f"Average x position: {sum(r[4] for r in results) / n:.1f}")
