"""Evaluate a trained agent on held-out seeds (10000+).

Defaults reproduce the historical v1 numbers: state model, easy, 30
episodes, deterministic. The rgb mode rebuilds the training-time view of
the pixel agent (VecFrameStack of 4 clean 84x84 frames).

Run:  python -m neuron_platformer_rl.evaluation.evaluate_agent
      [--model models/pixels_best/best_model.zip --obs rgb]
      [--difficulty easy] [--episodes 30] [--stochastic]
"""
import argparse
from pathlib import Path

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "ppo_neuron_platformer_v1_state.zip"))
    ap.add_argument("--obs", choices=("state", "rgb"), default="state")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of argmax")
    args = ap.parse_args()

    model = PPO.load(args.model)
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
    print(f"Mode: {args.obs}, difficulty: {args.difficulty}, "
          f"{'stochastic' if args.stochastic else 'deterministic'}")
    print(f"Episodes: {n}")
    print(f"Success rate: {sum(r[0] for r in results) / n:.2%}")
    print(f"Average reward: {sum(r[1] for r in results) / n:.2f}")
    print(f"Average chips: {sum(r[2] for r in results) / n:.2f}")
    print(f"Average enemy kills: {sum(r[3] for r in results) / n:.2f}")
    print(f"Average x position: {sum(r[4] for r in results) / n:.1f}")
