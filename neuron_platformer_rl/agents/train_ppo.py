from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

def make_env():
    # State mode is used first because it trains much faster than pixels.
    return NeuronPlatformerEnv(render_mode=None, difficulty="easy", observation_mode="state")

if __name__ == "__main__":
    # 8 parallel envs: the env itself is cheap (~19k steps/s), so batching
    # policy inference across environments is where the wall-clock win is.
    env = VecMonitor(DummyVecEnv([make_env] * 8), str(LOG_DIR))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        # Long rollouts: pit jumps and the flag are rare, sparse events, and
        # 512-step fragments rarely contained a full jump-death-retry cycle.
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        # SB3 defaults to ent_coef=0.0; two 500k runs collapsed to a
        # deterministic policy (entropy loss -> 0) long before convergence.
        # A small entropy bonus keeps exploration alive.
        ent_coef=0.01,
        tensorboard_log=str(LOG_DIR / "tensorboard"),
    )
    # Roughly 20-30 minutes on CPU with 8 envs. Shorter runs plateaued:
    # 50k only proved the pipeline and 500k single-env runs surfaced (and
    # fixed) pit-blind observations, entropy collapse, a reward farm and an
    # over-harsh death penalty, but still won only rarely. Sparse events
    # (edge jumps, the flag) need this much experience.
    model.learn(total_timesteps=2_000_000)
    model.save(MODEL_DIR / "ppo_neuron_platformer_v1_state")
    print(f"Saved model to: {MODEL_DIR / 'ppo_neuron_platformer_v1_state.zip'}")
