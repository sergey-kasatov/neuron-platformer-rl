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
    env = VecMonitor(DummyVecEnv([make_env]), str(LOG_DIR))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        tensorboard_log=str(LOG_DIR / "tensorboard"),
    )
    model.learn(total_timesteps=50_000)
    model.save(MODEL_DIR / "ppo_neuron_platformer_v1_state")
    print(f"Saved model to: {MODEL_DIR / 'ppo_neuron_platformer_v1_state.zip'}")
