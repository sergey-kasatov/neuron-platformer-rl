from pathlib import Path
from stable_baselines3 import PPO
from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "ppo_neuron_platformer_v1_state.zip"

if __name__ == "__main__":
    env = NeuronPlatformerEnv(render_mode="human", difficulty="demo", seed=42, observation_mode="state", debug=True)
    model = PPO.load(MODEL_PATH)
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()
        if terminated or truncated:
            print(info)
            obs, info = env.reset()
