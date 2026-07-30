from pathlib import Path
from stable_baselines3 import PPO
from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "ppo_neuron_platformer_v1_state.zip"

if __name__ == "__main__":
    model = PPO.load(MODEL_PATH)
    episodes = 30
    results = []
    for i in range(episodes):
        env = NeuronPlatformerEnv(render_mode=None, difficulty="easy", seed=10_000+i, observation_mode="state")
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        results.append((info["success"], total_reward, info["chips_collected"], info["enemy_kills"], info["x"]))
        env.close()
    success_rate = sum(r[0] for r in results) / episodes
    avg_reward = sum(r[1] for r in results) / episodes
    avg_chips = sum(r[2] for r in results) / episodes
    avg_kills = sum(r[3] for r in results) / episodes
    avg_x = sum(r[4] for r in results) / episodes
    print("Evaluation on unseen seeds")
    print(f"Episodes: {episodes}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average chips: {avg_chips:.2f}")
    print(f"Average enemy kills: {avg_kills:.2f}")
    print(f"Average x position: {avg_x:.1f}")
