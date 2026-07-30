from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv

if __name__ == "__main__":
    env = NeuronPlatformerEnv(render_mode="human", difficulty="demo", seed=42, observation_mode="state", debug=True)
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print(info)
            obs, info = env.reset()
