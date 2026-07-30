from neuron_platformer_rl.envs.platformer_env import NeuronPlatformerEnv

env = NeuronPlatformerEnv(render_mode=None, difficulty="easy", observation_mode="state")
obs, info = env.reset(seed=123)
print("Observation shape:", obs.shape)
print("Info:", info)
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("Environment test passed")
