import gymnasium as gym
from FreddyLanderAgent import FreddyLanderAgent
from tqdm import tqdm

OFFSCREEN_STEPS = 10000

env = gym.make("LunarLander-v3", render_mode="human")
env_train = gym.make("LunarLander-v3")

agent = FreddyLanderAgent()

observation, info = env_train.reset(seed=42)

for s in tqdm(range(1000000)):
    action = agent.get_action(observation)
    next_observation, reward, terminated, truncated, info = env_train.step(action)

    agent.update(observation, action, reward, terminated, next_observation)
    observation = next_observation

    if terminated or truncated:
        observation, info = env_train.reset()

    if s % OFFSCREEN_STEPS == 0:
        vis_obs, info = env.reset(seed=42)
        terminated_or_truncated = False
        while not terminated_or_truncated:
            action = agent.get_action(vis_obs)
            vis_obs, reward, terminated, truncated, info = env.step(action)
            terminated_or_truncated = terminated or truncated

agent.save("testagent")
env_train.close()
env.close()
