import gymnasium as gym
from FreddyLanderAgent import FreddyLanderAgent
from tqdm import tqdm

env = gym.make("LunarLander-v3", render_mode="human")

agent = FreddyLanderAgent()

observation, info = env.reset(seed=42)

for _ in tqdm(range(10000)):
    action = agent.get_action(observation)
    next_observation, reward, terminated, truncated, info = env.step(action)

    agent.update(observation, action, reward, terminated, next_observation)
    observation = next_observation

    if terminated or truncated:
        observation, info = env.reset()

agent.save("testagent")
env.close()
