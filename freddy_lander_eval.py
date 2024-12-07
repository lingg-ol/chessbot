import gymnasium as gym
from FreddyLanderAgent import FreddyLanderAgent
from tqdm import tqdm

env = gym.make("LunarLander-v3", render_mode="human")
agent = FreddyLanderAgent(inference=True)
agent.load("testagent")

observation, info = env.reset(seed=42)
for _ in tqdm(range(1000)):
    action = agent.get_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
