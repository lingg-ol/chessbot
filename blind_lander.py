import gymnasium as gym
from numpy import mod
from tqdm import tqdm
from blind_lander_agent import BlindLanderAgent
from not_so_blind_lander_agent import DoubleQAgent
from single_not_so_blind_agent import SingleQAgent

env_on_screen = gym.make("LunarLander-v3", render_mode="human")
env_off_screen = gym.make("LunarLander-v3")
env = env_on_screen;

# agent = BlindLanderAgent()
agent = DoubleQAgent(gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)
# agent = SingleQAgent(gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)

NUM_TRAINING_EPISODES = 500
LEARN_EVERY = 4

hidden_episodes = 10

for episode in tqdm(range(NUM_TRAINING_EPISODES)):

    env = env_on_screen if episode % hidden_episodes == 0 else env_off_screen

    observation, info = env.reset()
    terminated = truncated = False
    step = 0

    while not (terminated or truncated):
        # action = agent.get_action(observation)
        action = agent.choose_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        # agent.update(observation, action, float(reward), terminated, next_observation)
        agent.save(observation, action, reward, next_observation, terminated)
        observation = next_observation

        if ++step % LEARN_EVERY == 0:
            agent.learn()
    
# agent.enable_inference()

env = env_on_screen
observation, info = env.reset()

while True:
    # action = agent.get_action(observation)
    action = agent.choose_action(observation)
    next_observation, reward, terminated, truncated, info = env.step(action)
    observation = next_observation

    if terminated or truncated:
        observation, info = env.reset()

env_off_screen.close()
env_on_screen.close()