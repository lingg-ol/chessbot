import numpy as np
import torch
import gymnasium as gym
import tqdm
from util.ReplayBuffer import ReplayBuffer
from util.SequentialNet import SequentialNet
from SimpleDQNAgent import SimpleDQNAgent

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

buffer: ReplayBuffer = ReplayBuffer(device, 200000)
q_net: SequentialNet = SequentialNet(8, 4)
q_net = q_net.to(device=device)
target_q_net: SequentialNet = SequentialNet(8, 4)
target_q_net = target_q_net.to(device=device)

agent: SimpleDQNAgent = SimpleDQNAgent(device, q_net, target_q_net, buffer, steps_per_training=1)

env_train = gym.make("LunarLander-v3")
env = gym.make("LunarLander-v3", render_mode="human")

vis_every = 100

for i in tqdm.tqdm(range(500)):
    # perform a landing and then train
    obs, info = env_train.reset()

    terminated_or_truncated = False
    step = 0
    while not terminated_or_truncated:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env_train.step(action)
        agent.observe(obs, action, terminated, reward, next_obs)
        obs = next_obs
        step = step + 1
        if step % 4 == 0:
            agent.train()

        terminated_or_truncated = terminated or truncated
    
    if (i + 1) % vis_every == 0:
        obs, _ = env.reset()
        terminated_or_truncated = False
        while not terminated_or_truncated:
            action = agent.get_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            terminated_or_truncated = terminated or truncated

env_train.close()

while True:
    obs, _ = env.reset()
    terminated_or_truncated = False
    while not terminated_or_truncated:
        action = agent.get_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        terminated_or_truncated = terminated or truncated

env.close()