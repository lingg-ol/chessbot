from collections import deque, namedtuple
import gymnasium as gym
from typing import Any, List, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
    
OBSERVATION_SIZE = 8
ACTION_SIZE = 1
ACTION_VALUES = range(4)
MEMORY_SIZE = 100000
RANDOM_SAMPLE_SIZE = 1000
LEARNING_RATE = 0.002
LEARNING_BATCH_SIZE = 10
MOMENTUM = 0.9
ACTION_QUALITY_BASE = 1.3
NUM_NN_LAYER = 3
NUM_NN_NEURONS_IN_LAYER = 32
NEXT_REWARD_INFLUENCE = 0.9
ACCEL_BONUS_FACTOR = [-0.1, -0.5, -0.25]

class BlindLanderPolicy(nn.Module):

    def __init__(self, obs_size, action_size):
        super(BlindLanderPolicy, self).__init__()
        
        layers: List[nn.Module] = [nn.Linear(obs_size + action_size, NUM_NN_NEURONS_IN_LAYER)]
        for _ in range(NUM_NN_LAYER - 2):
            layers.extend([nn.ReLU(), nn.Linear(NUM_NN_NEURONS_IN_LAYER, NUM_NN_NEURONS_IN_LAYER)])
        layers.append(nn.Linear(NUM_NN_NEURONS_IN_LAYER, action_size))
        
        self.net = nn.Sequential(*layers)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return torch.squeeze(self.net(x), dim = -1)

class BlindLanderAgent:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Blindness is proodly provided by {self.device}")
        
        self.policy = BlindLanderPolicy(OBSERVATION_SIZE, ACTION_SIZE).to(self.device)
        self.memory = deque([], maxlen = MEMORY_SIZE)
        self.inference = False
        self.random_samples = RANDOM_SAMPLE_SIZE
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM) # type: ignore
    
    def enable_inference(self):
        self.inference = True

    def get_action(self, obs) -> int:
        if not self.inference and self.random_samples > 0:
            self.random_samples = self.random_samples - 1
            return random.choice(ACTION_VALUES)
        
        # estimate qualities of all possible actions
        with torch.no_grad():
            estimated_action_qualities = [self.policy(torch.from_numpy(np.array([*obs, float(a)])).to(device=self.device, dtype=torch.float32)).to(device="cpu").numpy() for a in ACTION_VALUES]

        return self.select_action_by_quality(estimated_action_qualities)

    def select_action_by_quality(self, qualities):
        qualities = [ACTION_QUALITY_BASE ** q for q in qualities]
        total = sum(qualities)
        qualities = [q / total for q in qualities]

        rand = random.random()
        acc = 0
        for action, quality in enumerate(qualities):
            acc = acc + quality
            if rand <= acc:
                return action
        
        return len(qualities) - 1
    
    def update(
        self,
        obs,
        action: float,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        self.memory.append([*obs, float(action), *next_obs, reward])

        # train on random samples from sample buffer
        if self.random_samples <= 0 and len(self.memory) > 2:
            samples = np.array(random.sample(self.memory, LEARNING_BATCH_SIZE))

            with torch.no_grad():
                estimated_action_qualities = [self.policy(torch.from_numpy(np.array([*next_obs, float(a)])).to(device=self.device, dtype=torch.float32)).to(device="cpu").numpy() for a in ACTION_VALUES]

            next_reward = np.mean(estimated_action_qualities)
            accel_bonus = ACCEL_BONUS_FACTOR[0] * abs(next_obs[2]) + ACCEL_BONUS_FACTOR[1] * abs(next_obs[3]) + ACCEL_BONUS_FACTOR[2] * abs(next_obs[5])

            tensor_inputs = torch.stack([torch.from_numpy(i[:-9]) for i in samples]).to(device=self.device, dtype=torch.float32)
            tensor_targets = torch.from_numpy(np.array([self.get_target_reward(i[-1], next_reward, accel_bonus) for i in samples])).to(device=self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            outputs = self.policy(tensor_inputs)
            loss = self.loss(outputs, tensor_targets)
            loss.backward()
            self.optimizer.step()

    def get_target_reward(self, reward, next_reward, bonus):
        return reward + (next_reward - reward) * NEXT_REWARD_INFLUENCE + bonus
