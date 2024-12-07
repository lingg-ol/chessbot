from copy import deepcopy
import numpy as np
import torch
from torch import nn
from typing import Any, Optional, TypedDict, Union

from util.ReplayBuffer import ReplayBuffer

class SimpleDQNAgent:
    def __init__(self,
                device: torch.device,
                q_net: nn.Module,
                target_q_net: nn.Module,
                replay_buffer: ReplayBuffer,
                batch_size: int = 128,
                steps_per_training: int = 100,
                q_target_lag: float = 0.1,
                discount_factor: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995,
                learning_rate: float = 0.001
    ):
        self.device = device
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.random = np.random.RandomState()
        self.buffer = replay_buffer

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.steps_per_training = steps_per_training
        self.q_target_lag = q_target_lag
        self.discount_factor = discount_factor

        self.train_counter = 0
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def get_action(self, observation):
        observation_tensor: torch.Tensor = observation if isinstance(observation, torch.Tensor) else torch.from_numpy(observation)
        observation_tensor = observation_tensor.to(device=self.device, dtype=torch.float32).unsqueeze(0)

        self.q_net.eval()
        with torch.no_grad():
            action_values = self.q_net(observation_tensor).cpu().numpy()
        self.q_net.train()

        action = 0
        if np.random.random() > self.epsilon:
            action = np.argmax(action_values)
        else:
            action = np.random.choice([i for i in range(4)])
        

        return action

    def observe(self, state, action, terminated, reward, next_state):
        self.buffer.save(state, action, terminated, reward, next_state)

    def train(self):
        if self.buffer.size() >= self.batch_size:
            for _ in range(self.steps_per_training):
                self.train_counter = self.train_counter + 1

                # sample buffer
                states, actions, terminals, rewards, next_states = self.buffer.sample(batch_size=self.batch_size)

                # calculate q targets
                next_q = self.target_q_net(next_states).detach().max(1)[0].unsqueeze(1)
                q_targets = rewards + self.discount_factor * next_q * (1 - terminals)

                # fit network
                outputs = self.q_net(states).gather(1, actions)
                loss = nn.functional.mse_loss(outputs, q_targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                if self.train_counter % 25 == 0:
                    for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                        target_param.data.copy_(local_param.data)