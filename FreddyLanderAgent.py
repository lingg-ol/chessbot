import math
import numpy as np
import random
import torch
from torch import nn
from torch.nn import init
import os

# obs_n, reward_n, action_n, obs_n+1
# predict(obs_n, action_n) = reward_n + alpha * max([predict(obs_n+1, a) for a in actions])

def _init_random_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class FreddyPolicy(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(n_inputs + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.relu_stack.apply(_init_random_weights)

    def forward(self, x):
        return torch.squeeze(self.relu_stack(x), dim=-1)


def _safe_exp(x):
    try:
        return min(100000, math.exp(x))
    except OverflowError:
        return 100000


class FreddyLanderAgent:
    def __init__(self, explore_alpha: float = 0.999, min_explore_alpha: float = 0.5, discount_factor: float = 0.3, random_samples: int = 1000, batch_size: int = 300, max_samples: int = 15000, inference: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"freddy lander uses device {self.device}")
        self.policy = FreddyPolicy(8).to(self.device)
        self.sample_buffer = []
        self.buf_pos = 0
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.inference = inference
        self.random_samples = random_samples
        self.discount_factor = discount_factor
        self.explore_alpha = explore_alpha
        self.initial_explore_alpha = explore_alpha
        self.min_explore_alpha = min_explore_alpha

        if not inference:
            self.policy.train(True)
            self.loss = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.001, momentum=0.9)

    def get_action(self, observation) -> int:
        # if not in inference mode, we gather random_samples random samples before starting to use the policy
        if not self.inference and self.random_samples > 0:
            self.random_samples = self.random_samples - 1
            return random.choice([0, 1, 2, 3])

        with torch.no_grad():
            qualities = [self.policy(torch.from_numpy(np.array([*observation, float(a)])).to(device=self.device, dtype=torch.float32)).to(device="cpu").numpy() for a in range(4)]

        action = int(np.argmax(qualities))
        if self.inference:
            return action

        # softmax on qualities
        qualities = [np.exp(q) for q in qualities]
        total = max(np.sum(qualities), 0.001)
        qualities = [q / total for q in qualities]

        # interpolate between one hot and softmax qualities depending on explore alpha
        #qualities = [q * self.explore_alpha + (1 - self.explore_alpha) * (1 if a == action else 0) for a, q in enumerate(qualities)]
        #self.explore_alpha = max(self.explore_alpha * self.initial_explore_alpha, self.min_explore_alpha)

        # stochastic action selection
        accu: float = float(0)
        r: float = random.random()
        for action, quality in enumerate(qualities):
            accu = accu + quality
            if accu >= r:
                return action

        return 3 #should never reach that line, but just in case, we choose the last action

    def calculate_q_targets(self, samples):
        tensor_inputs = torch.stack([torch.stack([torch.from_numpy(np.array([*i[11:], float(a)])) for a in range(4)]) for i in samples]).to(device=self.device, dtype=torch.float32)
        tensor_terminals = torch.from_numpy(np.array([i[10] for i in samples])).to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.policy(tensor_inputs)
            predictions = torch.max(predictions, dim=1).values

            tensor_rewards = torch.from_numpy(np.array([i[9] for i in samples])).to(device=self.device, dtype=torch.float32)
            return torch.add(tensor_rewards, torch.mul(torch.mul(predictions, self.discount_factor), tensor_terminals))

    def update(
        self,
        observation,
        action: int,
        reward: float,
        terminated: bool,
        next_observation
    ):
        if self.inference:
            print("Agent is in inference mode and cannot update weights")
            return

        # insert sample into sample buffer
        num_samples = len(self.sample_buffer)
        if num_samples <= self.buf_pos:
            self.sample_buffer.append([*observation, float(action), reward, float(1) if terminated else float(0), *next_observation])
        else:
            self.sample_buffer[self.buf_pos] = [*observation, float(action), reward, float(1) if terminated else float(0), *next_observation]
        self.buf_pos = (self.buf_pos + 1) % self.max_samples

        # train on random samples from sample buffer
        if self.random_samples <= 0 and num_samples > 2:
            samples = random.choices(self.sample_buffer, k=min(self.batch_size, num_samples))
            np_samples = np.array(samples)

            tensor_inputs = torch.stack([torch.from_numpy(i[:9]) for i in np_samples]).to(device=self.device, dtype=torch.float32)
            tensor_targets = self.calculate_q_targets(np_samples)

            self.optimizer.zero_grad()
            outputs = self.policy(tensor_inputs)
            loss = self.loss(outputs, tensor_targets)
            loss.backward()
            self.optimizer.step()

    def save(self, path: os.PathLike):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: os.PathLike):
        self.policy.load_state_dict(torch.load(path))
