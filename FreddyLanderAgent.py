import numpy as np
import random
import torch
from torch import nn
import os


class FreddyPolicy(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(n_inputs + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.relu_stack(x)


class FreddyLanderAgent:
    def __init__(self, random_samples: int = 1000, batch_size: int = 100, max_samples: int = 1000, inference: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"freddy lander uses device {self.device}")
        self.policy = FreddyPolicy(8).to(self.device)
        self.sample_buffer = []
        self.buf_pos = 0
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.inference = inference
        self.random_samples = random_samples

        if not inference:
            self.loss = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.001, momentum=0.9)

    def get_action(self, observation) -> int:
        # if not in inference mode, we gather random_samples random samples before starting to use the policy
        if not self.inference and self.random_samples > 0:
            self.random_samples = self.random_samples - 1
            return random.choice([0, 1, 2, 3])

        with torch.no_grad():
            qualities = [self.policy(torch.from_numpy(np.array([*observation, float(a)])).to(device=self.device, dtype=torch.float32)).numpy() for a in range(4)]

        if self.inference:
            return np.argmax(qualities)

        # softmax on qualities
        total = np.sum(qualities)
        qualities = [q / total for q in qualities]

        # stochastic action selection
        accu: float = float(0)
        r: float = random.random()
        for action, quality in enumerate(qualities):
            accu = accu + quality
            if accu >= r:
                return action

        return 3 #should never reach that line, but just in case, we choose the last action

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
            self.sample_buffer.append([*observation, float(action), reward])
        else:
            self.sample_buffer[self.buf_pos] = [*observation, float(action), reward]
        self.buf_pos = (self.buf_pos + 1) % self.max_samples

        # train on random samples from sample buffer
        if self.random_samples <= 0 and num_samples > 2:
            samples = random.choices(self.sample_buffer, k=min(self.batch_size, num_samples))
            np_samples = np.array(samples)

            tensor_inputs = torch.stack([torch.from_numpy(i[:-1]) for i in np_samples]).to(device=self.device, dtype=torch.float32)
            tensor_targets = torch.from_numpy(np.array([i[-1] for i in np_samples])).to(device=self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            outputs = self.policy(tensor_inputs)
            loss = self.loss(outputs, tensor_targets)
            loss.backward()
            self.optimizer.step()

            #print(loss.item())

    def save(self, path: os.PathLike):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: os.PathLike):
        self.policy.load_state_dict(path)
