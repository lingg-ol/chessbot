import collections
import numpy as np
import torch
import random
from typing import Any, List, NamedTuple, Tuple

class Sample(NamedTuple):
    state: Any
    action: Any
    terminal: Any
    reward: Any
    next_state: Any

class ReplayBuffer:
    def __init__(self, device: torch.device, buffer_size: int):
        self.buffer_size = buffer_size
        self.device = device

        self.buffer = collections.deque(maxlen=buffer_size)

    def save(self, state, action, terminal, reward, next_state):
        self.buffer.append(Sample(state, action, terminal, reward, next_state))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        samples = random.sample(self.buffer, batch_size)
        states = torch.from_numpy(np.vstack([s.state for s in samples if s is not None])).float().to(device=self.device)
        actions = torch.from_numpy(np.vstack([s.action for s in samples if s is not None])).long().to(device=self.device)
        terminals = torch.from_numpy(np.vstack([s.terminal for s in samples if s is not None]).astype(np.uint8)).float().to(device=self.device)
        rewards = torch.from_numpy(np.vstack([s.reward for s in samples if s is not None])).float().to(device=self.device)
        next_states = torch.from_numpy(np.vstack([s.next_state for s in samples if s is not None])).float().to(device=self.device)

        return states, actions, terminals, rewards, next_states

    def size(self) -> int:
        return len(self.buffer)