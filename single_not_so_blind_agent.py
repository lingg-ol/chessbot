import pickle
import numpy as np
import collections # For dequeue for the memory buffer
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


LEARNING_RATE = 0.002
LEARNING_BATCH_SIZE = 10
MOMENTUM = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemoryBuffer(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.trans_counter=0 # num of transitions in the memory
                             # this count is required to delay learning
                             # until the buffer is sensibly full
        self.index=0         # current pointer in the buffer
        self.buffer = collections.deque(maxlen=self.memory_size)
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])

    
    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size # should begin sampling only when sufficiently full
        transitions = random.sample(self.buffer, k=batch_size) # number of transitions to sample

        states = np.array([e.state for e in transitions if e is not None]).astype(float)
        actions = np.array([e.action for e in transitions if e is not None]).astype(int)
        rewards = np.array([e.reward for e in transitions if e is not None]).astype(int)
        new_states = np.array([e.new_state for e in transitions if e is not None]).astype(float)
        terminals = np.array([e.terminal for e in transitions if e is not None]).astype(int)
  
        return states, actions, rewards, new_states, terminals

    
class Agent(object):
    def __init__(self, lr, gamma, epsilon, batch_size,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000):
        self.gamma = gamma # alpha = learn rate, gamma = discount
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec # decrement of epsilon for larger spaces
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory = MemoryBuffer(mem_size)


    def save(self, state, action, reward, new_state, done):
        # self.memory.trans_counter += 1
        self.memory.save(state, action, reward, new_state, done)


    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand > self.epsilon: 
            # greedy, returning best known action
            d_state = torch.from_numpy(state).to(device=device, dtype=torch.float32)
            sa = self.q_func(d_state) # type: ignore
            return sa.argmax().cpu().numpy()
        else:
            # exploring: return a random action
            return np.random.choice([i for i in range(4)])
            

    def reduce_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min
        
        
    def learn(self) -> None:
        raise Exception("Not implemented")
        

    def save_model(self, path):
        self.q_func.save(path) # type: ignore
        with open(path + '.memory.pickle', 'wb') as handle:
            pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_saved_model(self, path):
        pass
#        self.q_func = load_model(path)
#        with open(path + '.memory.pickle', 'rb') as handle:
#            self.memory = pickle.load(handle)
        
    
class SingleQAgent(Agent):
    def __init__(self, lr, gamma, epsilon, batch_size,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000):
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
                 epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
                 mem_size=mem_size)

        net = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 4))

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM) # type: ignore
        self.q_func = net.to(device)
        
        
    def learn(self):
        if self.memory.trans_counter < self.batch_size: # wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        d_states = torch.from_numpy(states).to(device=device, dtype=torch.float32)
                
        # 2. Compute predicted q value for the sample states
        q = self.q_func(d_states)
        
        # 3. Compute (using the same Q network) q value for the new states
        q_next = self.q_func(torch.from_numpy(new_states).to(device=device, dtype=torch.float32))
        
        # 4. Improve the Q network
        inx = torch.range(0, self.batch_size - 1, dtype=torch.int32, device=device)

        d_rewards = torch.from_numpy(rewards).to(device=device, dtype=torch.float32)
        d_terminals = torch.from_numpy(terminals).to(device=device, dtype=torch.float32)
        q[inx, actions] = d_rewards + (torch.max(q_next, dim=1).values * 1-d_terminals) * self.gamma

        # self.q_func.fit(states, q, verbose=0)
        self.optimizer.zero_grad()

        outputs = self.q_func(d_states)
        loss = self.loss(outputs, q)
        loss.backward()
        self.optimizer.step()

        # 5. Reduce the exploration rate
        self.reduce_epsilon()
