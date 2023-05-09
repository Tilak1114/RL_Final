import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import gym


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.learning_rate = learning_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x


BATCH_SIZE = 128
MAX_BUFFER = 200000
MIN_BUFFER = 5000
GAMMA = 0.98
TAU = 1e-2
EPISODES = 25000


# class Env():
#   def __init__(self, obs_space, action_space):
#     self.obs_space =

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Agent():
    def __init__(self, env, tau):
        self.batch_size = BATCH_SIZE
        self.memory = list()
        self.max_buffer = MAX_BUFFER
        self.min_buffer = MIN_BUFFER

        state_size = np.prod(env.observation_space.shape)
        action_size = env.action_space.shape[0]

        self.tau = tau

        self.hidden_size = 256

        self.actor = Actor(state_size, self.hidden_size, action_size)
        self.actor_target = Actor(state_size, self.hidden_size, action_size)
        self.critic = Critic(state_size + action_size, self.hidden_size, action_size)
        self.critic_target = Critic(state_size + action_size, self.hidden_size, action_size)

    def update_buffer(self, current_state, action, reward, next_state,
                      is_terminal):
        self.memory.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "terminal": is_terminal
        })

        if len(self.memory) > self.max_buffer:
            self.memory.pop(0)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0, 0]
        return action

    def train(self):
        if len(self.memory) < self.min_buffer:
            return

        indices = np.random.choice(len(self.memory), self.batch_size, replace=True)
        batch = [self.memory[idx] for idx in indices]

        states = torch.tensor([b["current_state"] for b in batch], dtype=torch.float32)
        actions = torch.tensor([b["action"] for b in batch], dtype=torch.float32)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
        next_states = torch.tensor([b["next_state"] for b in batch], dtype=torch.float32)
        terminals = torch.tensor([b["terminal"] for b in batch], dtype=torch.bool)

        Qvals = agent.critic.forward(states, actions)
        next_actions = agent.actor_target.forward(next_states)
        next_Q = agent.critic_target.forward(next_states, next_actions.detach())

        targets = rewards.unsqueeze(1) + GAMMA * next_Q

        # print(target, Qvals)

        critic_loss = nn.MSELoss()(Qvals.view(-1), targets.view(-1))
        agent.critic.optimizer.zero_grad()
        critic_loss.backward()
        agent.critic.optimizer.step()

        policy_loss = -agent.critic.forward(states, agent.actor.forward(states)).mean()

        agent.actor.optimizer.zero_grad()
        policy_loss.backward()
        agent.actor.optimizer.step()

        # slow target updation
        for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


env = gym.make("BipedalWalker-v3")

noise = OUNoise(env.action_space)

scores = []

agent = Agent(env, TAU)

for e in range(EPISODES):
    state, _ = env.reset()
    noise.reset()
    terminated, truncated = False, False
    score = 0

    while not terminated and not truncated:
        action = agent.get_action(state)
        action = noise.get_action(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.update_buffer(state, action, reward, next_state, terminated)
        agent.train()
        score += reward
        state = next_state

    if e % 10 == 0:
        print("Episode " + str(e) + "/" + str(EPISODES) + ", Score: " + str(score))

    scores.append(score)

plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()
