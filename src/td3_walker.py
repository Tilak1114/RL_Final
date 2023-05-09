import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


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
        x = torch.tanh(self.linear3(x))
        return x


BATCH_SIZE = 128
MAX_BUFFER = 200000
MIN_BUFFER = 10000
GAMMA = 0.98
TAU = 1e-2
POLICY_NOISE = 0.2
POLICY_UPDATE_FREQ = 2
EPISODES = 10000


class Agent():
    def __init__(self, env, tau):
        self.batch_size = BATCH_SIZE
        self.memory = list()
        self.max_buffer = MAX_BUFFER
        self.min_buffer = MIN_BUFFER

        state_size = np.prod(env.observation_space.shape)
        action_size = env.action_space.shape[0]

        self.tau = tau

        self.hidden_size = 500

        self.actor = Actor(state_size, self.hidden_size, action_size).to(device)
        self.actor_target = Actor(state_size, self.hidden_size, action_size).to(device)

        self.critic1 = Critic(state_size + action_size, self.hidden_size, action_size).to(device)
        self.critic2 = Critic(state_size + action_size, self.hidden_size, action_size).to(device)

        self.critic1_target = Critic(state_size + action_size, self.hidden_size, action_size).to(device)
        self.critic2_target = Critic(state_size + action_size, self.hidden_size, action_size).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

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
        # action = action.detach().numpy()[0, 0]
        action = action.detach().numpy()[0]
        noise = np.random.normal(0, 0.1, len(action))
        action += noise
        return np.clip(action, -1.0, 1.0)

    def train(self, current_episode):
        if len(self.memory) < self.min_buffer:
            return

        indices = np.random.choice(len(self.memory), self.batch_size, replace=True)
        batch = [self.memory[idx] for idx in indices]

        states = torch.tensor([b["current_state"] for b in batch], dtype=torch.float32)
        actions = torch.tensor([b["action"] for b in batch], dtype=torch.float32)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
        next_states = torch.tensor([b["next_state"] for b in batch], dtype=torch.float32)
        terminals = torch.tensor([b["terminal"] for b in batch], dtype=torch.bool)

        next_actions = agent.actor_target.forward(next_states)
        noise = np.random.normal(0, 0.1, tuple(next_actions.shape))
        noise = torch.tensor(noise)
        next_actions += noise
        next_actions = torch.clamp(next_actions, -1.0, 1.0)

        with torch.no_grad():
            next_Q1 = agent.critic1_target.forward(next_states, next_actions.detach())
            next_Q2 = agent.critic2_target.forward(next_states, next_actions.detach())

            next_Q = torch.min(next_Q2, next_Q1)
            targets = rewards.unsqueeze(1).expand(-1, next_Q.shape[1]) + GAMMA * next_Q * (1-terminals.int().unsqueeze(1))

        # print(target, Qvals)

        Qvals1 = agent.critic1.forward(states, actions)
        Qvals2 = agent.critic2.forward(states, actions)
        critic_loss1 = nn.MSELoss()(Qvals1.view(-1), targets.view(-1))
        critic_loss2 = nn.MSELoss()(Qvals2.view(-1), targets.view(-1))

        agent.critic1.optimizer.zero_grad()
        critic_loss1.backward()
        agent.critic1.optimizer.step()

        agent.critic2.optimizer.zero_grad()
        critic_loss2.backward()
        agent.critic2.optimizer.step()

        if current_episode % POLICY_UPDATE_FREQ == 0:
            policy_loss = -agent.critic1.forward(states, agent.actor.forward(states)).mean()

            agent.actor.optimizer.zero_grad()
            policy_loss.backward()
            agent.actor.optimizer.step()

            # slow target updation
            for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for target_param, param in zip(agent.critic1_target.parameters(), agent.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for target_param, param in zip(agent.critic2_target.parameters(), agent.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


env = gym.make("BipedalWalker-v3")

scores = []

agent = Agent(env, TAU)

for e in range(EPISODES):
    state, _ = env.reset()
    terminated, truncated = False, False
    score = 0

    while not terminated and not truncated:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.update_buffer(state, action, reward, next_state, terminated)
        agent.train(e)
        score += reward
        state = next_state

    if e % 10 == 0:
        print("Episode " + str(e) + "/" + str(EPISODES) + ", Score: " + str(score))

    scores.append(score)

plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()
