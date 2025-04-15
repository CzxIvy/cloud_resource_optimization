import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sympy import print_tree

import rl_utils

class PolicyNet(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.fc2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc3 = nn.AdaptiveAvgPool2d((10, 175))
        self.fc4 = torch.nn.Flatten()
        self.fc5 = torch.nn.Linear(10500, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(self.fc5(x))
        return F.softmax(self.fc6(x), dim=1)

class ValueNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.fc2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc3 = nn.AdaptiveAvgPool2d((10, 175))
        self.fc4 = torch.nn.Flatten()
        self.fc5 = torch.nn.Linear(10500, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(self.fc5(x))
        return self.fc6(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(hidden_dim, action_dim).to(device)
        self.critic = ValueNet(hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.asarray([state]), dtype=torch.float).to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.asarray(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.asarray(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save_data(self, pg_resume):
        torch.save(self.actor.state_dict(), pg_resume + '_actor' + '.pth')
        torch.save(self.critic.state_dict(), pg_resume + '_critic' + '.pth')

class GRPO:
    ''' GRPO算法, 采用组内相对奖励方式 '''
    def __init__(self, hidden_dim, action_dim, actor_lr, gamma, epochs, eps, device):
        self.actor = PolicyNet(hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.epochs = epochs  # 数据用来训练的轮数
        self.eps = eps  # GRPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.asarray([state]), dtype=torch.float).to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.asarray(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * rewards
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # GRPO的损失函数

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

    def save_data(self, pg_resume):
        torch.save(self.actor.state_dict(), pg_resume + '_actor' + '.pth')


class AllocateActor(nn.Module):
    def __init__(self, hidden_dim, action_dim, lr, device):
        super(AllocateActor, self).__init__()

        self.fc1 = torch.nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.fc2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc3 = nn.AdaptiveAvgPool2d((10, 175))
        self.fc4 = torch.nn.Flatten()
        self.fc5 = torch.nn.Linear(10500, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        self.device = device

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(self.fc5(x))
        return F.softmax(self.fc6(x), dim=1)

    def take_action(self, state):
        state = torch.tensor(np.asarray([state]), dtype=torch.float).to(self.device)
        self.eval()
        with torch.no_grad():
            probs = self.forward(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        self.train()
        states = torch.tensor(np.asarray(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        log_probs = torch.log(self.forward(states).gather(1, actions))
        loss = torch.mean(-(log_probs * rewards))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_data(self, pg_resume):
        torch.save(self.state_dict(), pg_resume + '.pth')

    def load_data(self, pg_resume):
        self.load_state_dict(torch.load(pg_resume))
        self.eval()  # Set the model to evaluation mode