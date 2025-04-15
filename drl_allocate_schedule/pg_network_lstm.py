import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rl_utils

class LSTMPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(LSTMPolicyNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 输入x的形状应为 [batch_size, seq_len, input_dim]
        # 对于环境观察结果，我们需要重新组织维度
        batch_size = x.shape[0]

        # 确保输入是3D的
        if len(x.shape) == 4:  # 如果是图像格式 [batch, channel, height, width]
            # 将其转换为序列格式 [batch, height, width*channel]
            x = x.view(batch_size, x.shape[2], -1)

        # 运行LSTM
        output, (h_n, _) = self.lstm(x)

        # 使用最后一个时间步的隐藏状态
        return F.softmax(self.fc(h_n.squeeze(0)), dim=1)

    def take_action(self, state):
        # 处理单个状态的情况
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.asarray([state]), dtype=torch.float)
        if len(state.shape) == 3:  # [channel, height, width]
            state = state.unsqueeze(0)  # 添加batch维度

        state = state.to(next(self.parameters()).device)

        # 确保输入是3D的 [batch, seq, feature]
        if len(state.shape) == 4:  # 如果是图像格式 [batch, channel, height, width]
            state = state.view(state.shape[0], state.shape[2], -1)

        with torch.no_grad():
            probs = self.forward(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


class LSTMValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMValueNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 输入x的形状应为 [batch_size, seq_len, input_dim]
        batch_size = x.shape[0]

        # 确保输入是3D的
        if len(x.shape) == 4:  # 如果是图像格式 [batch, channel, height, width]
            # 将其转换为序列格式 [batch, height, width*channel]
            x = x.view(batch_size, x.shape[2], -1)

        # 运行LSTM
        output, (h_n, _) = self.lstm(x)

        # 使用最后一个时间步的隐藏状态
        return self.fc(h_n.squeeze(0))


class LSTM_PPO:
    '''使用LSTM网络的PPO算法实现'''

    def __init__(self, input_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = LSTMPolicyNet(input_dim, hidden_dim, action_dim).to(device)
        self.critic = LSTMValueNet(input_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        return self.actor.take_action(state)

    def update(self, transition_dict):
        # 将状态转换为张量
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

        # 确保输入是3D的 [batch, seq, feature]
        if len(states.shape) == 4:  # 如果是图像格式 [batch, channel, height, width]
            batch_size = states.shape[0]
            states = states.view(batch_size, states.shape[2], -1)
            next_states = next_states.view(batch_size, next_states.shape[2], -1)

        # 计算目标值和优势函数
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # PPO更新
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))

            # 更新网络
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save_data(self, pg_resume):
        torch.save(self.actor.state_dict(), pg_resume + '_actor' + '.pth')
        torch.save(self.critic.state_dict(), pg_resume + '_critic' + '.pth')


class AdaptiveLSTMPolicyNet(nn.Module):
    def __init__(self, hidden_dim, action_dim, fixed_width=350):
        super(AdaptiveLSTMPolicyNet, self).__init__()
        self.fixed_width = fixed_width
        self.hidden_dim = hidden_dim

        # 自适应池化层 - 将任意宽度转换为固定宽度
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, fixed_width))

        # LSTM层
        self.lstm = nn.LSTM(input_size=fixed_width, hidden_size=hidden_dim, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 处理输入 x 的形状可能是 [batch_size, channel, height, width]
        batch_size = x.shape[0]

        if len(x.shape) == 4:  # 图像格式 [batch, channel, height, width]
            # 使用自适应池化将宽度调整为固定大小
            # 输出形状: [batch, channel, height, fixed_width]
            x = self.adaptive_pool(x)

            # 重塑为LSTM的输入格式 [batch, height, fixed_width]
            x = x.squeeze(1)  # 假设channel=1，变成 [batch, height, fixed_width]
            if len(x.shape) == 2:  # 如果height=1，添加维度
                x = x.unsqueeze(1)

            # 应用LSTM
            _, (h_n, _) = self.lstm(x)

            # 使用最后一个隐藏状态
            return F.softmax(self.fc(h_n.squeeze(0)), dim=1)
        else:
            # 处理其他格式的输入
            x = x.view(batch_size, -1, self.fixed_width)
            _, (h_n, _) = self.lstm(x)
            return F.softmax(self.fc(h_n.squeeze(0)), dim=1)

    def take_action(self, state):
        # 处理单个状态
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.asarray([state]), dtype=torch.float)
        if len(state.shape) == 3:  # [channel, height, width]
            state = state.unsqueeze(0)  # 添加batch维度

        state = state.to(next(self.parameters()).device)

        with torch.no_grad():
            probs = self.forward(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


class Adaptive_LSTM_PPO:
    '''使用自适应池化LSTM网络的PPO算法实现'''

    def __init__(self, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, fixed_width=350):
        self.actor = AdaptiveLSTMPolicyNet(hidden_dim, action_dim, fixed_width).to(device)
        self.critic = AdaptiveLSTMPolicyNet(hidden_dim, 1, fixed_width).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        return self.actor.take_action(state)

    def update(self, transition_dict):
        # 将状态转换为张量
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

        # 计算目标值和优势函数
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # PPO更新
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))

            # 更新网络
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save_data(self, pg_resume):
        torch.save(self.actor.state_dict(), pg_resume + '_actor' + '.pth')
        torch.save(self.critic.state_dict(), pg_resume + '_critic' + '.pth')