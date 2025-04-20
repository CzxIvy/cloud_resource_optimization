import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import rl_utils

def create_mask(input_tensor, pad_value=-1):
    # 输入形状: (batch_size, channels, height, width)
    # 生成掩码：非填充位置为1，填充位置为0（按通道维度取"或"）
    mask = (input_tensor != pad_value).any(dim=1, keepdim=True)  # (B, 1, H, W)
    return mask.float()


class MaskedSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 通道压缩 + 空间注意力权重生成
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),  # 压缩到单通道
            nn.Sigmoid()  # 输出注意力权重 [0,1]
        )

    def forward(self, x, mask):
        # x形状: (B, C, H, W)
        # mask形状: (B, 1, H, W)

        # Step 1: 生成空间注意力权重
        attention = self.conv(x)  # (B, 1, H, W)

        # Step 2: 应用掩码，将填充区域的注意力权重置零
        masked_attention = attention * mask

        # Step 3: 对特征图加权
        weighted_x = x * masked_attention

        return weighted_x

class PolicyNetWithAttention(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(PolicyNetWithAttention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        
        self.attention = MaskedSpatialAttention(in_channels=6)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5880, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        mask = create_mask(x)  # (B, 1, H, W)
        features = self.cnn(x)
        # 下采样掩码以匹配特征图尺寸
        mask_downsampled = torch.nn.functional.adaptive_avg_pool2d(
            mask, (10, 98)
        )
        attended_features = self.attention(features, mask_downsampled)
        output = self.fc(attended_features)
        return output
    
class ValueNetWithAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ValueNetWithAttention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # 空间注意力模块
        self.attention = MaskedSpatialAttention(in_channels=6)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5880, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        mask = create_mask(x)  # (B, 1, H, W)
        features = self.cnn(x)
        # 下采样掩码以匹配特征图尺寸
        mask_downsampled = torch.nn.functional.adaptive_avg_pool2d(
            mask, (10, 98)
        )
        attended_features = self.attention(features, mask_downsampled)
        output = self.fc(attended_features)
        return output


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetWithAttention(hidden_dim, action_dim).to(device)
        self.critic = ValueNetWithAttention(hidden_dim).to(device)
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

class GRPOForSchedule:
    ''' GRPO算法, 采用组内相对奖励方式 '''
    def __init__(self, hidden_dim, action_dim, actor_lr, gamma, epochs, eps, device):
        self.actor = PolicyNetWithAttention(hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.epochs = epochs  # 数据用来训练的轮数
        self.eps = eps  # GRPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state_ = torch.tensor(np.asarray([state]), dtype=torch.float).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_)
        try:
            action_dist = torch.distributions.Categorical(probs)
        except Exception as e:
            print(f"Error in action distribution: {e}")
            print(f"state: {state}")
            print(f"Probs: {probs}")
            np.savetxt('error_state.txt', np.asarray(state[0]), fmt='%.6f', delimiter=',')
            raise e
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
        torch.save(self.actor.state_dict(), pg_resume + '_schedule_actor' + '.pth')
        
    def load_data(self, pg_resume):
        self.actor.load_state_dict(torch.load(pg_resume))
        self.actor.eval()


class GPROForAllocate:
    def __init__(self, hidden_dim, action_dim, actor_lr, gamma, epochs, eps, device):
        self.actor = PolicyNetWithAttention(hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state_ = torch.tensor(np.asarray([state]), dtype=torch.float).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_)
        try:
            action_dist = torch.distributions.Categorical(probs)
        except Exception as e:
            print(f"Error in action distribution: {e}")
            print(f"state: {state}")
            print(f"Probs: {probs}")
            np.savetxt('error_state.txt', np.asarray(state[0]), fmt='%.6f', delimiter=',')
            raise e
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
        torch.save(self.actor.state_dict(), pg_resume + '_allocate_actor' + '.pth')

    def load_data(self, pg_resume):
        self.actor.load_state_dict(torch.load(pg_resume))
        self.actor.eval()