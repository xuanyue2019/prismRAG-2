import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

class BaseAgent(nn.Module):
    """基础强化学习智能体类"""
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 hidden_size: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        """
        初始化基础智能体
        
        Args:
            state_size: 状态空间维度
            action_size: 动作空间大小
            hidden_size: 隐藏层维度
            num_layers: 网络层数
            dropout: Dropout 率
            use_layer_norm: 是否使用层归一化
        """
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # 状态编码器
        self.state_encoder = self._build_encoder(
            state_size, hidden_size, num_layers, dropout, use_layer_norm
        )
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _build_encoder(self, 
                      input_size: int, 
                      hidden_size: int,
                      num_layers: int,
                      dropout: float,
                      use_layer_norm: bool) -> nn.Module:
        """构建状态编码器"""
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 输入状态张量 [batch_size, state_size]
            
        Returns:
            action_probs: 动作概率分布 [batch_size, action_size]
            state_value: 状态价值估计 [batch_size, 1]
        """
        # 编码状态
        hidden = self.state_encoder(state)
        
        # 计算动作概率
        action_logits = self.policy_net(hidden)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 计算状态价值
        state_value = self.value_net(hidden)
        
        return action_probs, state_value
    
    def get_action(self, 
                  state: np.ndarray, 
                  deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        根据状态选择动作
        
        Args:
            state: 输入状态
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 选择的动作
            action_probs: 动作概率
            state_value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, state_value = self.forward(state_tensor)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
        else:
            action = torch.multinomial(action_probs, 1).item()
        
        return action, action_probs.squeeze(0), state_value.squeeze(0)
    
    def evaluate_actions(self, 
                        states: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于训练）
        
        Args:
            states: 状态批次 [batch_size, state_size]
            actions: 动作批次 [batch_size]
            
        Returns:
            action_log_probs: 动作对数概率 [batch_size]
            entropy: 策略熵 [batch_size]
            state_values: 状态价值 [batch_size, 1]
        """
        action_probs, state_values = self.forward(states)
        
        # 计算动作对数概率
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        
        # 计算策略熵
        entropy = dist.entropy()
        
        return action_log_probs, entropy, state_values
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_size': self.hidden_size
            }
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: str = 'cpu'):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['config']
        
        agent = cls(
            state_size=config['state_size'],
            action_size=config['action_size'],
            hidden_size=config.get('hidden_size', 512)
        )
        
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.to(device)
        
        return agent


class TextGameAgent(BaseAgent):
    """文本游戏专用智能体"""
    
    def __init__(self, 
                 embedding_size: int = 384,
                 action_size: int = 10,
                 hidden_size: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        初始化文本游戏智能体
        
        Args:
            embedding_size: 文本嵌入维度
            action_size: 动作数量
            hidden_size: 隐藏层大小
            num_layers: 网络层数
            dropout: Dropout 率
        """
        super().__init__(
            state_size=embedding_size,
            action_size=action_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 额外的文本处理层
        self.text_processor = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """重写前向传播以包含文本处理"""
        # 文本特征提取
        text_features = self.text_processor(state)
        
        # 编码状态
        hidden = self.state_encoder(text_features)
        
        # 计算动作概率
        action_logits = self.policy_net(hidden)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 计算状态价值
        state_value = self.value_net(hidden)
        
        return action_probs, state_value


class RecurrentAgent(BaseAgent):
    """循环神经网络智能体（用于序列数据）"""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        初始化循环智能体
        
        Args:
            state_size: 状态维度
            action_size: 动作数量
            hidden_size: 隐藏层大小
            num_layers: RNN 层数
            dropout: Dropout 率
        """
        super().__init__(state_size, action_size)
        
        # RNN 编码器
        self.rnn = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 策略和价值网络
        self.policy_net = nn.Linear(hidden_size, action_size)
        self.value_net = nn.Linear(hidden_size, 1)
        
        # 隐藏状态
        self.hidden_state = None
    
    def forward(self, 
               state: torch.Tensor,
               hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播（支持序列输入）
        
        Args:
            state: 输入状态 [batch_size, seq_len, state_size] 或 [batch_size, state_size]
            hidden_state: 之前的隐藏状态
            
        Returns:
            action_probs: 动作概率
            state_value: 状态价值
            new_hidden_state: 新的隐藏状态
        """
        if state.dim() == 2:
            # 单步输入，添加序列维度
            state = state.unsqueeze(1)
        
        # RNN 编码
        rnn_out, new_hidden_state = self.rnn(state, hidden_state)
        
        # 取最后一个时间步的输出
        last_output = rnn_out[:, -1, :]
        
        # 计算动作概率和价值
        action_logits = self.policy_net(last_output)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.value_net(last_output)
        
        return action_probs, state_value, new_hidden_state
    
    def reset_hidden_state(self, batch_size: int = 1):
        """重置隐藏状态"""
        device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device),
            torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
        )


# 工具函数
def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """创建指定类型的智能体"""
    agent_classes = {
        'base': BaseAgent,
        'text': TextGameAgent,
        'recurrent': RecurrentAgent
    }
    
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agent_classes.keys())}")
    
    return agent_classes[agent_type](**kwargs)


def compute_advantages(rewards: torch.Tensor,
                      values: torch.Tensor,
                      next_values: torch.Tensor,
                      dones: torch.Tensor,
                      gamma: float = 0.99,
                      gae_lambda: float = 0.95) -> torch.Tensor:
    """
    计算广义优势估计 (GAE)
    
    Args:
        rewards: 奖励序列
        values: 价值估计序列
        next_values: 下一个状态价值序列
        dones: 终止标志序列
        gamma: 折扣因子
        gae_lambda: GAE 参数
        
    Returns:
        advantages: 优势估计
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    # 反向计算 GAE
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1] * (1 - dones[t])
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * gae_lambda * last_advantage * (1 - dones[t])
        last_advantage = advantages[t]
    
    return advantages


if __name__ == "__main__":
    # 测试基础智能体
    agent = BaseAgent(state_size=10, action_size=5)
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters())}")
    
    # 测试前向传播
    test_state = torch.randn(1, 10)
    action_probs, value = agent(test_state)
    print(f"Action probs shape: {action_probs.shape}")
    print(f"Value shape: {value.shape}")
    
    # 测试动作选择
    action, probs, val = agent.get_action(test_state.numpy()[0])
    print(f"Selected action: {action}")