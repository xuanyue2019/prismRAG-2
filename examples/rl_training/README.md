# Agent Lightning 强化学习示例

本目录包含使用 Agent Lightning 进行强化学习的完整示例。

## 🎯 示例场景

### 1. 文本游戏智能体 (TextGameAgent)
- **环境**: 基于文本的冒险游戏
- **任务**: 探索环境、收集物品、完成任务
- **奖励**: 完成任务 +10，收集物品 +2，无效动作 -1

### 2. 对话助手 (DialogueAgent)  
- **环境**: 多轮对话系统
- **任务**: 提供有帮助、准确的回答
- **奖励**: 用户满意度 +5，相关回答 +3，错误回答 -2

### 3. 代码生成智能体 (CodeGenAgent)
- **环境**: 编程任务环境
- **任务**: 根据需求生成正确的代码
- **奖励**: 通过测试 +10，语法正确 +3，编译错误 -5

## 📁 文件结构

```
examples/rl_training/
├── README.md                    # 本文件
├── requirements.txt             # 训练额外依赖
├── config/
│   ├── text_game_config.yaml   # 文本游戏配置
│   ├── dialogue_config.yaml    # 对话助手配置
│   └── codegen_config.yaml     # 代码生成配置
├── environments/
│   ├── text_game_env.py        # 文本游戏环境
│   ├── dialogue_env.py         # 对话环境
│   └── codegen_env.py          # 代码生成环境
├── agents/
│   ├── base_agent.py           # 基础智能体类
│   ├── text_game_agent.py      # 文本游戏智能体
│   ├── dialogue_agent.py       # 对话智能体
│   └── codegen_agent.py        # 代码生成智能体
├── trainers/
│   ├── ppo_trainer.py          # PPO 训练器
│   ├── a2c_trainer.py          # A2C 训练器
│   └── reward_shaping.py       # 奖励塑造工具
├── scripts/
│   ├── train_text_game.py      # 文本游戏训练脚本
│   ├── train_dialogue.py       # 对话训练脚本
│   ├── train_codegen.py        # 代码生成训练脚本
│   └── evaluate_agent.py       # 智能体评估脚本
├── data/
│   ├── text_game_scenarios/    # 文本游戏场景数据
│   ├── dialogue_datasets/      # 对话数据集
│   └── coding_problems/        # 编程问题集
└── utils/
    ├── data_loader.py          # 数据加载工具
    ├── metrics.py              # 评估指标
    └── visualization.py        # 训练可视化
```

## 🚀 快速开始

### 1. 安装额外依赖

```bash
pip install -r examples/rl_training/requirements.txt
```

### 2. 训练文本游戏智能体

```bash
# 基本训练
python examples/rl_training/scripts/train_text_game.py \
  --config examples/rl_training/config/text_game_config.yaml \
  --epochs 100 \
  --batch-size 32

# 使用 GPU 训练
python examples/rl_training/scripts/train_text_game.py \
  --config examples/rl_training/config/text_game_config.yaml \
  --device cuda \
  --epochs 200

# 继续训练现有模型
python examples/rl_training/scripts/train_text_game.py \
  --config examples/rl_training/config/text_game_config.yaml \
  --resume checkpoint.pth \
  --epochs 50
```

### 3. 评估智能体性能

```bash
# 评估文本游戏智能体
python examples/rl_training/scripts/evaluate_agent.py \
  --agent text_game \
  --model-path models/text_game_agent.pth \
  --episodes 10

# 评估对话智能体
python examples/rl_training/scripts/evaluate_agent.py \
  --agent dialogue \
  --model-path models/dialogue_agent.pth \
  --test-data data/dialogue_datasets/test.json

# 生成评估报告
python examples/rl_training/scripts/evaluate_agent.py \
  --agent codegen \
  --model-path models/codegen_agent.pth \
  --output report.html
```

## ⚙️ 配置说明

### 训练配置 (YAML 格式)

```yaml
# 训练参数
training:
  algorithm: "ppo"           # 算法: ppo, a2c, dqn
  learning_rate: 0.0001      # 学习率
  gamma: 0.99                # 折扣因子
  clip_epsilon: 0.2          # PPO 裁剪参数
  entropy_coef: 0.01         # 熵系数

# 环境参数
environment:
  max_steps: 1000            # 最大步数
  reward_scale: 1.0          # 奖励缩放
  difficulty: "medium"       # 难度级别

# 模型参数
model:
  hidden_size: 512           # 隐藏层大小
  num_layers: 3              # 层数
  dropout: 0.1               # Dropout 率

# 监控参数
monitoring:
  log_interval: 10           # 日志间隔
  save_interval: 100         # 保存间隔
  eval_interval: 50          # 评估间隔
```

## 🎮 环境接口

### 基础环境类

```python
class BaseRLEnv:
    def reset(self):
        """重置环境，返回初始状态"""
        pass
        
    def step(self, action):
        """执行动作，返回 (next_state, reward, done, info)"""
        pass
        
    def get_state(self):
        """获取当前状态"""
        pass
        
    def render(self):
        """渲染环境状态"""
        pass
```

### 文本游戏环境示例

```python
class TextGameEnv(BaseRLEnv):
    def __init__(self, scenario_file):
        self.scenario = load_scenario(scenario_file)
        self.current_room = self.scenario.start_room
        self.inventory = []
        self.score = 0
        self.steps = 0
        
    def reset(self):
        self.current_room = self.scenario.start_room
        self.inventory = []
        self.score = 0
        self.steps = 0
        return self._get_state()
        
    def step(self, action):
        # 解析和执行动作
        reward = self._execute_action(action)
        done = self._check_termination()
        next_state = self._get_state()
        info = {"score": self.score, "steps": self.steps}
        
        return next_state, reward, done, info
```

## 🤖 智能体架构

### 基础智能体类

```python
class BaseAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super().__init__()
        self.state_encoder = nn.Linear(state_size, hidden_size)
        self.action_policy = nn.Linear(hidden_size, action_size)
        self.value_net = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        hidden = F.relu(self.state_encoder(state))
        action_probs = F.softmax(self.action_policy(hidden), dim=-1)
        state_value = self.value_net(hidden)
        return action_probs, state_value
        
    def act(self, state):
        with torch.no_grad():
            action_probs, _ = self.forward(state)
            action = torch.multinomial(action_probs, 1).item()
        return action
```

## 📊 训练流程

### PPO 训练循环

```python
def train_ppo(agent, env, config):
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate)
    
    for episode in range(config.episodes):
        state = env.reset()
        episode_reward = 0
        
        states, actions, rewards, dones = [], [], [], []
        
        for step in range(config.max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        # PPO 更新
        advantages = compute_advantages(rewards, dones, agent, config.gamma)
        loss = compute_ppo_loss(agent, states, actions, advantages, config.clip_epsilon)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 📈 评估指标

### 性能指标
- **平均奖励**: 每个回合的平均累积奖励
- **成功率**: 完成任务的比例
- **步数效率**: 完成任务的平均步数
- **探索率**: 访问不同状态的比例

### 训练监控
```bash
# 训练进度示例
Epoch: 50/100 | Reward: 15.2 ± 3.1 | Success: 72% | Steps: 45.3
Epoch: 100/100 | Reward: 28.7 ± 2.5 | Success: 92% | Steps: 32.1
```

## 🎯 高级功能

### 课程学习
```yaml
curriculum:
  enabled: true
  levels:
    - difficulty: "easy"
      scenarios: ["beginner_*.json"]
      min_success: 0.8
    - difficulty: "medium" 
      scenarios: ["intermediate_*.json"]
      min_success: 0.7
    - difficulty: "hard"
      scenarios: ["expert_*.json"]
      min_success: 0.6
```

### 集成学习
```python
# 多个智能体集成
ensemble_agents = [
    load_agent("models/agent1.pth"),
    load_agent("models/agent2.pth"), 
    load_agent("models/agent3.pth")
]

def ensemble_act(state):
    actions = [agent.act(state) for agent in ensemble_agents]
    return max(set(actions), key=actions.count)
```

## 🔧 故障排除

### 常见问题

1. **奖励不收敛**
   - 调整奖励缩放
   - 检查奖励函数设计
   - 增加熵系数鼓励探索

2. **训练不稳定**
   - 减小学习率
   - 增加批量大小
   - 使用梯度裁剪

3. **过拟合**
   - 增加 Dropout
   - 使用早停策略
   - 增加正则化

## 📝 下一步

1. **自定义环境**: 修改 `environments/` 中的类创建新环境
2. **调整配置**: 修改 YAML 配置文件优化训练参数
3. **添加监控**: 集成 TensorBoard 或 WandB 进行可视化
4. **部署生产**: 使用训练好的模型进行实际应用

---

**维护团队**: 强化学习研究组  
**最后更新**: 2025-08-19