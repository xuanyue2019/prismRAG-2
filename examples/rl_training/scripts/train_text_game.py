#!/usr/bin/env python3
"""
文本游戏强化学习训练脚本
使用 PPO 算法训练文本游戏智能体
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.rl_training.environments.text_game_env import TextGameEnv, register_text_game_env
from examples.rl_training.agents.base_agent import TextGameAgent, compute_advantages
from examples.rl_training.utils.metrics import TrainingMetrics
from examples.rl_training.utils.visualization import plot_training_progress


class PPOTrainer:
    """PPO 算法训练器"""
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # 创建环境
        self.env = TextGameEnv(
            scenario_files=config['environment']['scenario_files'],
            difficulty=config['environment']['difficulty'],
            max_steps=config['environment']['max_steps'],
            use_embeddings=True
        )
        
        # 创建智能体
        self.agent = TextGameAgent(
            embedding_size=384,  # all-MiniLM-L6-v2 的嵌入维度
            action_size=self.env.action_space.n,
            hidden_size=config['model']['text_encoder']['hidden_size'],
            num_layers=config['model']['text_encoder']['num_layers'],
            dropout=config['model']['text_encoder']['dropout']
        ).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.agent.parameters(), 
            lr=config['training']['learning_rate'],
            eps=1e-5
        )
        
        # 训练状态
        self.global_step = 0
        self.episode = 0
        self.best_reward = -float('inf')
        
        # 监控
        self.writer = None
        self.metrics = TrainingMetrics()
        
        # 创建输出目录
        self._setup_directories()
    
    def _setup_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config['monitoring']['checkpoint_dir'], exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('runs', exist_ok=True)
    
    def compute_ppo_loss(self, old_action_log_probs, advantages, action_log_probs, ratio, clip_epsilon):
        """计算 PPO 损失"""
        # 策略损失
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        ).mean()
        
        # 价值损失
        value_loss = advantages.pow(2).mean()
        
        return policy_loss + 0.5 * value_loss
    
    def train_epoch(self):
        """训练一个周期"""
        batch_states, batch_actions, batch_rewards, batch_dones = [], [], [], []
        batch_old_log_probs, batch_values = [], []
        
        # 收集经验
        state, info = self.env.reset()
        episode_reward = 0
        
        for step in range(self.config['training']['n_steps']):
            # 选择动作
            action, action_probs, value = self.agent.get_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 存储经验
            batch_states.append(state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_values.append(value.item())
            
            # 计算旧的对数概率
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([action]).to(self.device)
            old_log_probs, _, _ = self.agent.evaluate_actions(state_tensor, action_tensor)
            batch_old_log_probs.append(old_log_probs.item())
            
            state = next_state
            episode_reward += reward
            self.global_step += 1
            
            if done:
                # 记录回合结果
                self.metrics.record_episode(episode_reward, step + 1, info.get('score', 0))
                self.episode += 1
                
                # 重置环境
                state, info = self.env.reset()
                episode_reward = 0
                
                # 记录指标
                if self.episode % self.config['monitoring']['log_interval'] == 0:
                    self._log_metrics()
        
        # 准备训练数据
        states = torch.FloatTensor(np.array(batch_states)).to(self.device)
        actions = torch.LongTensor(batch_actions).to(self.device)
        rewards = torch.FloatTensor(batch_rewards).to(self.device)
        dones = torch.FloatTensor(batch_dones).to(self.device)
        old_log_probs = torch.FloatTensor(batch_old_log_probs).to(self.device)
        values = torch.FloatTensor(batch_values).to(self.device)
        
        # 计算优势
        with torch.no_grad():
            next_value = self.agent.get_action(state)[2].item()
        
        advantages = compute_advantages(
            rewards, values, torch.FloatTensor([next_value]), dones,
            self.config['training']['gamma'], 0.95
        )
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮优化
        for epoch in range(self.config['training']['n_epochs']):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.config['training']['batch_size']):
                end = start + self.config['training']['batch_size']
                batch_indices = indices[start:end]
                
                batch_states_epoch = states[batch_indices]
                batch_actions_epoch = actions[batch_indices]
                batch_advantages_epoch = advantages[batch_indices]
                batch_old_log_probs_epoch = old_log_probs[batch_indices]
                
                # 计算新策略的对数概率
                new_log_probs, entropy, new_values = self.agent.evaluate_actions(
                    batch_states_epoch, batch_actions_epoch
                )
                
                # 计算比率和损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs_epoch)
                loss = self.compute_ppo_loss(
                    batch_old_log_probs_epoch,
                    batch_advantages_epoch,
                    new_log_probs,
                    ratio,
                    self.config['training']['clip_epsilon']
                )
                
                # 添加熵奖励
                entropy_loss = -self.config['training']['entropy_coef'] * entropy.mean()
                total_loss = loss + entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), 
                    self.config['training']['max_grad_norm']
                )
                
                self.optimizer.step()
        
        return self.metrics.get_recent_metrics()
    
    def _log_metrics(self):
        """记录训练指标"""
        metrics = self.metrics.get_recent_metrics()
        
        if self.writer:
            self.writer.add_scalar('Reward/Mean', metrics['reward_mean'], self.global_step)
            self.writer.add_scalar('Reward/Max', metrics['reward_max'], self.global_step)
            self.writer.add_scalar('Reward/Min', metrics['reward_min'], self.global_step)
            self.writer.add_scalar('Steps/Mean', metrics['steps_mean'], self.global_step)
            self.writer.add_scalar('Score/Mean', metrics['score_mean'], self.global_step)
        
        print(f"Episode {self.episode} - "
              f"Reward: {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f} - "
              f"Steps: {metrics['steps_mean']:.1f} - "
              f"Score: {metrics['score_mean']:.1f}")
    
    def evaluate(self, num_episodes=10):
        """评估智能体性能"""
        total_rewards = []
        total_scores = []
        success_count = 0
        
        for _ in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _, _ = self.agent.get_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            total_scores.append(info.get('score', 0))
            
            if info.get('completed_objects', 0) > 0:
                success_count += 1
        
        metrics = {
            'eval_reward_mean': np.mean(total_rewards),
            'eval_reward_std': np.std(total_rewards),
            'eval_score_mean': np.mean(total_scores),
            'eval_success_rate': success_count / num_episodes
        }
        
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Eval/{key}', value, self.global_step)
        
        return metrics
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'global_step': self.global_step,
            'episode': self.episode,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'config': self.config
        }
        
        # 常规保存
        checkpoint_path = os.path.join(
            self.config['monitoring']['checkpoint_dir'],
            f'checkpoint_{self.global_step}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(
                self.config['monitoring']['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
    
    def train(self, total_timesteps):
        """主训练循环"""
        # 初始化 TensorBoard
        if self.config['monitoring']['use_tensorboard']:
            self.writer = SummaryWriter(
                log_dir=os.path.join('runs', self.config['experiment']['name'])
            )
        
        print(f"开始训练 {self.config['experiment']['name']}")
        print(f"设备: {self.device}")
        print(f"总时间步数: {total_timesteps}")
        
        try:
            while self.global_step < total_timesteps:
                # 训练一个周期
                metrics = self.train_epoch()
                
                # 定期评估
                if self.episode % self.config['monitoring']['eval_interval'] == 0:
                    eval_metrics = self.evaluate(self.config['monitoring']['eval_episodes'])
                    print(f"评估结果 - 奖励: {eval_metrics['eval_reward_mean']:.2f} - "
                          f"成功率: {eval_metrics['eval_success_rate']:.2%}")
                    
                    # 更新最佳奖励
                    if eval_metrics['eval_reward_mean'] > self.best_reward:
                        self.best_reward = eval_metrics['eval_reward_mean']
                        self.save_checkpoint(is_best=True)
                
                # 定期保存
                if self.episode % self.config['monitoring']['save_interval'] == 0:
                    self.save_checkpoint()
        
        except KeyboardInterrupt:
            print("训练被用户中断")
        finally:
            # 保存最终模型
            self.save_checkpoint()
            
            if self.writer:
                self.writer.close()
            
            # 生成训练报告
            self._generate_report()
    
    def _generate_report(self):
        """生成训练报告"""
        report_path = os.path.join(
            self.config['monitoring']['checkpoint_dir'],
            'training_report.txt'
        )
        
        with open(report_path, 'w') as f:
            f.write(f"训练报告 - {self.config['experiment']['name']}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总回合数: {self.episode}\n")
            f.write(f"总时间步数: {self.global_step}\n")
            f.write(f"最佳评估奖励: {self.best_reward:.2f}\n\n")
            
            f.write("最终指标:\n")
            metrics = self.metrics.get_recent_metrics()
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description='训练文本游戏智能体')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备 (auto, cpu, cuda)')
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='总训练时间步数')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 创建训练器
    trainer = PPOTrainer(config, device)
    
    # 恢复训练（如果指定）
    if args.resume:
        print(f"从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.agent.load_state_dict(checkpoint['agent_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint['global_step']
        trainer.episode = checkpoint['episode']
        trainer.best_reward = checkpoint['best_reward']
    
    # 开始训练
    trainer.train(args.timesteps)


if __name__ == "__main__":
    # 注册环境
    register_text_game_env()
    
    # 设置默认配置文件路径
    if len(sys.argv) == 1:
        default_config = "examples/rl_training/config/text_game_config.yaml"
        if os.path.exists(default_config):
            sys.argv.extend(['--config', default_config])
    
    main()