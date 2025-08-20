import numpy as np
from collections import deque
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class TrainingMetrics:
    """训练指标跟踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # 训练指标
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_steps = deque(maxlen=window_size)
        self.episode_scores = deque(maxlen=window_size)
        self.episode_success = deque(maxlen=window_size)
        
        # 时间序列数据
        self.history = {
            'rewards': [],
            'steps': [],
            'scores': [],
            'success': [],
            'timesteps': []
        }
    
    def record_episode(self, reward: float, steps: int, score: float = 0, success: bool = False):
        """记录一个回合的指标"""
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.episode_scores.append(score)
        self.episode_success.append(1 if success else 0)
        
        # 更新历史记录
        self.history['rewards'].append(reward)
        self.history['steps'].append(steps)
        self.history['scores'].append(score)
        self.history['success'].append(1 if success else 0)
    
    def get_recent_metrics(self) -> Dict[str, float]:
        """获取最近窗口的指标统计"""
        if not self.episode_rewards:
            return {
                'reward_mean': 0,
                'reward_std': 0,
                'reward_max': 0,
                'reward_min': 0,
                'steps_mean': 0,
                'steps_std': 0,
                'score_mean': 0,
                'score_std': 0,
                'success_rate': 0
            }
        
        rewards = np.array(self.episode_rewards)
        steps = np.array(self.episode_steps)
        scores = np.array(self.episode_scores)
        success = np.array(self.episode_success)
        
        return {
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_max': float(np.max(rewards)),
            'reward_min': float(np.min(rewards)),
            'steps_mean': float(np.mean(steps)),
            'steps_std': float(np.std(steps)),
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'success_rate': float(np.mean(success))
        }
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        """获取所有历史指标"""
        return self.history.copy()
    
    def save(self, filepath: str):
        """保存指标到文件"""
        data = {
            'window_size': self.window_size,
            'recent_rewards': list(self.episode_rewards),
            'recent_steps': list(self.episode_steps),
            'recent_scores': list(self.episode_scores),
            'recent_success': list(self.episode_success),
            'history': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """从文件加载指标"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.window_size = data['window_size']
        self.episode_rewards = deque(data['recent_rewards'], maxlen=self.window_size)
        self.episode_steps = deque(data['recent_steps'], maxlen=self.window_size)
        self.episode_scores = deque(data['recent_scores'], maxlen=self.window_size)
        self.episode_success = deque(data['recent_success'], maxlen=self.window_size)
        self.history = data['history']


class RLMetricsAnalyzer:
    """强化学习指标分析器"""
    
    def __init__(self):
        self.metrics_data = {}
    
    def add_experiment(self, name: str, metrics: TrainingMetrics):
        """添加实验数据"""
        self.metrics_data[name] = metrics.get_all_metrics()
    
    def plot_learning_curves(self, output_dir: str = "plots"):
        """绘制学习曲线"""
        Path(output_dir).mkdir(exist_ok=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        for name, data in self.metrics_data.items():
            episodes = range(1, len(data['rewards']) + 1)
            
            # 奖励曲线
            ax1.plot(episodes, data['rewards'], label=name, alpha=0.7)
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 成功率曲线（滑动平均）
            success_rates = np.convolve(data['success'], np.ones(100)/100, mode='valid')
            ax2.plot(episodes[:len(success_rates)], success_rates, label=name, alpha=0.7)
            ax2.set_title('Success Rate (100-episode moving average)')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Success Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 步数曲线
            ax3.plot(episodes, data['steps'], label=name, alpha=0.7)
            ax3.set_title('Episode Steps')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Steps')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 分数曲线
            ax4.plot(episodes, data['scores'], label=name, alpha=0.7)
            ax4.set_title('Episode Scores')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Score')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_comparison(self, output_dir: str = "plots"):
        """绘制统计比较图"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 准备数据
        names = []
        final_rewards = []
        final_success = []
        efficiency = []
        
        for name, data in self.metrics_data.items():
            names.append(name)
            
            # 最后100个回合的指标
            final_100 = slice(-100, None) if len(data['rewards']) > 100 else slice(None)
            
            final_rewards.append(np.mean(data['rewards'][final_100]))
            final_success.append(np.mean(data['success'][final_100]))
            
            # 效率：平均奖励/平均步数
            avg_reward = np.mean(data['rewards'][final_100])
            avg_steps = np.mean(data['steps'][final_100])
            efficiency.append(avg_reward / avg_steps if avg_steps > 0 else 0)
        
        # 创建比较图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 最终奖励比较
        bars1 = ax1.bar(names, final_rewards, alpha=0.7)
        ax1.set_title('Final Average Reward (last 100 episodes)')
        ax1.set_ylabel('Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 最终成功率比较
        bars2 = ax2.bar(names, final_success, alpha=0.7)
        ax2.set_title('Final Success Rate (last 100 episodes)')
        ax2.set_ylabel('Success Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom')
        
        # 效率比较
        bars3 = ax3.bar(names, efficiency, alpha=0.7)
        ax3.set_title('Efficiency (Reward/Step)')
        ax3.set_ylabel('Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/statistical_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_dir: str = "reports"):
        """生成详细报告"""
        Path(output_dir).mkdir(exist_ok=True)
        
        report = {
            'experiments': {},
            'summary': {}
        }
        
        for name, data in self.metrics_data.items():
            n_episodes = len(data['rewards'])
            final_100 = slice(-100, None) if n_episodes > 100 else slice(None)
            
            experiment_stats = {
                'total_episodes': n_episodes,
                'final_reward_mean': float(np.mean(data['rewards'][final_100])),
                'final_reward_std': float(np.std(data['rewards'][final_100])),
                'final_success_rate': float(np.mean(data['success'][final_100])),
                'final_steps_mean': float(np.mean(data['steps'][final_100])),
                'max_reward': float(np.max(data['rewards'])),
                'min_reward': float(np.min(data['rewards'])),
                'efficiency': float(np.mean(data['rewards'][final_100]) / 
                               np.mean(data['steps'][final_100]) if np.mean(data['steps'][final_100]) > 0 else 0)
            }
            
            report['experiments'][name] = experiment_stats
        
        # 总体统计
        if len(self.metrics_data) > 1:
            best_experiment = max(
                report['experiments'].items(),
                key=lambda x: x[1]['final_reward_mean']
            )
            report['summary']['best_experiment'] = best_experiment[0]
            report['summary']['best_reward'] = best_experiment[1]['final_reward_mean']
        
        # 保存报告
        with open(f"{output_dir}/metrics_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成文本报告
        with open(f"{output_dir}/metrics_report.txt", 'w') as f:
            f.write("强化学习训练结果报告\n")
            f.write("=" * 50 + "\n\n")
            
            for name, stats in report['experiments'].items():
                f.write(f"实验: {name}\n")
                f.write(f"  总回合数: {stats['total_episodes']}\n")
                f.write(f"  最终平均奖励: {stats['final_reward_mean']:.2f} ± {stats['final_reward_std']:.2f}\n")
                f.write(f"  最终成功率: {stats['final_success_rate']:.2%}\n")
                f.write(f"  最终平均步数: {stats['final_steps_mean']:.1f}\n")
                f.write(f"  最高奖励: {stats['max_reward']:.2f}\n")
                f.write(f"  最低奖励: {stats['min_reward']:.2f}\n")
                f.write(f"  效率(奖励/步数): {stats['efficiency']:.4f}\n\n")
            
            if 'best_experiment' in report['summary']:
                f.write(f"最佳实验: {report['summary']['best_experiment']}\n")
                f.write(f"最佳奖励: {report['summary']['best_reward']:.2f}\n")
        
        return report


# 工具函数
def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
    """计算置信区间"""
    from scipy import stats
    import numpy as np
    
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    if n < 2:
        return mean, 0
    
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, h


def moving_average(data: List[float], window_size: int = 100) -> List[float]:
    """计算滑动平均"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def detect_convergence(rewards: List[float], window: int = 100, threshold: float = 0.01) -> int:
    """检测收敛点"""
    if len(rewards) < window * 2:
        return -1
    
    for i in range(window, len(rewards) - window):
        prev_mean = np.mean(rewards[i-window:i])
        next_mean = np.mean(rewards[i:i+window])
        
        if abs(next_mean - prev_mean) / (abs(prev_mean) + 1e-8) < threshold:
            return i
    
    return -1


if __name__ == "__main__":
    # 测试代码
    metrics = TrainingMetrics()
    
    # 模拟一些数据
    for i in range(200):
        reward = np.random.normal(i * 0.1, 1)
        steps = np.random.randint(20, 100)
        success = np.random.random() > 0.7
        metrics.record_episode(reward, steps, reward * 2, success)
    
    print("最近指标:", metrics.get_recent_metrics())
    
    # 测试分析器
    analyzer = RLMetricsAnalyzer()
    analyzer.add_experiment("test", metrics)
    analyzer.plot_learning_curves()
    analyzer.generate_report()