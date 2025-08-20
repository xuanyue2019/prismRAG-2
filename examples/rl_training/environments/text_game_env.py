import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

class TextGameEnv(gym.Env):
    """基于文本的冒险游戏环境"""
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, scenario_files: List[str] = None, difficulty: str = "medium", 
                 max_steps: int = 1000, use_embeddings: bool = True):
        super().__init__()
        
        self.scenario_files = scenario_files or []
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.use_embeddings = use_embeddings
        
        # 加载场景
        self.scenarios = self._load_scenarios()
        self.current_scenario = None
        self.current_state = None
        
        # 文本编码器
        if use_embeddings:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # 定义动作空间（基本游戏命令）
        self.actions = [
            "look", "go north", "go south", "go east", "go west",
            "take", "use", "talk", "inventory", "examine"
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        
        # 定义观察空间
        if use_embeddings:
            # 使用嵌入向量作为观察
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32
            )
        else:
            # 使用文本作为观察（需要预处理）
            self.observation_space = spaces.Dict({
                'text': spaces.Text(max_length=512)
            })
        
        self.reset()
    
    def _load_scenarios(self) -> List[Dict]:
        """加载游戏场景"""
        scenarios = []
        for scenario_file in self.scenario_files:
            try:
                with open(scenario_file, 'r', encoding='utf-8') as f:
                    scenario = json.load(f)
                    scenarios.append(scenario)
            except FileNotFoundError:
                print(f"Warning: Scenario file {scenario_file} not found")
        return scenarios
    
    def _encode_text(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        if not self.use_embeddings:
            return text
        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                              padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机选择一个场景
        if self.scenarios:
            self.current_scenario = random.choice(self.scenarios)
        else:
            # 默认场景（如果没有提供场景文件）
            self.current_scenario = self._create_default_scenario()
        
        # 初始化游戏状态
        self.current_state = {
            'current_room': self.current_scenario['start_room'],
            'inventory': [],
            'visited_rooms': set([self.current_scenario['start_room']]),
            'completed_objects': set(),
            'score': 0,
            'steps': 0,
            'game_over': False
        }
        
        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _create_default_scenario(self) -> Dict:
        """创建默认游戏场景"""
        return {
            "name": "Default Adventure",
            "start_room": "entrance",
            "rooms": {
                "entrance": {
                    "description": "You are in a grand entrance hall. There are doors to the north and east.",
                    "exits": {"north": "library", "east": "dining_room"},
                    "items": ["torch"]
                },
                "library": {
                    "description": "A dusty library filled with ancient books. There's a door to the south.",
                    "exits": {"south": "entrance"},
                    "items": ["key"]
                },
                "dining_room": {
                    "description": "A large dining room with a long table. There's a door to the west.",
                    "exits": {"west": "entrance"},
                    "items": ["apple"]
                }
            },
            "objects": {
                "treasure_chest": {
                    "location": "library",
                    "required_items": ["key"],
                    "description": "A locked treasure chest.",
                    "reward": 10
                }
            }
        }
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """执行一个动作"""
        if self.current_state['game_over']:
            return self._get_observation(), 0, True, False, self._get_info()
        
        action_text = self.actions[action]
        reward = 0
        terminated = False
        truncated = False
        
        # 执行动作
        if action_text == "look":
            reward = self._handle_look()
        elif action_text.startswith("go "):
            direction = action_text.split()[1]
            reward = self._handle_move(direction)
        elif action_text == "take":
            reward = self._handle_take()
        elif action_text == "use":
            reward = self._handle_use()
        elif action_text == "talk":
            reward = self._handle_talk()
        elif action_text == "inventory":
            reward = self._handle_inventory()
        elif action_text == "examine":
            reward = self._handle_examine()
        
        # 更新状态
        self.current_state['steps'] += 1
        self.current_state['score'] += reward
        
        # 检查终止条件
        if self.current_state['steps'] >= self.max_steps:
            truncated = True
        
        # 检查游戏是否完成
        if self._check_game_completion():
            reward += 10  # 完成游戏额外奖励
            terminated = True
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _handle_look(self) -> float:
        """处理查看动作"""
        current_room = self.current_state['current_room']
        room_info = self.current_scenario['rooms'][current_room]
        return 0.1  # 小奖励鼓励探索
    
    def _handle_move(self, direction: str) -> float:
        """处理移动动作"""
        current_room = self.current_state['current_room']
        room_info = self.current_scenario['rooms'][current_room]
        
        if direction in room_info['exits']:
            new_room = room_info['exits'][direction]
            self.current_state['current_room'] = new_room
            
            # 探索奖励
            if new_room not in self.current_state['visited_rooms']:
                self.current_state['visited_rooms'].add(new_room)
                return 1.0  # 探索新区域奖励
            return 0.1  # 移动奖励
        else:
            return -0.5  # 无效移动惩罚
    
    def _handle_take(self) -> float:
        """处理拾取动作"""
        current_room = self.current_state['current_room']
        room_info = self.current_scenario['rooms'][current_room]
        
        if room_info['items']:
            item = room_info['items'].pop(0)
            self.current_state['inventory'].append(item)
            return 2.0  # 拾取物品奖励
        return -0.5  # 没有物品可拾取
    
    def _handle_use(self) -> float:
        """处理使用动作"""
        # 简化的使用逻辑 - 检查是否能打开宝箱
        current_room = self.current_state['current_room']
        
        if "key" in self.current_state['inventory'] and current_room == "library":
            if "treasure_chest" not in self.current_state['completed_objects']:
                self.current_state['completed_objects'].add("treasure_chest")
                return 5.0  # 成功使用物品奖励
        
        return -0.3  # 无法使用物品
    
    def _handle_talk(self) -> float:
        """处理对话动作"""
        return -0.2  # 当前场景没有NPC
    
    def _handle_inventory(self) -> float:
        """处理查看库存"""
        return 0.0  # 中性动作
    
    def _handle_examine(self) -> float:
        """处理检查动作"""
        return 0.1  # 小奖励鼓励仔细检查
    
    def _get_observation(self) -> Any:
        """获取当前观察"""
        current_room = self.current_state['current_room']
        room_info = self.current_scenario['rooms'][current_room]
        
        # 构建观察文本
        obs_text = f"Location: {current_room}\n"
        obs_text += f"Description: {room_info['description']}\n"
        obs_text += f"Exits: {', '.join(room_info['exits'].keys())}\n"
        
        if room_info['items']:
            obs_text += f"Items here: {', '.join(room_info['items'])}\n"
        
        obs_text += f"Inventory: {', '.join(self.current_state['inventory'])}"
        
        if self.use_embeddings:
            return self._encode_text(obs_text)
        else:
            return {'text': obs_text}
    
    def _get_info(self) -> Dict:
        """获取环境信息"""
        return {
            'score': self.current_state['score'],
            'steps': self.current_state['steps'],
            'current_room': self.current_state['current_room'],
            'inventory': self.current_state['inventory'],
            'visited_rooms': len(self.current_state['visited_rooms']),
            'completed_objects': len(self.current_state['completed_objects'])
        }
    
    def _check_game_completion(self) -> bool:
        """检查游戏是否完成"""
        # 简化的完成条件：打开宝箱
        return "treasure_chest" in self.current_state['completed_objects']
    
    def render(self, mode: str = 'human'):
        """渲染环境"""
        if mode == 'human':
            obs = self._get_observation()
            if isinstance(obs, dict):
                print(obs['text'])
            else:
                print("Current state (embedded)")
            print(f"Score: {self.current_state['score']}")
            print(f"Steps: {self.current_state['steps']}")
        elif mode == 'ansi':
            return str(self._get_observation())
    
    def close(self):
        """清理资源"""
        if hasattr(self, 'embedding_model'):
            del self.embedding_model


# 环境注册
def register_text_game_env():
    """注册文本游戏环境"""
    gym.register(
        id='TextGame-v1',
        entry_point='examples.rl_training.environments.text_game_env:TextGameEnv',
        kwargs={
            'scenario_files': [
                'data/text_game_scenarios/beginner_1.json',
                'data/text_game_scenarios/beginner_2.json'
            ],
            'difficulty': 'medium',
            'max_steps': 1000,
            'use_embeddings': True
        }
    )


if __name__ == "__main__":
    # 测试环境
    env = TextGameEnv(use_embeddings=False)
    obs, info = env.reset()
    print("Initial observation:")
    print(obs['text'])
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step + 1}:")
        print(f"Action: {env.actions[action]}")
        print(f"Reward: {reward}")
        print(f"Observation: {obs['text'][:100]}...")
        
        if terminated or truncated:
            break