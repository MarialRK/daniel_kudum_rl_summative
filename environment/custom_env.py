import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class PlasticPollutionEnv(gym.Env):
    """
    Enhanced Custom Environment for Plastic Pollution Cleanup
    Optimized reward structure for faster learning
    """
    
    def __init__(self, render_mode=None, grid_size=12):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=collect
        self.action_space = spaces.Discrete(5)
        
        # Partial observability: 5x5 view
        self.view_size = 5
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(self.view_size, self.view_size), dtype=np.int8
        )
        
        # Current flow
        self.current_direction = 3
        self.current_strength = 0.2  # Reduced from 0.3
        
        # Wind effect
        self.wind_active = False
        
        # Plastic types with IMPROVED REWARDS
        self.plastic_types = {
            1: {"name": "small", "reward": 20, "color": "brown"},      # Was 5
            2: {"name": "large", "reward": 50, "color": "dark brown"},  # Was 15
            3: {"name": "micro", "reward": 10, "color": "light brown"}  # Was 2
        }
        
        self.agent_pos = None
        self.plastic_positions = []
        self.obstacle_positions = []
        self.step_count = 0
        self.max_steps = 200
        self.collected_plastic = 0
        self.total_plastic = 0
        self.total_reward = 0
        
        # IMPROVED REWARD PARAMETERS
        self.step_penalty = -0.5      # Was -1 (less punishing)
        self.obstacle_penalty = -3    # Was -8 (less punishing)
        self.current_penalty = -1     # Was -2
        self.completion_bonus = 100   # Was 50
        
        self.full_grid = None
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.collected_plastic = 0
        self.total_reward = 0
        
        self.current_direction = random.randint(0, 3)
        self.current_strength = random.uniform(0.1, 0.3)
        self.wind_active = random.random() < 0.15
        
        self.full_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Fewer obstacles for easier learning
        num_obstacles = random.randint(3, 5)
        self.obstacle_positions = []
        for _ in range(num_obstacles):
            while True:
                pos = (random.randint(0, self.grid_size-1), 
                       random.randint(0, self.grid_size-1))
                if pos != (0, 0) and pos not in self.obstacle_positions:
                    if abs(pos[0]) + abs(pos[1]) > 3:
                        self.obstacle_positions.append(pos)
                        self.full_grid[pos] = 3
                        break
        
        # Fewer plastic items for easier learning
        num_plastic = random.randint(5, 7)
        self.total_plastic = num_plastic
        self.plastic_positions = []
        
        for _ in range(num_plastic):
            plastic_type = random.choices([1, 2, 3], weights=[40, 40, 20])[0]
            
            while True:
                pos = (random.randint(0, self.grid_size-1), 
                       random.randint(0, self.grid_size-1))
                if (pos != (0, 0) and pos not in self.obstacle_positions and 
                    pos not in [(p[0], p[1]) for p in self.plastic_positions]):
                    self.plastic_positions.append((pos[0], pos[1], plastic_type))
                    self.full_grid[pos] = plastic_type
                    break
        
        self.agent_pos = (0, 0)
        self.full_grid[self.agent_pos] = 4
        
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        x, y = self.agent_pos
        half = self.view_size // 2
        obs = np.zeros((self.view_size, self.view_size), dtype=np.int8)
        
        for i in range(self.view_size):
            for j in range(self.view_size):
                world_x = x + i - half
                world_y = y + j - half
                
                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    obs[i, j] = self.full_grid[world_x, world_y]
                else:
                    obs[i, j] = -1
        
        return obs
    
    def _get_info(self):
        return {
            "collected": self.collected_plastic,
            "total_plastic": self.total_plastic,
            "steps": self.step_count,
            "remaining": self.total_plastic - self.collected_plastic,
            "current_direction": self.current_direction,
            "wind_active": self.wind_active
        }
    
    def _apply_current(self, x, y):
        new_x, new_y = x, y
        
        if random.random() < self.current_strength:
            if self.current_direction == 0:
                new_x = max(0, x - 1)
            elif self.current_direction == 1:
                new_x = min(self.grid_size - 1, x + 1)
            elif self.current_direction == 2:
                new_y = max(0, y - 1)
            elif self.current_direction == 3:
                new_y = min(self.grid_size - 1, y + 1)
            
            if self.full_grid[new_x, new_y] == 3:
                return x, y, True
            return new_x, new_y, False
        
        return x, y, False
    
    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False
        truncated = False
        
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        # Execute action
        if action == 0:
            new_x = max(0, x - 1)
        elif action == 1:
            new_x = min(self.grid_size - 1, x + 1)
        elif action == 2:
            new_y = max(0, y - 1)
        elif action == 3:
            new_y = min(self.grid_size - 1, y + 1)
        elif action == 4:
            cell_value = self.full_grid[x, y]
            if cell_value in [1, 2, 3]:
                for px, py, ptype in self.plastic_positions:
                    if px == x and py == y:
                        plastic_reward = self.plastic_types[ptype]["reward"]
                        reward += plastic_reward
                        self.collected_plastic += 1
                        self.full_grid[x, y] = 0
                        self.plastic_positions.remove((px, py, ptype))
                        break
        
        if action in [0, 1, 2, 3]:
            if self.full_grid[new_x, new_y] == 3:
                reward += self.obstacle_penalty
                new_x, new_y = x, y
            else:
                if self.full_grid[x, y] == 4:
                    self.full_grid[x, y] = 0
                
                self.agent_pos = (new_x, new_y)
                
                current_x, current_y, hit_obstacle = self._apply_current(new_x, new_y)
                if hit_obstacle:
                    reward += self.obstacle_penalty
                elif (current_x, current_y) != (new_x, new_y):
                    reward += self.current_penalty
                    self.agent_pos = (current_x, current_y)
                
                cell_value = self.full_grid[self.agent_pos]
                if cell_value in [1, 2, 3]:
                    for px, py, ptype in self.plastic_positions:
                        if px == self.agent_pos[0] and py == self.agent_pos[1]:
                            plastic_reward = self.plastic_types[ptype]["reward"]
                            reward += plastic_reward
                            self.collected_plastic += 1
                            self.full_grid[self.agent_pos] = 0
                            self.plastic_positions.remove((px, py, ptype))
                            break
            
            self.full_grid[self.agent_pos] = 4
            reward += self.step_penalty
        
        if self.wind_active and random.random() < 0.1:
            reward += -1
        
        if self.collected_plastic >= self.total_plastic:
            done = True
            reward += self.completion_bonus
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        self.total_reward += reward
        
        return self._get_obs(), reward, done, truncated, self._get_info()
    
    def render(self):
        pass
    
    def close(self):
        pass