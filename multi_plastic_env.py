"""
Multi-Plastic Pollution Environment
Supports Small (+20), Large (+50), and Micro (+10) plastics
Compatible with Stable-Baselines3 (Gymnasium)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class MultiPlasticEnv(gym.Env):
    """Environment with multiple plastic types (small, large, micro)"""
    
    def __init__(self, grid_size=8, num_plastics=5):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_plastics = num_plastics
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=collect
        self.action_space = spaces.Discrete(5)
        
        # Observation space: grid values (0-4)
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(grid_size, grid_size), dtype=np.int8
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent_pos = (0, 0)
        self.steps = 0
        self.collected = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Plastic types: 1=small (+20), 2=large (+50), 3=micro (+10)
        self.reward_map = {1: 20, 2: 50, 3: 10}
        self.type_names = {1: "Small", 2: "Large", 3: "Micro"}
        self.plastics = []
        
        # Create plastics at random positions
        for i in range(self.num_plastics):
            plastic_type = random.choice([1, 2, 3])
            
            while True:
                pos = (random.randint(0, self.grid_size-1), 
                       random.randint(0, self.grid_size-1))
                if pos != self.agent_pos and pos not in [(p[0], p[1]) for p in self.plastics]:
                    self.plastics.append((pos[0], pos[1], plastic_type))
                    self.grid[pos] = plastic_type
                    break
        
        self.grid[self.agent_pos] = 4  # agent
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        return self.grid.copy()
    
    def _get_info(self):
        return {
            "collected": self.collected,
            "total": self.num_plastics,
            "steps": self.steps,
            "remaining": self.num_plastics - self.collected
        }
    
    def step(self, action):
        self.steps += 1
        reward = 0
        done = False
        truncated = False
        
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        # Execute action
        if action == 0:  # up
            new_x = max(0, x - 1)
        elif action == 1:  # down
            new_x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # left
            new_y = max(0, y - 1)
        elif action == 3:  # right
            new_y = min(self.grid_size - 1, y + 1)
        elif action == 4:  # collect
            # Check if agent is on plastic
            for i, (px, py, ptype) in enumerate(self.plastics):
                if px == x and py == y:
                    reward += self.reward_map[ptype]
                    self.collected += 1
                    self.grid[x, y] = 0
                    self.plastics.pop(i)
                    print(f"    🟢 Collected {self.type_names[ptype]} plastic! +{self.reward_map[ptype]}")
                    break
        
        # Move agent if action was movement
        if action in [0, 1, 2, 3]:
            # Clear old position
            if self.grid[x, y] == 4:
                self.grid[x, y] = 0
            
            # Move to new position
            self.agent_pos = (new_x, new_y)
            self.grid[self.agent_pos] = 4
            
            # Step penalty
            reward += -0.5
            
            # Auto-collect if stepping on plastic
            for i, (px, py, ptype) in enumerate(self.plastics):
                if px == new_x and py == new_y:
                    reward += self.reward_map[ptype]
                    self.collected += 1
                    self.plastics.pop(i)
                    print(f"    🟢 Collected {self.type_names[ptype]} plastic! +{self.reward_map[ptype]}")
                    break
        
        # Check if all plastic collected
        if self.collected >= self.num_plastics:
            done = True
            reward += 100
            print(f"    🎉 ALL PLASTIC COLLECTED! Bonus +100 🎉")
        
        if self.steps >= 200:
            truncated = True
        
        return self._get_obs(), reward, done, truncated, self._get_info()
    
    def render(self):
        pass
    
    def close(self):
        pass