"""
Working Plastic Pollution Environment
Compatible with Stable-Baselines3 (Gymnasium)
Small (+20), Large (+50), Micro (+10) plastics
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class PlasticPollutionEnv(gym.Env):
    """Working environment with multiple plastic types"""
    
    def __init__(self, grid_size=8, num_plastics=5):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_plastics = num_plastics
        
        # Action: 0=up, 1=down, 2=left, 3=right, 4=collect
        self.action_space = spaces.Discrete(5)
        
        # Observation: grid with values 0-4
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(grid_size, grid_size), dtype=np.int8
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent_pos = [0, 0]  # Start at top-left
        self.steps = 0
        self.collected = 0
        
        # Create grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Plastic positions with types (0=small+20, 1=large+50, 2=micro+10)
        self.plastics = []
        self.plastic_rewards = [20, 50, 10]
        self.plastic_names = ["Small", "Large", "Micro"]
        
        for _ in range(self.num_plastics):
            while True:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                if [x, y] != self.agent_pos and [x, y] not in [[p[0], p[1]] for p in self.plastics]:
                    plastic_type = random.randint(0, 2)
                    self.plastics.append([x, y, plastic_type])
                    self.grid[x, y] = plastic_type + 1  # 1,2,3 for plastic types
                    break
        
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 4  # Agent
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        return self.grid.copy()
    
    def _get_info(self):
        return {
            "collected": self.collected,
            "total": self.num_plastics,
            "steps": self.steps
        }
    
    def step(self, action):
        self.steps += 1
        reward = -0.5  # Step penalty
        done = False
        truncated = False
        
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        # Movement
        if action == 0:  # up
            new_x = max(0, x - 1)
        elif action == 1:  # down
            new_x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # left
            new_y = max(0, y - 1)
        elif action == 3:  # right
            new_y = min(self.grid_size - 1, y + 1)
        elif action == 4:  # collect
            # Check if on plastic
            for i, p in enumerate(self.plastics):
                if p[0] == x and p[1] == y:
                    reward += self.plastic_rewards[p[2]]
                    self.collected += 1
                    self.grid[x, y] = 0
                    del self.plastics[i]
                    print(f"    🟢 Collected {self.plastic_names[p[2]]} plastic! +{self.plastic_rewards[p[2]]}")
                    break
        
        # Move if action was movement
        if action in [0, 1, 2, 3]:
            # Clear old position
            if self.grid[x, y] == 4:
                self.grid[x, y] = 0
            
            self.agent_pos = [new_x, new_y]
            self.grid[new_x, new_y] = 4
            
            # Auto-collect if stepping on plastic
            for i, p in enumerate(self.plastics):
                if p[0] == new_x and p[1] == new_y:
                    reward += self.plastic_rewards[p[2]]
                    self.collected += 1
                    self.grid[new_x, new_y] = 4
                    del self.plastics[i]
                    print(f"    🟢 Collected {self.plastic_names[p[2]]} plastic! +{self.plastic_rewards[p[2]]}")
                    break
        
        # Check completion
        if self.collected >= self.num_plastics:
            reward += 100
            done = True
            print(f"    🎉 ALL PLASTIC COLLECTED! Bonus +100 🎉")
        
        if self.steps >= 200:
            truncated = True
        
        return self._get_obs(), reward, done, truncated, self._get_info()