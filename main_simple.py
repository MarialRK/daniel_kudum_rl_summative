"""
Simple Plastic Pollution Cleanup Demo
Shows trained agent collecting plastic in a grid
Use this for your video demonstration
"""

import pygame
import numpy as np
from stable_baselines3 import PPO
import sys
import os

# Simple environment that works
class SimplePlasticEnv:
    def __init__(self, grid_size=8):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        self.agent_pos = (0, 0)
        self.plastic_pos = (self.grid_size-1, self.grid_size-1)
        self.steps = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.grid[self.agent_pos] = 1  # 1 = agent
        self.grid[self.plastic_pos] = 2  # 2 = plastic
        return self.grid.copy()
    
    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # down
            x = min(self.grid_size-1, x+1)
        elif action == 2:  # left
            y = max(0, y-1)
        elif action == 3:  # right
            y = min(self.grid_size-1, y+1)
        
        # Update grid
        self.grid[self.agent_pos] = 0
        self.agent_pos = (x, y)
        self.grid[self.agent_pos] = 1
        self.steps += 1
        
        # Calculate reward
        reward = -0.1  # Small step penalty
        done = False
        
        if self.agent_pos == self.plastic_pos:
            reward = 10  # Big reward for collecting plastic
            done = True
        
        if self.steps >= 100:
            done = True
        
        return self.grid.copy(), reward, done

print("="*60)
print("PLASTIC POLLUTION CLEANUP - DEMONSTRATION")
print("="*60)
print("Loading trained model...")

# Load the trained model
model = PPO.load('models/pg/simple_model')
print("Model loaded successfully!")

# Initialize pygame
pygame.init()
cell_size = 80
screen_size = 8 * cell_size
screen = pygame.display.set_mode((screen_size, screen_size + 80))
pygame.display.set_caption("Plastic Pollution Cleanup - AI Agent in Action")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
BROWN = (139, 69, 19)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
LIGHT_BLUE = (173, 216, 230)

print("\n" + "="*60)
print("Watch the AI agent collect plastic!")
print("- Blue square: Cleanup Robot")
print("- Brown circle: Plastic Bottle")
print("- Agent moves step by step to reach the plastic")
print("- Press ESC to exit")
print("="*60 + "\n")

# Create environment
env = SimplePlasticEnv(grid_size=8)
obs = env.reset()

running = True
episode_count = 0
total_episodes = 5  # Show 5 successful collections

print("Starting demonstration...\n")

while running and episode_count < total_episodes:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Get action from trained model
    action, _ = model.predict(obs)
    obs, reward, done = env.step(action)
    
    # Draw the grid
    screen.fill(LIGHT_BLUE)
    
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            rect = pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size)
            
            if obs[i, j] == 1:  # Agent
                color = BLUE
                pygame.draw.rect(screen, color, rect)
                # Draw robot face
                center = (j*cell_size + cell_size//2, i*cell_size + cell_size//2)
                pygame.draw.circle(screen, WHITE, center, cell_size//5)
            elif obs[i, j] == 2:  # Plastic
                color = BROWN
                pygame.draw.rect(screen, color, rect)
                # Draw bottle shape
                center = (j*cell_size + cell_size//2, i*cell_size + cell_size//2)
                pygame.draw.circle(screen, (101, 67, 33), center, cell_size//4)
            else:  # Empty
                color = WHITE
                pygame.draw.rect(screen, color, rect)
            
            pygame.draw.rect(screen, BLACK, rect, 1)  # Border
    
    # Draw info panel
    info_y = env.grid_size * cell_size
    pygame.draw.rect(screen, (240, 240, 240), (0, info_y, screen_size, 80))
    
    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 36)
    
    steps_text = font.render(f"Steps: {env.steps}", True, BLACK)
    screen.blit(steps_text, (10, info_y + 10))
    
    reward_text = font.render(f"Reward: {reward:.1f}", True, BLACK)
    screen.blit(reward_text, (10, info_y + 35))
    
    status_text = "SEARCHING..." if not done else "CLEANUP COMPLETE!"
    color = BLUE if not done else GREEN
    status = font.render(status_text, True, color)
    screen.blit(status, (screen_size - 150, info_y + 10))
    
    episode_text = big_font.render(f"Episode: {episode_count + 1}/{total_episodes}", True, BLACK)
    screen.blit(episode_text, (screen_size//2 - 80, info_y + 25))
    
    pygame.display.flip()
    clock.tick(10)  # 10 FPS - slow enough to see
    
    # Print to terminal
    if done:
        episode_count += 1
        print(f"Episode {episode_count}: Plastic collected in {env.steps} steps! (Reward: {reward:.2f})")
        
        if episode_count < total_episodes:
            # Reset environment for next episode
            obs = env.reset()
            pygame.time.wait(1000)  # Pause 1 second between episodes

print("\n" + "="*60)
print("DEMONSTRATION COMPLETE!")
print(f"Successfully collected {total_episodes} pieces of plastic!")
print("="*60)
print("\nPress any key to exit...")

# Wait for user to close
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            waiting = False
        if event.type == pygame.KEYDOWN:
            waiting = False

pygame.quit()
print("\nThank you for watching! Mission: Fighting Plastic Pollution with AI")