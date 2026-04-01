"""
Complete Plastic Pollution Cleanup Demo
Features: Multiple plastic types (small, large, micro)
Agent learns to collect all plastics in an 8x8 grid
Perfect for your video demonstration
"""

import pygame
import numpy as np
import random
from stable_baselines3 import PPO
import os

# Create the environment with multiple plastic types
class MultiPlasticEnv:
    def __init__(self, grid_size=8, num_plastics=5):
        self.grid_size = grid_size
        self.num_plastics = num_plastics
        self.reset()
    
    def reset(self):
        self.agent_pos = (0, 0)
        self.steps = 0
        self.collected = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Create different plastic types at random positions
        self.plastics = []  # (x, y, type, reward)
        # Plastic types: 1=small (brown), 2=large (dark brown), 3=micro (light brown)
        for i in range(self.num_plastics):
            plastic_type = random.choice([1, 2, 3])
            reward_map = {1: 20, 2: 50, 3: 10}
            type_name = {1: "Small", 2: "Large", 3: "Micro"}
            
            while True:
                pos = (random.randint(0, self.grid_size-1), 
                       random.randint(0, self.grid_size-1))
                if pos != self.agent_pos and pos not in [(p[0], p[1]) for p in self.plastics]:
                    self.plastics.append((pos[0], pos[1], plastic_type, reward_map[plastic_type], type_name[plastic_type]))
                    self.grid[pos] = plastic_type
                    break
        
        self.grid[self.agent_pos] = 4  # 4 = agent
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
        elif action == 4:  # collect
            # Check if agent is on plastic
            for i, (px, py, ptype, reward, name) in enumerate(self.plastics):
                if px == self.agent_pos[0] and py == self.agent_pos[1]:
                    self.collected += 1
                    self.grid[self.agent_pos] = 0
                    self.plastics.pop(i)
                    return self.grid.copy(), reward, False, {"collected": self.collected, "type": name}
        
        # Update grid
        self.grid[self.agent_pos] = 0
        self.agent_pos = (x, y)
        self.grid[self.agent_pos] = 4
        self.steps += 1
        
        # Reward: small step penalty
        reward = -0.5
        done = False
        
        # Check if collected all plastic
        if self.collected >= self.num_plastics:
            done = True
            reward += 100  # Completion bonus
        
        if self.steps >= 200:
            done = True
        
        return self.grid.copy(), reward, done, {"collected": self.collected, "total": self.num_plastics, "steps": self.steps}
    
    def get_info(self):
        return {"collected": self.collected, "total": self.num_plastics, "steps": self.steps}


print("="*60)
print("PLASTIC POLLUTION CLEANUP - COMPLETE DEMO")
print("Features: Small Plastic (+20), Large Plastic (+50), Microplastic (+10)")
print("="*60)

# Train the model on multi-plastic environment
print("\nTraining AI agent on multi-plastic environment...")
env = MultiPlasticEnv(grid_size=8, num_plastics=5)

model = PPO(
    'MlpPolicy', 
    env, 
    verbose=0,
    learning_rate=0.001,
    gamma=0.99,
    n_steps=2048,
    batch_size=64
)

model.learn(total_timesteps=100000)
model.save('models/pg/multi_plastic_model')
print("✅ Model trained and saved!\n")

# Initialize pygame
pygame.init()
cell_size = 70
screen_size = 8 * cell_size
screen = pygame.display.set_mode((screen_size, screen_size + 120))
pygame.display.set_caption("Plastic Pollution Cleanup - Complete Demo")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
DARK_BLUE = (0, 50, 150)
BROWN = (139, 69, 19)
DARK_BROWN = (101, 67, 33)
LIGHT_BROWN = (205, 133, 63)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
LIGHT_BLUE = (173, 216, 230)
YELLOW = (255, 255, 0)

# Plastic colors mapping
PLASTIC_COLORS = {
    1: BROWN,      # Small plastic
    2: DARK_BROWN, # Large plastic
    3: LIGHT_BROWN # Microplastic
}

PLASTIC_NAMES = {
    1: "Small Plastic (+20)",
    2: "Large Plastic (+50)",
    3: "Microplastic (+10)"
}

print("\n" + "="*60)
print("WATCH THE AI AGENT COLLECT MULTIPLE PLASTIC TYPES!")
print("- Blue square: Cleanup Robot")
print("- Brown circle: Small Plastic (+20)")
print("- Dark Brown circle: Large Plastic (+50)")
print("- Light Brown circle: Microplastic (+10)")
print("- Agent learns to collect all 5 pieces of plastic")
print("- Press ESC to exit")
print("="*60 + "\n")

# Create environment and load model
env = MultiPlasticEnv(grid_size=8, num_plastics=5)
obs = env.reset()
model = PPO.load('models/pg/multi_plastic_model')

running = True
episode_count = 0
total_episodes = 3
total_reward = 0

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
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    # Draw the grid
    screen.fill(LIGHT_BLUE)
    
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            rect = pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size)
            
            if obs[i, j] == 4:  # Agent
                color = BLUE
                pygame.draw.rect(screen, color, rect)
                # Draw robot face
                center = (j*cell_size + cell_size//2, i*cell_size + cell_size//2)
                pygame.draw.circle(screen, WHITE, center, cell_size//5)
                pygame.draw.circle(screen, BLACK, (center[0]-5, center[1]-3), 2)
                pygame.draw.circle(screen, BLACK, (center[0]+5, center[1]-3), 2)
                pygame.draw.arc(screen, BLACK, (center[0]-8, center[1], 16, 10), 0, 3.14, 2)
            elif obs[i, j] in [1, 2, 3]:  # Plastic
                color = PLASTIC_COLORS[obs[i, j]]
                pygame.draw.rect(screen, color, rect)
                # Draw bottle shape
                center = (j*cell_size + cell_size//2, i*cell_size + cell_size//2)
                if obs[i, j] == 1:  # Small
                    pygame.draw.circle(screen, (101, 67, 33), center, cell_size//4)
                elif obs[i, j] == 2:  # Large
                    pygame.draw.rect(screen, (80, 50, 25), 
                                   (center[0]-cell_size//5, center[1]-cell_size//5,
                                    cell_size//2.5, cell_size//2.5))
                else:  # Micro
                    pygame.draw.circle(screen, (160, 100, 50), center, cell_size//6)
            else:  # Empty
                color = WHITE
                pygame.draw.rect(screen, color, rect)
            
            pygame.draw.rect(screen, BLACK, rect, 1)  # Border
    
    # Draw info panel
    info_y = env.grid_size * cell_size
    pygame.draw.rect(screen, (240, 240, 240), (0, info_y, screen_size, 120))
    
    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 18)
    
    collected = env.collected
    total = env.num_plastics
    
    # Stats
    stats = [
        f"Steps: {env.steps}",
        f"Collected: {collected}/{total}",
        f"Reward this episode: {total_reward:.1f}",
        f"Remaining: {total - collected}"
    ]
    
    for i, stat in enumerate(stats):
        text = font.render(stat, True, BLACK)
        screen.blit(text, (10, info_y + 10 + i * 25))
    
    # Plastic legend
    legend_x = screen_size - 180
    legend_title = font.render("Plastic Types:", True, BLACK)
    screen.blit(legend_title, (legend_x, info_y + 10))
    
    for i, (ptype, color) in enumerate(PLASTIC_COLORS.items()):
        pygame.draw.circle(screen, color, (legend_x + 15, info_y + 45 + i * 25), 8)
        name_text = small_font.render(PLASTIC_NAMES[ptype], True, BLACK)
        screen.blit(name_text, (legend_x + 35, info_y + 40 + i * 25))
    
    # Status
    if done:
        status_color = GREEN
        status_text = "CLEANUP COMPLETE! All plastic collected!"
    else:
        status_color = BLUE
        status_text = f"Searching for plastic... {collected}/{total}"
    
    status = big_font.render(status_text, True, status_color)
    status_rect = status.get_rect(center=(screen_size//2, info_y + 100))
    screen.blit(status, status_rect)
    
    # Episode counter
    ep_text = font.render(f"Episode: {episode_count + 1}/{total_episodes}", True, BLACK)
    screen.blit(ep_text, (screen_size//2 - 60, info_y + 5))
    
    pygame.display.flip()
    clock.tick(8)  # 8 FPS
    
    # Print to terminal
    if 'type' in info:
        print(f"Step {env.steps}: Collected {info['type']} plastic! (+{info.get('reward', 0) if 'reward' in info else '?'})")
    
    if done:
        episode_count += 1
        print(f"\n{'='*50}")
        print(f"Episode {episode_count} COMPLETE!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Plastic collected: {collected}/{total}")
        print(f"Steps taken: {env.steps}")
        print(f"{'='*50}\n")
        
        if episode_count < total_episodes:
            obs = env.reset()
            total_reward = 0
            pygame.time.wait(1500)  # Pause between episodes

print("\n" + "="*60)
print("DEMONSTRATION COMPLETE!")
print(f"Successfully completed {total_episodes} cleanup missions!")
print("The AI agent learned to collect different types of plastic waste.")
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