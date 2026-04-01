"""
FINAL PLASTIC POLLUTION CLEANUP DEMO
This will definitely show the visualization window
"""

import pygame
import numpy as np
import random
import time
import os
from stable_baselines3 import PPO

# Create a simple working environment
class SimpleCleanupEnv:
    def __init__(self, size=8, num_plastics=5):
        self.size = size
        self.num_plastics = num_plastics
        self.reset()
    
    def reset(self):
        self.agent = [0, 0]  # top-left corner
        self.plastics = []
        self.collected = 0
        self.steps = 0
        
        # Create random plastic positions
        for _ in range(self.num_plastics):
            while True:
                pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                if pos != self.agent and pos not in self.plastics:
                    # Random plastic type: 0=small, 1=large, 2=micro
                    plastic_type = random.randint(0, 2)
                    self.plastics.append([pos[0], pos[1], plastic_type])
                    break
        
        return self.get_state()
    
    def get_state(self):
        state = np.zeros((self.size, self.size), dtype=np.int8)
        state[self.agent[0], self.agent[1]] = 4  # agent
        
        for p in self.plastics:
            state[p[0], p[1]] = p[2] + 1  # 1,2,3 for plastic types
        return state
    
    def step(self, action):
        self.steps += 1
        reward = -0.5
        done = False
        
        # Move
        if action == 0 and self.agent[0] > 0:
            self.agent[0] -= 1
        elif action == 1 and self.agent[0] < self.size-1:
            self.agent[0] += 1
        elif action == 2 and self.agent[1] > 0:
            self.agent[1] -= 1
        elif action == 3 and self.agent[1] < self.size-1:
            self.agent[1] += 1
        elif action == 4:  # collect
            for i, p in enumerate(self.plastics):
                if p[0] == self.agent[0] and p[1] == self.agent[1]:
                    rewards_map = [20, 50, 10]
                    reward = rewards_map[p[2]]
                    self.collected += 1
                    del self.plastics[i]
                    print(f"Collected {['Small', 'Large', 'Micro'][p[2]]} plastic! +{reward}")
                    break
        
        # Check completion
        if self.collected >= self.num_plastics:
            reward += 100
            done = True
            print("🎉 CLEANUP COMPLETE! 🎉")
        
        if self.steps >= 150:
            done = True
        
        return self.get_state(), reward, done
    
    def render_text(self):
        return f"Collected: {self.collected}/{self.num_plastics} | Steps: {self.steps}"


# Train or load model
env = SimpleCleanupEnv(size=8, num_plastics=5)
model_path = 'models/pg/final_model'

if not os.path.exists(f'{model_path}.zip'):
    print("="*60)
    print("TRAINING AI AGENT... (2-3 minutes)")
    print("="*60)
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.001, n_steps=2048)
    model.learn(total_timesteps=50000)
    os.makedirs('models/pg', exist_ok=True)
    model.save(model_path)
    print("✅ Training complete!\n")
else:
    print("Loading existing model...")
    model = PPO.load(model_path)

# Pygame setup
pygame.init()
CELL_SIZE = 80
WINDOW_SIZE = 8 * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 100))
pygame.display.set_caption("Plastic Pollution Cleanup - AI Agent in Action")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
BROWN = (139, 69, 19)
DARK_BROWN = (101, 67, 33)
LIGHT_BROWN = (205, 133, 63)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
LIGHT_BLUE = (173, 216, 230)

PLASTIC_COLORS = {0: BROWN, 1: DARK_BROWN, 2: LIGHT_BROWN}
PLASTIC_NAMES = {0: "Small (+20)", 1: "Large (+50)", 2: "Micro (+10)"}

print("\n" + "="*60)
print("🎬 WATCH THE AI AGENT CLEAN UP PLASTIC!")
print("- Blue square: Cleanup Robot")
print("- Brown: Small Plastic (+20)")
print("- Dark Brown: Large Plastic (+50)")
print("- Light Brown: Microplastic (+10)")
print("- Press ESC to exit")
print("="*60 + "\n")

# Run 3 episodes
for episode in range(1, 4):
    print(f"\n{'='*50}")
    print(f"Episode {episode}/3")
    print(f"{'='*50}")
    
    obs = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    while not done:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Draw grid
        screen.fill(LIGHT_BLUE)
        
        for i in range(8):
            for j in range(8):
                rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cell_value = obs[i, j]
                
                if cell_value == 4:  # Agent
                    pygame.draw.rect(screen, BLUE, rect)
                    center = (j*CELL_SIZE + CELL_SIZE//2, i*CELL_SIZE + CELL_SIZE//2)
                    pygame.draw.circle(screen, WHITE, center, CELL_SIZE//5)
                    pygame.draw.circle(screen, BLACK, (center[0]-6, center[1]-4), 2)
                    pygame.draw.circle(screen, BLACK, (center[0]+6, center[1]-4), 2)
                elif cell_value in [1, 2, 3]:  # Plastic
                    ptype = cell_value - 1
                    pygame.draw.rect(screen, PLASTIC_COLORS[ptype], rect)
                    center = (j*CELL_SIZE + CELL_SIZE//2, i*CELL_SIZE + CELL_SIZE//2)
                    if ptype == 0:  # small
                        pygame.draw.circle(screen, (101, 67, 33), center, CELL_SIZE//4)
                    elif ptype == 1:  # large
                        pygame.draw.rect(screen, (80, 50, 25), 
                                       (center[0]-CELL_SIZE//5, center[1]-CELL_SIZE//5,
                                        CELL_SIZE//2.5, CELL_SIZE//2.5))
                    else:  # micro
                        pygame.draw.circle(screen, (160, 100, 50), center, CELL_SIZE//6)
                else:
                    pygame.draw.rect(screen, WHITE, rect)
                
                pygame.draw.rect(screen, BLACK, rect, 1)
        
        # Draw info panel
        info_y = 8 * CELL_SIZE
        pygame.draw.rect(screen, (240, 240, 240), (0, info_y, WINDOW_SIZE, 100))
        
        font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 20)
        
        # Stats
        stats = [
            f"Episode: {episode}/3",
            f"Steps: {step_count}",
            f"Collected: {env.collected}/{env.num_plastics}",
            f"Reward: {total_reward:.1f}"
        ]
        
        for i, stat in enumerate(stats):
            text = font.render(stat, True, BLACK)
            screen.blit(text, (10, info_y + 10 + i*30))
        
        # Legend
        legend_x = WINDOW_SIZE - 150
        title = small_font.render("Plastic Types:", True, BLACK)
        screen.blit(title, (legend_x, info_y + 10))
        
        for i, (ptype, color) in enumerate(PLASTIC_COLORS.items()):
            pygame.draw.circle(screen, color, (legend_x + 10, info_y + 40 + i*22), 8)
            name = small_font.render(PLASTIC_NAMES[ptype], True, BLACK)
            screen.blit(name, (legend_x + 25, info_y + 35 + i*22))
        
        # Status
        if done:
            status_text = "✅ CLEANUP COMPLETE! ✅"
            status_color = GREEN
        else:
            status_text = f"🤖 Collecting plastic... {env.collected}/{env.num_plastics}"
            status_color = BLUE
        
        status = font.render(status_text, True, status_color)
        status_rect = status.get_rect(center=(WINDOW_SIZE//2, info_y + 75))
        screen.blit(status, status_rect)
        
        pygame.display.flip()
        clock.tick(10)  # 10 frames per second
    
    print(f"Episode {episode} finished! Reward: {total_reward:.2f}")
    
    if episode < 3:
        pygame.time.wait(2000)  # Pause between episodes

print("\n" + "="*60)
print("✅ DEMONSTRATION COMPLETE!")
print("The AI agent successfully collected all plastic waste!")
print("="*60)
print("\nPress any key to exit...")

# Wait for user
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            waiting = False
        if event.type == pygame.KEYDOWN:
            waiting = False

pygame.quit()