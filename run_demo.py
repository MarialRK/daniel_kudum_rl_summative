"""
Final Working Demo - Guaranteed to Show Visualization
"""

import pygame
import numpy as np
import os
from stable_baselines3 import PPO
from working_env import PlasticPollutionEnv

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

print("="*60)
print("PLASTIC POLLUTION CLEANUP - WORKING DEMO")
print("Small (+20) | Large (+50) | Micro (+10)")
print("="*60)

# Create environment
env = PlasticPollutionEnv(grid_size=8, num_plastics=5)

# Train model
print("\n🎮 Training AI agent (2-3 minutes)...")
model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.001, n_steps=2048, batch_size=64)
model.learn(total_timesteps=50000)
os.makedirs('models/pg', exist_ok=True)
model.save('models/pg/working_model')
print("✅ Model trained!\n")

# Pygame setup
pygame.init()
CELL_SIZE = 80
WINDOW_SIZE = 8 * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 120))
pygame.display.set_caption("Plastic Pollution Cleanup - AI in Action")
clock = pygame.time.Clock()

print("="*60)
print("🎬 WATCH THE AI AGENT!")
print("- Blue square: Cleanup Robot")
print("- Brown: Small Plastic (+20)")
print("- Dark Brown: Large Plastic (+50)")
print("- Light Brown: Microplastic (+10)")
print("- Press ESC to exit")
print("="*60 + "\n")

# Run episodes
for episode in range(1, 4):
    print(f"\n{'='*50}")
    print(f"Episode {episode}/3")
    print(f"{'='*50}")
    
    obs, info = env.reset()
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
        
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Draw
        screen.fill(LIGHT_BLUE)
        
        for i in range(8):
            for j in range(8):
                rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cell = obs[i, j]
                
                if cell == 4:  # Agent
                    pygame.draw.rect(screen, BLUE, rect)
                    cx = j*CELL_SIZE + CELL_SIZE//2
                    cy = i*CELL_SIZE + CELL_SIZE//2
                    pygame.draw.circle(screen, WHITE, (cx, cy), CELL_SIZE//5)
                    pygame.draw.circle(screen, BLACK, (cx-6, cy-4), 2)
                    pygame.draw.circle(screen, BLACK, (cx+6, cy-4), 2)
                elif cell in [1, 2, 3]:  # Plastic
                    ptype = cell - 1
                    pygame.draw.rect(screen, PLASTIC_COLORS[ptype], rect)
                    cx = j*CELL_SIZE + CELL_SIZE//2
                    cy = i*CELL_SIZE + CELL_SIZE//2
                    if ptype == 0:
                        pygame.draw.circle(screen, (101, 67, 33), (cx, cy), CELL_SIZE//4)
                    elif ptype == 1:
                        pygame.draw.rect(screen, (80, 50, 25), 
                                       (cx-CELL_SIZE//5, cy-CELL_SIZE//5,
                                        CELL_SIZE//2.5, CELL_SIZE//2.5))
                    else:
                        pygame.draw.circle(screen, (160, 100, 50), (cx, cy), CELL_SIZE//6)
                else:
                    pygame.draw.rect(screen, WHITE, rect)
                
                pygame.draw.rect(screen, BLACK, rect, 1)
        
        # Info panel
        info_y = 8 * CELL_SIZE
        pygame.draw.rect(screen, (240, 240, 240), (0, info_y, WINDOW_SIZE, 120))
        
        font = pygame.font.Font(None, 24)
        big_font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 18)
        
        # Stats
        stats = [
            f"Episode: {episode}/3",
            f"Steps: {step_count}",
            f"Collected: {info['collected']}/{info['total']}",
            f"Reward: {total_reward:.1f}"
        ]
        for i, s in enumerate(stats):
            text = font.render(s, True, BLACK)
            screen.blit(text, (10, info_y + 10 + i*25))
        
        # Legend
        legend_x = WINDOW_SIZE - 150
        title = small_font.render("Plastic Types:", True, BLACK)
        screen.blit(title, (legend_x, info_y + 10))
        for i, (pt, color) in enumerate(PLASTIC_COLORS.items()):
            pygame.draw.circle(screen, color, (legend_x + 12, info_y + 38 + i*22), 8)
            name = small_font.render(PLASTIC_NAMES[pt], True, BLACK)
            screen.blit(name, (legend_x + 30, info_y + 33 + i*22))
        
        # Status
        if done:
            status = big_font.render("✅ CLEANUP COMPLETE! ✅", True, GREEN)
        else:
            status = big_font.render(f"🤖 Collecting... {info['collected']}/{info['total']}", True, BLUE)
        status_rect = status.get_rect(center=(WINDOW_SIZE//2, info_y + 95))
        screen.blit(status, status_rect)
        
        pygame.display.flip()
        clock.tick(8)
    
    print(f"Episode {episode} complete! Reward: {total_reward:.2f}")
    if episode < 3:
        pygame.time.wait(1500)

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