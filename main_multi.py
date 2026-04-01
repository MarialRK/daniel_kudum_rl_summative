"""
Multi-Plastic Pollution Cleanup Demo
Small (+20), Large (+50), Micro (+10) plastics
Agent learns to collect all plastics in an 8x8 grid
"""

import pygame
import numpy as np
import os
from stable_baselines3 import PPO
from multi_plastic_env import MultiPlasticEnv

print("="*60)
print("PLASTIC POLLUTION CLEANUP - MULTI-PLASTIC DEMO")
print("Small Plastic (+20) | Large Plastic (+50) | Microplastic (+10)")
print("="*60)

# Create environment
env = MultiPlasticEnv(grid_size=8, num_plastics=5)

# Train the model
print("\n🎮 Training AI agent (2-3 minutes)...")
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=0,
    learning_rate=0.001,
    gamma=0.99,
    n_steps=2048,
    batch_size=64
)

model.learn(total_timesteps=50000)
os.makedirs('models/pg', exist_ok=True)
model.save('models/pg/multi_plastic_model')
print("✅ Model trained and saved!\n")

# Initialize pygame
pygame.init()
cell_size = 70
screen_size = 8 * cell_size
screen = pygame.display.set_mode((screen_size, screen_size + 140))
pygame.display.set_caption("Plastic Pollution Cleanup - Multi-Plastic Demo")
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

# Plastic mapping
PLASTIC_COLORS = {1: BROWN, 2: DARK_BROWN, 3: LIGHT_BROWN}
PLASTIC_NAMES = {1: "Small", 2: "Large", 3: "Micro"}
PLASTIC_REWARDS = {1: "+20", 2: "+50", 3: "+10"}

print("="*60)
print("🎬 WATCH THE AI AGENT CLEAN UP PLASTIC!")
print("- Blue square: Cleanup Robot")
print("- Brown: Small Plastic (+20)")
print("- Dark Brown: Large Plastic (+50)")
print("- Light Brown: Microplastic (+10)")
print("- Agent must collect all 5 pieces")
print("- Press ESC to exit")
print("="*60 + "\n")

# Test the trained model
obs, info = env.reset()
model = PPO.load('models/pg/multi_plastic_model')

running = True
episode_count = 0
total_episodes = 3
episode_reward = 0

print("🎯 Starting demonstration...\n")

while running and episode_count < total_episodes:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Get action from trained model
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    
    # Draw the grid
    screen.fill(LIGHT_BLUE)
    
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            rect = pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size)
            
            if obs[i, j] == 4:  # Agent
                pygame.draw.rect(screen, BLUE, rect)
                center = (j*cell_size + cell_size//2, i*cell_size + cell_size//2)
                pygame.draw.circle(screen, WHITE, center, cell_size//5)
                pygame.draw.circle(screen, BLACK, (center[0]-5, center[1]-3), 2)
                pygame.draw.circle(screen, BLACK, (center[0]+5, center[1]-3), 2)
            elif obs[i, j] in [1, 2, 3]:  # Plastic
                pygame.draw.rect(screen, PLASTIC_COLORS[obs[i, j]], rect)
                center = (j*cell_size + cell_size//2, i*cell_size + cell_size//2)
                if obs[i, j] == 1:
                    pygame.draw.circle(screen, (101, 67, 33), center, cell_size//4)
                elif obs[i, j] == 2:
                    pygame.draw.rect(screen, (80, 50, 25), 
                                   (center[0]-cell_size//5, center[1]-cell_size//5,
                                    cell_size//2.5, cell_size//2.5))
                else:
                    pygame.draw.circle(screen, (160, 100, 50), center, cell_size//6)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            
            pygame.draw.rect(screen, BLACK, rect, 1)
    
    # Draw info panel
    info_y = env.grid_size * cell_size
    pygame.draw.rect(screen, (240, 240, 240), (0, info_y, screen_size, 140))
    
    font = pygame.font.Font(None, 22)
    big_font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 16)
    
    collected = info['collected']
    total = info['total']
    
    # Stats
    stats = [
        f"Steps: {info['steps']}",
        f"Collected: {collected}/{total}",
        f"Remaining: {total - collected}",
        f"Episode Reward: {episode_reward:.1f}"
    ]
    
    for i, stat in enumerate(stats):
        text = font.render(stat, True, BLACK)
        screen.blit(text, (10, info_y + 10 + i * 22))
    
    # Legend
    legend_x = screen_size - 160
    legend_title = font.render("Plastic Types:", True, BLACK)
    screen.blit(legend_title, (legend_x, info_y + 10))
    
    for i, (ptype, color) in enumerate(PLASTIC_COLORS.items()):
        pygame.draw.circle(screen, color, (legend_x + 12, info_y + 45 + i * 22), 8)
        name_text = small_font.render(f"{PLASTIC_NAMES[ptype]} {PLASTIC_REWARDS[ptype]}", True, BLACK)
        screen.blit(name_text, (legend_x + 28, info_y + 40 + i * 22))
    
    # Status
    if done:
        status_color = GREEN
        status_text = f"🎉 CLEANUP COMPLETE! All {total} plastics collected! 🎉"
    else:
        status_color = BLUE
        status_text = f"🤖 Searching... {collected}/{total} collected"
    
    status = big_font.render(status_text, True, status_color)
    status_rect = status.get_rect(center=(screen_size//2, info_y + 115))
    screen.blit(status, status_rect)
    
    ep_text = font.render(f"Episode: {episode_count + 1}/{total_episodes}", True, BLACK)
    screen.blit(ep_text, (screen_size//2 - 60, info_y + 5))
    
    pygame.display.flip()
    clock.tick(8)
    
    if done:
        episode_count += 1
        print(f"\n{'='*50}")
        print(f"Episode {episode_count} COMPLETE!")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Plastic collected: {collected}/{total}")
        print(f"Steps: {info['steps']}")
        print(f"{'='*50}\n")
        
        if episode_count < total_episodes:
            obs, info = env.reset()
            episode_reward = 0
            pygame.time.wait(2000)

print("\n" + "="*60)
print("✅ DEMONSTRATION COMPLETE!")
print(f"Successfully completed {total_episodes} cleanup missions!")
print("The AI agent collected different plastic types:")
print("  - Small Plastic: +20 points")
print("  - Large Plastic: +50 points")
print("  - Microplastic: +10 points")
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
print("\n🌊 Thank you for watching! Mission: Fighting Plastic Pollution with AI")