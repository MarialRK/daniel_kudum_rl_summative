"""
SIMPLE WORKING DEMO - No training needed
Uses a simple hardcoded pathfinding agent that collects all plastics
Perfect for your video demonstration
"""

import pygame
import random
import math

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
PLASTIC_REWARDS = {0: 20, 1: 50, 2: 10}

class SimpleAgent:
    """Simple AI that finds the closest plastic and moves toward it"""
    
    def __init__(self):
        self.target = None
    
    def get_action(self, agent_pos, plastics, grid_size):
        """Find closest plastic and move toward it"""
        if not plastics:
            return 4  # Collect if at plastic (but no plastics left)
        
        # Find closest plastic
        closest = None
        closest_dist = float('inf')
        
        for p in plastics:
            dist = abs(agent_pos[0] - p[0]) + abs(agent_pos[1] - p[1])
            if dist < closest_dist:
                closest_dist = dist
                closest = p
        
        if closest is None:
            return 4
        
        # Move toward closest plastic
        if agent_pos[0] < closest[0]:
            return 1  # down
        elif agent_pos[0] > closest[0]:
            return 0  # up
        elif agent_pos[1] < closest[1]:
            return 3  # right
        elif agent_pos[1] > closest[1]:
            return 2  # left
        else:
            return 4  # collect


print("="*60)
print("PLASTIC POLLUTION CLEANUP - AI DEMONSTRATION")
print("Small (+20) | Large (+50) | Micro (+10)")
print("="*60)

# Pygame setup
pygame.init()
CELL_SIZE = 80
GRID_SIZE = 8
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 120))
pygame.display.set_caption("Plastic Pollution Cleanup - AI Agent")
clock = pygame.time.Clock()

# Create environment
class Environment:
    def __init__(self, size=8, num_plastics=5):
        self.size = size
        self.num_plastics = num_plastics
        self.reset()
    
    def reset(self):
        self.agent = [0, 0]
        self.plastics = []
        self.collected = 0
        self.steps = 0
        self.total_reward = 0
        
        # Create plastics
        for _ in range(self.num_plastics):
            while True:
                x = random.randint(0, self.size-1)
                y = random.randint(0, self.size-1)
                if [x, y] != self.agent and [x, y] not in [[p[0], p[1]] for p in self.plastics]:
                    plastic_type = random.randint(0, 2)
                    self.plastics.append([x, y, plastic_type])
                    break
        
        return self.get_state()
    
    def get_state(self):
        state = [[0 for _ in range(self.size)] for _ in range(self.size)]
        state[self.agent[0]][self.agent[1]] = 4  # agent
        for p in self.plastics:
            state[p[0]][p[1]] = p[2] + 1  # 1,2,3 for plastics
        return state
    
    def step(self, action):
        self.steps += 1
        reward = -0.5
        done = False
        
        # Move
        if action == 0 and self.agent[0] > 0:  # up
            self.agent[0] -= 1
        elif action == 1 and self.agent[0] < self.size-1:  # down
            self.agent[0] += 1
        elif action == 2 and self.agent[1] > 0:  # left
            self.agent[1] -= 1
        elif action == 3 and self.agent[1] < self.size-1:  # right
            self.agent[1] += 1
        elif action == 4:  # collect
            for i, p in enumerate(self.plastics):
                if p[0] == self.agent[0] and p[1] == self.agent[1]:
                    reward += PLASTIC_REWARDS[p[2]]
                    self.collected += 1
                    del self.plastics[i]
                    print(f"  🟢 Collected {PLASTIC_NAMES[p[2]]}! +{PLASTIC_REWARDS[p[2]]}")
                    break
        
        # Check if on plastic after movement
        for i, p in enumerate(self.plastics):
            if p[0] == self.agent[0] and p[1] == self.agent[1]:
                reward += PLASTIC_REWARDS[p[2]]
                self.collected += 1
                del self.plastics[i]
                print(f"  🟢 Collected {PLASTIC_NAMES[p[2]]}! +{PLASTIC_REWARDS[p[2]]}")
                break
        
        self.total_reward += reward
        
        if self.collected >= self.num_plastics:
            reward += 100
            done = True
            print(f"  🎉 ALL {self.num_plastics} PLASTICS COLLECTED! +100 BONUS! 🎉")
        
        if self.steps >= 200:
            done = True
        
        return self.get_state(), reward, done

# Create environment and agent
env = Environment(size=8, num_plastics=5)
agent = SimpleAgent()

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
    
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < 200:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
        
        # Get action from simple AI
        action = agent.get_action(env.agent, env.plastics, GRID_SIZE)
        state, reward, done = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Draw
        screen.fill(LIGHT_BLUE)
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cell = state[i][j]
                
                if cell == 4:  # Agent
                    pygame.draw.rect(screen, BLUE, rect)
                    cx = j*CELL_SIZE + CELL_SIZE//2
                    cy = i*CELL_SIZE + CELL_SIZE//2
                    pygame.draw.circle(screen, WHITE, (cx, cy), CELL_SIZE//5)
                    pygame.draw.circle(screen, BLACK, (cx-6, cy-4), 2)
                    pygame.draw.circle(screen, BLACK, (cx+6, cy-4), 2)
                    # Smile
                    pygame.draw.arc(screen, BLACK, (cx-10, cy-2, 20, 12), 0, 3.14, 2)
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
        info_y = GRID_SIZE * CELL_SIZE
        pygame.draw.rect(screen, (240, 240, 240), (0, info_y, WINDOW_SIZE, 120))
        
        font = pygame.font.Font(None, 24)
        big_font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 18)
        
        # Stats
        stats = [
            f"Episode: {episode}/3",
            f"Steps: {step_count}",
            f"Collected: {env.collected}/{env.num_plastics}",
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
            status = big_font.render(f"🤖 Collecting... {env.collected}/{env.num_plastics}", True, BLUE)
        status_rect = status.get_rect(center=(WINDOW_SIZE//2, info_y + 95))
        screen.blit(status, status_rect)
        
        pygame.display.flip()
        clock.tick(10)  # 10 FPS - easy to watch
    
    print(f"\nEpisode {episode} complete!")
    print(f"  Steps: {step_count}")
    print(f"  Plastic collected: {env.collected}/{env.num_plastics}")
    print(f"  Total reward: {total_reward:.2f}")
    
    if episode < 3:
        pygame.time.wait(2000)

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