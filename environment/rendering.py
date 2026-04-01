import pygame
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PlasticPollutionRenderer:
    """
    Enhanced Pygame renderer for Plastic Pollution Environment
    Shows current flow, wind effects, and plastic types
    """
    
    def __init__(self, env, cell_size=50):
        self.env = env
        self.cell_size = cell_size
        self.grid_size = env.grid_size
        
        # Calculate window size
        self.window_width = self.grid_size * self.cell_size
        self.window_height = self.grid_size * self.cell_size + 150  # Extra space for info panel
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Plastic Pollution Cleanup Agent - Enhanced Environment")
        self.clock = pygame.time.Clock()
        
        # Colors (RGB)
        self.COLORS = {
            'empty': (200, 220, 240),      # Light blue - water
            'agent': (0, 100, 255),        # Blue - cleanup robot
            'plastic_small': (139, 69, 19),    # Brown - small plastic
            'plastic_large': (101, 67, 33),    # Dark brown - large plastic
            'plastic_micro': (205, 133, 63),   # Light brown - microplastic
            'obstacle': (100, 100, 100),   # Dark gray - obstacle/rock
            'water': (173, 216, 230),      # Light blue - water background
            'text': (0, 0, 0),             # Black - text
            'reward_good': (0, 255, 0),    # Green - positive reward
            'reward_bad': (255, 0, 0),     # Red - negative reward
            'info_bg': (240, 240, 240),    # Light gray - info panel background
            'current': (100, 150, 200),    # Blue - current indicator
            'wind': (150, 150, 200)        # Purple - wind indicator
        }
        
        # Fonts
        self.font = pygame.font.Font(None, 20)
        self.big_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 16)
    
    def render(self, observation, reward=0, done=False, info=None):
        """
        Render the current state of the environment
        """
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Clear screen with water background
        self.screen.fill(self.COLORS['water'])
        
        # Draw grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j * self.cell_size
                y = i * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # Get cell value from full grid (since observation is partial)
                cell_value = self.env.full_grid[i, j]
                
                # Set color based on cell type
                if cell_value == 4:  # Agent
                    color = self.COLORS['agent']
                elif cell_value == 1:  # Small plastic
                    color = self.COLORS['plastic_small']
                elif cell_value == 2:  # Large plastic
                    color = self.COLORS['plastic_large']
                elif cell_value == 3:  # Microplastic
                    color = self.COLORS['plastic_micro']
                elif cell_value == 3 and cell_value != 1 and cell_value != 2:  # Obstacle
                    color = self.COLORS['obstacle']
                else:  # Empty
                    color = self.COLORS['empty']
                
                # Draw cell
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Border
                
                # Add icons for different plastic types
                if cell_value == 1:  # Small plastic
                    center = (x + self.cell_size//2, y + self.cell_size//2)
                    pygame.draw.circle(self.screen, (101, 67, 33), center, self.cell_size//5)
                elif cell_value == 2:  # Large plastic
                    center = (x + self.cell_size//2, y + self.cell_size//2)
                    pygame.draw.rect(self.screen, (80, 50, 25), 
                                   (center[0]-self.cell_size//5, center[1]-self.cell_size//5,
                                    self.cell_size//2.5, self.cell_size//2.5))
                elif cell_value == 3:  # Microplastic
                    center = (x + self.cell_size//2, y + self.cell_size//2)
                    pygame.draw.circle(self.screen, (160, 100, 50), center, self.cell_size//6)
                elif cell_value == 3 and cell_value not in [1, 2]:  # Obstacle
                    points = [
                        (x + self.cell_size//2, y + self.cell_size//4),
                        (x + 3*self.cell_size//4, y + self.cell_size//2),
                        (x + self.cell_size//2, y + 3*self.cell_size//4),
                        (x + self.cell_size//4, y + self.cell_size//2)
                    ]
                    pygame.draw.polygon(self.screen, (80, 80, 80), points)
        
        # Draw info panel
        info_y = self.grid_size * self.cell_size
        info_rect = pygame.Rect(0, info_y, self.window_width, 150)
        pygame.draw.rect(self.screen, self.COLORS['info_bg'], info_rect)
        
        # Display statistics
        collected = info.get('collected', 0) if info else 0
        total = info.get('total_plastic', 0) if info else 0
        steps = info.get('steps', 0) if info else 0
        remaining = info.get('remaining', 0) if info else 0
        current_dir = info.get('current_direction', 0) if info else 0
        wind = info.get('wind_active', False) if info else False
        
        current_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        current_text = current_names[current_dir] if current_dir in range(4) else 'NONE'
        
        stats = [
            f"Plastic: {collected}/{total} collected",
            f"Steps: {steps}",
            f"Remaining: {remaining}",
            f"Reward: {reward:.1f}",
            f"Current Flow: {current_text}",
            f"Wind: {'ACTIVE' if wind else 'CALM'}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, self.COLORS['text'])
            self.screen.blit(text, (10, info_y + 10 + i * 22))
        
        # Display legend
        legend_x = self.window_width - 150
        legend_items = [
            ("Agent", self.COLORS['agent']),
            ("Small Plastic", self.COLORS['plastic_small']),
            ("Large Plastic", self.COLORS['plastic_large']),
            ("Microplastic", self.COLORS['plastic_micro']),
            ("Obstacle", self.COLORS['obstacle']),
            ("Current", self.COLORS['current'])
        ]
        
        legend_title = self.small_font.render("Legend:", True, self.COLORS['text'])
        self.screen.blit(legend_title, (legend_x, info_y + 10))
        
        for i, (name, color) in enumerate(legend_items):
            pygame.draw.rect(self.screen, color, (legend_x, info_y + 35 + i * 18, 15, 15))
            text = self.small_font.render(name, True, self.COLORS['text'])
            self.screen.blit(text, (legend_x + 20, info_y + 35 + i * 18))
        
        # Draw current indicator arrow
        arrow_x = self.window_width - 60
        arrow_y = info_y + 120
        if current_dir == 0:  # Up
            points = [(arrow_x, arrow_y + 15), (arrow_x - 10, arrow_y + 5), (arrow_x + 10, arrow_y + 5)]
            pygame.draw.polygon(self.screen, self.COLORS['current'], points)
        elif current_dir == 1:  # Down
            points = [(arrow_x, arrow_y + 5), (arrow_x - 10, arrow_y + 15), (arrow_x + 10, arrow_y + 15)]
            pygame.draw.polygon(self.screen, self.COLORS['current'], points)
        elif current_dir == 2:  # Left
            points = [(arrow_x + 15, arrow_y + 10), (arrow_x + 5, arrow_y), (arrow_x + 5, arrow_y + 20)]
            pygame.draw.polygon(self.screen, self.COLORS['current'], points)
        elif current_dir == 3:  # Right
            points = [(arrow_x + 5, arrow_y + 10), (arrow_x + 15, arrow_y), (arrow_x + 15, arrow_y + 20)]
            pygame.draw.polygon(self.screen, self.COLORS['current'], points)
        
        # Display status if done
        if done:
            status_text = "CLEANUP COMPLETE! Press ESC to exit"
            text = self.big_font.render(status_text, True, self.COLORS['reward_good'])
            text_rect = text.get_rect(center=(self.window_width//2, info_y + 130))
            self.screen.blit(text, text_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)
        
        return True
    
    def render_random_actions(self, num_steps=200, delay=0.05):
        """
        Show agent taking random actions (no training)
        Demonstrates enhanced environment features
        """
        obs, info = self.env.reset()
        running = True
        step = 0
        
        print("=" * 60)
        print("PLASTIC POLLUTION ENVIRONMENT - ENHANCED VERSION")
        print("Features: Current Flow | Wind Effects | Multiple Plastic Types")
        print("=" * 60)
        print(f"Grid Size: {self.env.grid_size}x{self.env.grid_size}")
        
        total_plastic = info.get('total_plastic', 0) if info else 0
        print(f"Total Plastic: {total_plastic} (Small/Large/Micro mixed)")
        print(f"Obstacles: {len(self.env.obstacle_positions)}")
        current_dir = info.get('current_direction', 0) if info else 0
        current_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"Current Flow: {current_names[current_dir]}")
        print(f"Wind: {'ACTIVE' if info.get('wind_active') else 'CALM'}")
        print(f"Max Steps: {num_steps}")
        print("-" * 60)
        print("Press ESC to exit, or watch random agent explore")
        print("-" * 60)
        
        while running and step < num_steps:
            # Take random action
            action = np.random.randint(0, 5)
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'COLLECT']
            
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Render
            running = self.render(obs, reward, done, info)
            
            # Get current stats
            collected = info.get('collected', 0) if info else 0
            total = info.get('total_plastic', 0) if info else 0
            remaining = info.get('remaining', 0) if info else 0
            
            # Print to terminal
            print(f"Step {step+1:3d}: {action_names[action]:6s} | "
                  f"Reward: {reward:+6.1f} | "
                  f"Collected: {collected}/{total} | "
                  f"Remaining: {remaining}")
            
            step += 1
            pygame.time.wait(int(delay * 1000))
            
            if done:
                print("\n" + "=" * 60)
                print(f"SUCCESS! All plastic collected in {step} steps!")
                print("=" * 60)
                break
        
        if step >= num_steps and not done:
            print("\n" + "=" * 60)
            print(f"Demo complete: Max steps ({num_steps}) reached")
            print(f"Plastic remaining: {remaining}")
            print("=" * 60)
        
        # Keep window open
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False
        
        pygame.quit()
    
    def close(self):
        """Close the renderer"""
        pygame.quit()


def run_random_demo():
    """Run random actions demo with enhanced environment"""
    from environment.custom_env import PlasticPollutionEnv
    
    env = PlasticPollutionEnv(grid_size=12)
    renderer = PlasticPollutionRenderer(env, cell_size=50)
    renderer.render_random_actions(num_steps=200, delay=0.05)
    env.close()


if __name__ == "__main__":
    run_random_demo()