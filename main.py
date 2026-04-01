"""
Plastic Pollution Cleanup Agent - Main Entry Point
This script runs the best performing trained agent with visualization
"""

import sys
import os
import numpy as np
import pygame

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import PlasticPollutionEnv
from environment.rendering import PlasticPollutionRenderer

# Import stable baselines for loading models
from stable_baselines3 import DQN, PPO, A2C

def load_best_model(model_type="ppo"):
    """
    Load the best trained model
    Options: "dqn", "ppo", "a2c"
    """
    
    model_paths = {
        "dqn": "models/dqn/best_dqn_model.zip",
        "ppo": "models/pg/best_ppo_model.zip",
        "a2c": "models/pg/best_a2c_model.zip"
    }
    
    if model_type not in model_paths:
        print(f"Model type '{model_type}' not found. Using PPO.")
        model_type = "ppo"
    
    path = model_paths[model_type]
    
    if not os.path.exists(path):
        print(f"\n{'='*60}")
        print(f"WARNING: Model file not found: {path}")
        print(f"\nPlease train the models first by running:")
        print(f"  python training/dqn_training.py")
        print(f"  python training/pg_training.py")
        print(f"{'='*60}\n")
        
        # Offer to run random demo instead
        response = input("Would you like to run the random actions demo instead? (y/n): ")
        if response.lower() == 'y':
            return None
        else:
            sys.exit(1)
    
    print(f"Loading model: {path}")
    
    if model_type == "dqn":
        model = DQN.load(path)
    elif model_type == "ppo":
        model = PPO.load(path)
    elif model_type == "a2c":
        model = A2C.load(path)
    else:
        model = None
    
    return model

def run_trained_agent(model, env, renderer, max_steps=300):
    """
    Run the trained agent in the environment with visualization
    """
    obs, info = env.reset()
    total_reward = 0
    step = 0
    done = False
    
    print("\n" + "=" * 60)
    print("PLASTIC POLLUTION CLEANUP - TRAINED AGENT IN ACTION")
    print("=" * 60)
    print(f"Total Plastic: {info['total_plastic']}")
    print(f"Obstacles: {len(env.obstacle_positions)}")
    print("-" * 60)
    print("Press ESC to exit, watch agent clean up plastic!")
    print("-" * 60)
    
    running = True
    clock = pygame.time.Clock()
    
    while running and step < max_steps and not done:
        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Action names for display
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'COLLECT']
        
        # Take action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Render the environment
        running = renderer.render(obs, total_reward, done, info)
        
        # Print to terminal
        print(f"Step {step:3d}: Action = {action_names[action]:6s} | "
              f"Reward = {reward:+6.1f} | "
              f"Collected = {info['collected']:2d}/{info['total_plastic']:2d} | "
              f"Remaining = {info['remaining']:2d}")
        
        # Control speed
        clock.tick(10)  # 10 FPS - slower so you can see what's happening
        
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        if done:
            print("\n" + "=" * 60)
            print(f"SUCCESS! All plastic collected in {step} steps!")
            print(f"Total Reward: {total_reward:.2f}")
            print("=" * 60)
            break
    
    if step >= max_steps and not done:
        print("\n" + "=" * 60)
        print(f"Demo complete: Max steps ({max_steps}) reached")
        print(f"Plastic remaining: {info['remaining']}")
        print(f"Total Reward: {total_reward:.2f}")
        print("=" * 60)
    
    # Keep window open until user closes
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    waiting = False
    
    pygame.quit()

def run_random_demo():
    """
    Run random actions demo if models are not trained yet
    This satisfies the requirement: "Create a static file that shows the agent 
    taking random actions in the custom environment"
    """
    print("\n" + "=" * 60)
    print("RUNNING RANDOM ACTIONS DEMO")
    print("(No trained model - showing environment behavior)")
    print("=" * 60)
    
    env = PlasticPollutionEnv(grid_size=10)
    renderer = PlasticPollutionRenderer(env, cell_size=60)
    renderer.render_random_actions(num_steps=150, delay=0.05)
    env.close()

def main():
    """
    Main entry point
    """
    print("\n" + "=" * 60)
    print("PLASTIC POLLUTION CLEANUP AGENT")
    print("Reinforcement Learning for Environmental Sustainability")
    print("=" * 60)
    print("\nDaniel Kudum - RL Summative Assignment")
    print("\nThis program demonstrates a trained AI agent")
    print("that learns to collect plastic pollution in a simulated environment.\n")
    
    # Ask user which model to run
    print("Select model to run:")
    print("  1. PPO (Best overall performance)")
    print("  2. DQN (Value-based method)")
    print("  3. A2C (Actor-Critic method)")
    print("  4. Random actions demo (no training)")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    # Map choices to model types
    model_map = {
        "1": "ppo",
        "2": "dqn", 
        "3": "a2c",
        "4": "random"
    }
    
    model_type = model_map.get(choice, "ppo")
    
    if model_type == "random":
        run_random_demo()
        return
    
    # Load the model
    model = load_best_model(model_type)
    
    if model is None:
        # Model not found, offer random demo
        run_random_demo()
        return
    
    # Create environment and renderer
    env = PlasticPollutionEnv(grid_size=10)
    renderer = PlasticPollutionRenderer(env, cell_size=60)
    
    # Run the trained agent
    run_trained_agent(model, env, renderer)
    
    env.close()
    print("\nDemo completed. Thank you for using Plastic Pollution Cleanup Agent!")

if __name__ == "__main__":
    main()