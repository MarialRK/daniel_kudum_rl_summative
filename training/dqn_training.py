import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import PlasticPollutionEnv

def create_env():
    """Create the plastic pollution environment (enhanced 12x12)"""
    env = PlasticPollutionEnv(grid_size=12)
    return env

def train_dqn(learning_rate, gamma, buffer_size, batch_size, exploration_fraction, total_timesteps=80000):
    """
    Train DQN with given hyperparameters
    Returns: mean reward over last 50 episodes
    """
    env = create_env()
    env = Monitor(env)
    
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        train_freq=4,
        target_update_interval=1000,
        verbose=0
    )
    
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    
    # Evaluate
    eval_env = create_env()
    rewards = []
    for episode in range(50):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            if truncated:
                break
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    eval_env.close()
    env.close()
    
    return mean_reward

def run_dqn_experiments():
    """Run 10 DQN experiments with different hyperparameters"""
    print("=" * 70)
    print("DQN HYPERPARAMETER TUNING EXPERIMENTS (Enhanced Environment)")
    print("=" * 70)
    print()
    
    experiments = [
        {"learning_rate": 0.0001, "gamma": 0.99, "buffer_size": 100000, "batch_size": 32, "exploration_fraction": 0.1},
        {"learning_rate": 0.0003, "gamma": 0.99, "buffer_size": 100000, "batch_size": 32, "exploration_fraction": 0.1},
        {"learning_rate": 0.0005, "gamma": 0.99, "buffer_size": 100000, "batch_size": 32, "exploration_fraction": 0.1},
        {"learning_rate": 0.0001, "gamma": 0.95, "buffer_size": 100000, "batch_size": 32, "exploration_fraction": 0.1},
        {"learning_rate": 0.0001, "gamma": 0.99, "buffer_size": 50000, "batch_size": 32, "exploration_fraction": 0.1},
        {"learning_rate": 0.0001, "gamma": 0.99, "buffer_size": 100000, "batch_size": 64, "exploration_fraction": 0.1},
        {"learning_rate": 0.0001, "gamma": 0.99, "buffer_size": 100000, "batch_size": 32, "exploration_fraction": 0.2},
        {"learning_rate": 0.0002, "gamma": 0.98, "buffer_size": 75000, "batch_size": 48, "exploration_fraction": 0.15},
        {"learning_rate": 0.0005, "gamma": 0.97, "buffer_size": 150000, "batch_size": 128, "exploration_fraction": 0.05},
        {"learning_rate": 0.0008, "gamma": 0.99, "buffer_size": 200000, "batch_size": 64, "exploration_fraction": 0.12}
    ]
    
    results = []
    
    for i, params in enumerate(experiments, 1):
        print(f"Experiment {i}/10")
        print(f"  Learning Rate: {params['learning_rate']}")
        print(f"  Gamma: {params['gamma']}")
        print(f"  Buffer Size: {params['buffer_size']}")
        print(f"  Batch Size: {params['batch_size']}")
        print(f"  Exploration Fraction: {params['exploration_fraction']}")
        print("  Training...", end=" ", flush=True)
        
        mean_reward = train_dqn(**params)
        
        print(f"Mean Reward: {mean_reward:.2f}")
        print()
        
        results.append({"experiment": i, **params, "mean_reward": mean_reward})
    
    # Print summary
    print("\n" + "=" * 70)
    print("DQN EXPERIMENTS SUMMARY")
    print("=" * 70)
    print(f"{'Exp':<5} {'LR':<12} {'Gamma':<8} {'Buffer':<10} {'Batch':<8} {'Exploration':<12} {'Mean Reward':<12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['experiment']:<5} {r['learning_rate']:<12} {r['gamma']:<8} "
              f"{r['buffer_size']:<10} {r['batch_size']:<8} {r['exploration_fraction']:<12} "
              f"{r['mean_reward']:<12.2f}")
    
    # Find and save best model
    best = max(results, key=lambda x: x['mean_reward'])
    print(f"\nBest DQN Configuration: LR={best['learning_rate']}, Gamma={best['gamma']}, Mean Reward={best['mean_reward']:.2f}")
    
    print("\nTraining final best DQN model...")
    final_env = create_env()
    final_model = DQN(
        policy="MlpPolicy",
        env=final_env,
        learning_rate=best['learning_rate'],
        gamma=best['gamma'],
        buffer_size=best['buffer_size'],
        batch_size=best['batch_size'],
        exploration_fraction=best['exploration_fraction'],
        verbose=0
    )
    final_model.learn(total_timesteps=150000)
    
    os.makedirs("models/dqn", exist_ok=True)
    final_model.save("models/dqn/best_dqn_model")
    print("Best DQN model saved to models/dqn/best_dqn_model.zip")
    final_env.close()
    
    return results

if __name__ == "__main__":
    results = run_dqn_experiments()
    print("\nDQN Training Complete!")