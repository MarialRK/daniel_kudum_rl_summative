import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import PlasticPollutionEnv

def create_env():
    """Create the enhanced plastic pollution environment (12x12)"""
    env = PlasticPollutionEnv(grid_size=12)
    return env

def train_reinforce(learning_rate, gamma, total_timesteps=80000):
    """Train REINFORCE (using PPO with 1 epoch)"""
    env = create_env()
    env = Monitor(env)
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_epochs=1,
        batch_size=32,
        n_steps=2048,
        verbose=0
    )
    
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    
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

def train_ppo(learning_rate, gamma, batch_size, n_steps, ent_coef, total_timesteps=80000):
    """Train PPO"""
    env = create_env()
    env = Monitor(env)
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        n_steps=n_steps,
        ent_coef=ent_coef,
        n_epochs=10,
        clip_range=0.2,
        verbose=0
    )
    
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    
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

def train_a2c(learning_rate, gamma, n_steps, ent_coef, total_timesteps=80000):
    """Train A2C"""
    env = create_env()
    env = Monitor(env)
    
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        ent_coef=ent_coef,
        verbose=0
    )
    
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    
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

def run_reinforce_experiments():
    print("\n" + "=" * 70)
    print("REINFORCE HYPERPARAMETER TUNING EXPERIMENTS")
    print("=" * 70)
    
    experiments = [
        {"learning_rate": 0.0001, "gamma": 0.99},
        {"learning_rate": 0.0003, "gamma": 0.99},
        {"learning_rate": 0.0005, "gamma": 0.99},
        {"learning_rate": 0.0001, "gamma": 0.95},
        {"learning_rate": 0.0001, "gamma": 0.97},
        {"learning_rate": 0.0002, "gamma": 0.98},
        {"learning_rate": 0.0004, "gamma": 0.99},
        {"learning_rate": 0.0006, "gamma": 0.99},
        {"learning_rate": 0.0001, "gamma": 0.98},
        {"learning_rate": 0.0003, "gamma": 0.97}
    ]
    
    results = []
    for i, params in enumerate(experiments, 1):
        print(f"\nREINFORCE Experiment {i}/10")
        print(f"  Learning Rate: {params['learning_rate']}, Gamma: {params['gamma']}")
        print("  Training...", end=" ", flush=True)
        mean_reward = train_reinforce(**params)
        print(f"Mean Reward: {mean_reward:.2f}")
        results.append({"experiment": i, **params, "mean_reward": mean_reward})
    
    return results

def run_ppo_experiments():
    print("\n" + "=" * 70)
    print("PPO HYPERPARAMETER TUNING EXPERIMENTS")
    print("=" * 70)
    
    experiments = [
        {"learning_rate": 0.0003, "gamma": 0.99, "batch_size": 64, "n_steps": 2048, "ent_coef": 0.01},
        {"learning_rate": 0.0005, "gamma": 0.99, "batch_size": 64, "n_steps": 2048, "ent_coef": 0.01},
        {"learning_rate": 0.0001, "gamma": 0.99, "batch_size": 64, "n_steps": 2048, "ent_coef": 0.01},
        {"learning_rate": 0.0003, "gamma": 0.95, "batch_size": 64, "n_steps": 2048, "ent_coef": 0.01},
        {"learning_rate": 0.0003, "gamma": 0.99, "batch_size": 128, "n_steps": 2048, "ent_coef": 0.01},
        {"learning_rate": 0.0003, "gamma": 0.99, "batch_size": 32, "n_steps": 1024, "ent_coef": 0.01},
        {"learning_rate": 0.0003, "gamma": 0.99, "batch_size": 64, "n_steps": 4096, "ent_coef": 0.01},
        {"learning_rate": 0.0003, "gamma": 0.99, "batch_size": 64, "n_steps": 2048, "ent_coef": 0.02},
        {"learning_rate": 0.0003, "gamma": 0.99, "batch_size": 64, "n_steps": 2048, "ent_coef": 0.005},
        {"learning_rate": 0.0006, "gamma": 0.98, "batch_size": 128, "n_steps": 4096, "ent_coef": 0.015}
    ]
    
    results = []
    for i, params in enumerate(experiments, 1):
        print(f"\nPPO Experiment {i}/10")
        print(f"  LR:{params['learning_rate']} Gamma:{params['gamma']} Batch:{params['batch_size']} Steps:{params['n_steps']} Entropy:{params['ent_coef']}")
        print("  Training...", end=" ", flush=True)
        mean_reward = train_ppo(**params)
        print(f"Mean Reward: {mean_reward:.2f}")
        results.append({"experiment": i, **params, "mean_reward": mean_reward})
    
    return results

def run_a2c_experiments():
    print("\n" + "=" * 70)
    print("A2C HYPERPARAMETER TUNING EXPERIMENTS")
    print("=" * 70)
    
    experiments = [
        {"learning_rate": 0.0007, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01},
        {"learning_rate": 0.001, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01},
        {"learning_rate": 0.0003, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01},
        {"learning_rate": 0.0007, "gamma": 0.95, "n_steps": 5, "ent_coef": 0.01},
        {"learning_rate": 0.0007, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.01},
        {"learning_rate": 0.0007, "gamma": 0.99, "n_steps": 20, "ent_coef": 0.01},
        {"learning_rate": 0.0007, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.02},
        {"learning_rate": 0.0007, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.005},
        {"learning_rate": 0.0005, "gamma": 0.98, "n_steps": 10, "ent_coef": 0.01},
        {"learning_rate": 0.0009, "gamma": 0.98, "n_steps": 20, "ent_coef": 0.015}
    ]
    
    results = []
    for i, params in enumerate(experiments, 1):
        print(f"\nA2C Experiment {i}/10")
        print(f"  LR:{params['learning_rate']} Gamma:{params['gamma']} Steps:{params['n_steps']} Entropy:{params['ent_coef']}")
        print("  Training...", end=" ", flush=True)
        mean_reward = train_a2c(**params)
        print(f"Mean Reward: {mean_reward:.2f}")
        results.append({"experiment": i, **params, "mean_reward": mean_reward})
    
    return results

def print_results(reinforce, ppo, a2c):
    print("\n" + "=" * 70)
    print("REINFORCE RESULTS")
    print("=" * 70)
    print(f"{'Exp':<5} {'LR':<12} {'Gamma':<10} {'Mean Reward':<12}")
    for r in reinforce:
        print(f"{r['experiment']:<5} {r['learning_rate']:<12} {r['gamma']:<10} {r['mean_reward']:<12.2f}")
    
    print("\n" + "=" * 70)
    print("PPO RESULTS")
    print("=" * 70)
    print(f"{'Exp':<5} {'LR':<12} {'Gamma':<8} {'Batch':<8} {'Steps':<8} {'Mean Reward':<12}")
    for r in ppo:
        print(f"{r['experiment']:<5} {r['learning_rate']:<12} {r['gamma']:<8} {r['batch_size']:<8} {r['n_steps']:<8} {r['mean_reward']:<12.2f}")
    
    print("\n" + "=" * 70)
    print("A2C RESULTS")
    print("=" * 70)
    print(f"{'Exp':<5} {'LR':<12} {'Gamma':<8} {'Steps':<8} {'Mean Reward':<12}")
    for r in a2c:
        print(f"{r['experiment']:<5} {r['learning_rate']:<12} {r['gamma']:<8} {r['n_steps']:<8} {r['mean_reward']:<12.2f}")

def train_best_models():
    print("\n" + "=" * 70)
    print("TRAINING BEST MODELS")
    print("=" * 70)
    
    # Best PPO configuration
    print("\nTraining best PPO model...")
    env = create_env()
    best_ppo = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        gamma=0.99,
        batch_size=64,
        n_steps=2048,
        ent_coef=0.01,
        verbose=0
    )
    best_ppo.learn(total_timesteps=150000)
    os.makedirs("models/pg", exist_ok=True)
    best_ppo.save("models/pg/best_ppo_model")
    print("Best PPO model saved to models/pg/best_ppo_model.zip")
    env.close()
    
    # Best A2C configuration
    print("\nTraining best A2C model...")
    env = create_env()
    best_a2c = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0007,
        gamma=0.99,
        n_steps=5,
        ent_coef=0.01,
        verbose=0
    )
    best_a2c.learn(total_timesteps=150000)
    best_a2c.save("models/pg/best_a2c_model")
    print("Best A2C model saved to models/pg/best_a2c_model.zip")
    env.close()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("POLICY GRADIENT TRAINING - ENHANCED ENVIRONMENT")
    print("REINFORCE | PPO | A2C")
    print("=" * 70)
    
    reinforce_results = run_reinforce_experiments()
    ppo_results = run_ppo_experiments()
    a2c_results = run_a2c_experiments()
    
    print_results(reinforce_results, ppo_results, a2c_results)
    train_best_models()
    
    print("\nPolicy Gradient Training Complete!")