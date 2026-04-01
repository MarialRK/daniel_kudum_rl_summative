
# Daniel Kudum - Reinforcement Learning Summative

## Plastic Pollution Cleanup Agent

This project implements a reinforcement learning agent that learns to clean plastic pollution in a simulated environment. The agent navigates an 8x8 grid to collect different types of plastic waste: small (+20), large (+50), and micro (+10).

## Mission
To develop an AI-powered solution for autonomous plastic waste collection in water bodies, contributing to the fight against plastic pollution.

## Algorithms Implemented
- **DQN** (Deep Q-Network) - Value-based method
- **REINFORCE** - Policy gradient method
- **PPO** (Proximal Policy Optimization) - Advanced policy gradient
- **A2C** (Advantage Actor-Critic) - Actor-critic method

## Environment
A custom Gymnasium environment representing a river cleanup zone:
- **Grid Size**: 8x8
- **Agent**: Cleanup robot (blue square)
- **Plastic Types**: Small (+20), Large (+50), Micro (+10)
- **Plastic Items**: 5 randomly placed per episode
- **Obstacles**: 3-5 obstacles to avoid
- **Rewards**: +20/50/10 for collecting plastic, -0.5 per step, +100 completion bonus

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

text

### 2. Train DQN
python training/dqn_training.py

text

### 3. Train Policy Gradient methods (REINFORCE, PPO, A2C)
python training/pg_training.py

text

### 4. Run the best performing model
python main.py

text

## Project Structure
daniel_kudum_rl_summative/
├── environment/
│ ├── custom_env.py # Custom Gymnasium environment
│ └── rendering.py # Pygame visualization
├── training/
│ ├── dqn_training.py # DQN training script
│ └── pg_training.py # REINFORCE, PPO, A2C training
├── models/
│ ├── dqn/ # Saved DQN models
│ └── pg/ # Saved policy gradient models
├── main.py # Run best performing model
├── requirements.txt # Dependencies
└── README.md # Documentation

text

## Video Demonstration
[https://screenrec.com/share/6hVPqgsA2y](https://screenrec.com/share/6hVPqgsA2y)

## GitHub Repository
[https://github.com/MarialRK/daniel_kudum_rl_summative](https://github.com/MarialRK/daniel_kudum_rl_summative)

## Results Summary
- **Best Algorithm**: DQN achieved the highest mean reward of -0.23
- **Convergence Speed**: PPO converged fastest at 120 episodes
- **Best Generalization**: PPO maintained 85% performance on new configurations

## Author
Daniel Kudum
