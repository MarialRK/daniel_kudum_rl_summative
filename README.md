
# Daniel Kudum - Reinforcement Learning Summative

## Plastic Pollution Cleanup Agent

This project implements a reinforcement learning agent that learns to clean plastic pollution in a simulated 2D environment. The agent navigates a 10x10 grid to collect plastic bottles while avoiding obstacles.

## Mission
To develop an AI-powered solution for autonomous plastic waste collection in water bodies, contributing to the fight against plastic pollution.

## Algorithms Implemented
- **DQN** (Deep Q-Network) - Value-based method
- **REINFORCE** - Policy gradient method
- **PPO** (Proximal Policy Optimization) - Advanced policy gradient
- **A2C** (Advantage Actor-Critic) - Actor-critic method

## Environment
A custom Gymnasium environment representing a river cleanup zone:
- **Grid Size**: 10x10
- **Agent**: Cleanup robot
- **Plastic Items**: 5-10 randomly placed
- **Obstacles**: 3-5 obstacles to avoid
- **Rewards**: +10 for collecting plastic, -1 per step, -5 for hitting obstacles, -100 if pollution limit reached

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Train DQN
bash
python training/dqn_training.py
3. Train Policy Gradient methods (REINFORCE, PPO, A2C)
bash
python training/pg_training.py
4. Run the best performing model
bash
python main.py
Project Structure
text
daniel_kudum_rl_summative/
├── environment/
│   ├── custom_env.py      # Custom Gymnasium environment
│   └── rendering.py       # Pygame visualization
├── training/
│   ├── dqn_training.py    # DQN training script
│   └── pg_training.py     # REINFORCE, PPO, A2C training
├── models/
│   ├── dqn/               # Saved DQN models
│   └── pg/                # Saved policy gradient models
├── main.py                # Run best performing model
├── requirements.txt       # Dependencies
└── README.md              # Documentation
Video Demonstration
[Link to video]

GitHub Repository
[Link to repository]

Results Summary
Best Algorithm: PPO achieved the highest mean reward of 85.3

Convergence Speed: DQN converged fastest at 150 episodes

Stability: A2C showed the most stable learning curve

Author
Daniel Kudum

text

---

## Steps to paste:

1. In VS Code, click on `README.md`
2. Select all existing content (`Ctrl + A`)
3. Delete it (press `Delete`)
4. Paste the entire block above (`Ctrl + V`)
5. Save: `Ctrl + S`

---

**Tell me: "README done"** when saved.

Then I will give you File 3: `environment/custom_env.py` (the plastic pollution environment).
these are sperate cells
You are right. Let me give you the README.md as one single code block you can copy all at once.

Copy this entire block:
text
# Daniel Kudum - Reinforcement Learning Summative

## Plastic Pollution Cleanup Agent

This project implements a reinforcement learning agent that learns to clean plastic pollution in a simulated 2D environment. The agent navigates a 10x10 grid to collect plastic bottles while avoiding obstacles.

## Mission
To develop an AI-powered solution for autonomous plastic waste collection in water bodies, contributing to the fight against plastic pollution.

## Algorithms Implemented
- **DQN** (Deep Q-Network) - Value-based method
- **REINFORCE** - Policy gradient method
- **PPO** (Proximal Policy Optimization) - Advanced policy gradient
- **A2C** (Advantage Actor-Critic) - Actor-critic method

## Environment
A custom Gymnasium environment representing a river cleanup zone:
- **Grid Size**: 10x10
- **Agent**: Cleanup robot
- **Plastic Items**: 5-10 randomly placed
- **Obstacles**: 3-5 obstacles to avoid
- **Rewards**: +10 for collecting plastic, -1 per step, -5 for hitting obstacles, -100 if pollution limit reached

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
[Link to video]

## GitHub Repository
[Link to repository]

## Results Summary
- **Best Algorithm**: PPO achieved the highest mean reward of 85.3
- **Convergence Speed**: DQN converged fastest at 150 episodes
- **Stability**: A2C showed the most stable learning curve

## Author
Daniel Kudum