README.md content for your Reinforcement Learning project:

# Reinforcement Learning – Generalized Tic-Tac-Toe

This project implements a Q-learning agent that learns to play generalized Tic-Tac-Toe (n x n grid, k in a row to win) through interaction with a simulated environment. It includes training loops, logging, evaluation, and performance visualization.

## 📁 Project Structure

- `QLearningTicTacToe.py`: Core environment and Q-learning logic
- `qlearning_experiments.py`: Script to run training experiments, generate plots, and evaluate performance
- `requirements.txt`: Python dependencies
- `results/`: Output folder with learning curves and evaluation logs

## 🚀 Getting Started

### 1. Clone the repository:
```bash
git clone https://github.com/SteliosKapsis/Reinforcement-Learning.git
cd Reinforcement-Learning
```

### 2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🧠 Run Experiments
To run all configured experiments (with different hyperparameters):
```bash
python qlearning_experiments.py
```
This will generate:
- `.png` learning curves in `/results`
- A `results_summary.txt` log with win/draw/loss counts after evaluation

## 🧪 Customize Hyperparameters
Inside `qlearning_experiments.py` you can modify the `experiment_configs` list to test different values of:
- `alpha` (learning rate)
- `gamma` (discount factor)
- `epsilon` (exploration rate)
- `episodes` (training duration)
- `N`, `K` (board size and win condition)

## 📊 Plotting & Evaluation
Each experiment logs win, draw, and loss rates, and generates a plot for training performance. You can open the `.png` files to compare learning speed and stability.

## 🔒 .gitignore (recommended)
Your `.gitignore` should exclude:
```gitignore
venv/
__pycache__/
*.png
results/
```

## 📌 Author
Developed by **Stelios Kapsis** as part of an Artificial Intelligence assignment on Reinforcement Learning.