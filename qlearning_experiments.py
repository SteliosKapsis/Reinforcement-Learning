# qlearning_experiments.py
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from QLearningTicTacToe import *

# =========================
# 1) Εκπαίδευση με καταγραφή στατιστικών
# =========================
def train_and_log(episodes, opponent_func, print_interval=1000):
    win_rates, draw_rates, loss_rates = [], [], []
    win_count = draw_count = loss_count = 0

    for episode in range(episodes):
        board = create_empty_board(N)
        state = (board, 1)
        done = False
        last_player = None

        while not done:
            board, current_player = state
            last_player = current_player
            if current_player == 1:
                action = choose_action_epsilon_greedy(state, N)
                next_state, done, reward = make_move(state, action, N, K)
                update_Q(state, action, reward, next_state, done)
            else:
                action = opponent_func(state, N)
                next_state, done, reward = make_move(state, action, N, K)
            state = next_state

        if reward == REWARD_WIN:
            if last_player == 1:
                win_count += 1
            else:
                loss_count += 1
        elif reward == REWARD_DRAW:
            draw_count += 1

        if (episode + 1) % print_interval == 0:
            total = win_count + draw_count + loss_count
            win_rates.append(win_count / total)
            draw_rates.append(draw_count / total)
            loss_rates.append(loss_count / total)
            win_count = draw_count = loss_count = 0

    return win_rates, draw_rates, loss_rates

# =========================
# 2) Γράφημα Μάθησης
# =========================
def plot_learning_curve(win_rates, draw_rates, loss_rates, interval, filename):
    x = [i * interval for i in range(len(win_rates))]
    plt.plot(x, win_rates, label='Win Rate')
    plt.plot(x, draw_rates, label='Draw Rate')
    plt.plot(x, loss_rates, label='Loss Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Rate')
    plt.title('Q-learning Agent Performance')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

# =========================
# 3) Αξιολόγηση χωρίς εξερεύνηση (greedy)
# =========================
def evaluate_agent(opponent_func, games=1000):
    win = draw = loss = 0
    for _ in range(games):
        board = create_empty_board(N)
        state = (board, 1)
        done = False
        last_player = None

        while not done:
            board, current_player = state
            last_player = current_player
            if current_player == 1:
                legal_actions = get_legal_actions(state, N)
                q_values = [get_Q_value(state, a) for a in legal_actions]
                best_q = max(q_values)
                best_actions = [a for a, q in zip(legal_actions, q_values) if q == best_q]
                action = random.choice(best_actions)
            else:
                action = opponent_func(state, N)
            next_state, done, reward = make_move(state, action, N, K)
            state = next_state

        if reward == REWARD_WIN:
            if last_player == 1:
                win += 1
            else:
                loss += 1
        elif reward == REWARD_DRAW:
            draw += 1

    return win, draw, loss

# =========================
# 4) Πειράματα με πολλαπλά παραμετρικά runs
# =========================
def run_experiments(configs, interval=1000):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for config in configs:
        global ALPHA, GAMMA, EPSILON, Q
        ALPHA, GAMMA, EPSILON = config['alpha'], config['gamma'], config['epsilon']
        EPISODES = config.get('episodes', 30000)
        Q = {}  # reset Q-table

        print(f"\nRunning Config: alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, episodes={EPISODES}")
        win_rates, draw_rates, loss_rates = train_and_log(EPISODES, random_player_move, interval)

        filename = f"{results_dir}/curve_eps{EPSILON}_alpha{ALPHA}_gamma{GAMMA}_ep{EPISODES}.png"
        plot_learning_curve(win_rates, draw_rates, loss_rates, interval, filename)

        win, draw, loss = evaluate_agent(random_player_move, games=1000)
        with open(f"results/results_summary.txt", "a", encoding="utf-8") as f:
            f.write(f"Board size:{N} (e.g., 10 -> 10x10 board),  # Number of consecutive marks needed to win:{K} (e.g., 5 in a row)\n")
            f.write(f"\n")
            f.write(f"Config: alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, episodes={EPISODES} → Win: {win}, Draw: {draw}, Loss: {loss}\n")
            f.write(f"\n")

# =========================
# 5) Κύρια Εκτέλεση
# =========================
if __name__ == "__main__":
    experiment_configs = [
        {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.2, "episodes": 30000},
        {"alpha": 0.05, "gamma": 0.95, "epsilon": 0.3, "episodes": 40000},
        {"alpha": 0.2, "gamma": 0.9, "epsilon": 0.1, "episodes": 50000},
    ]
    run_experiments(experiment_configs, interval=1000)
