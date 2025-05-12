# qlearning_experiments.py
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from QLearningTicTacToe import *
from shared_q import Q  # ✅ shared Q-table

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
                print(f"Agent played → reward: {reward}, done: {done}")
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
    global ALPHA, GAMMA, EPSILON, N, K
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for config in configs:
        ALPHA = config['alpha']
        GAMMA = config['gamma']
        EPSILON = config['epsilon']
        N = config.get('N', 3)
        K = config.get('K', 3)
        EPISODES = config.get('episodes', 30000)
        Q.clear()  # ✅ properly clear shared Q-table

        print(f"\nRunning config: α={ALPHA}, γ={GAMMA}, ε={EPSILON}, N={N}, K={K}, episodes={EPISODES}")
        win_rates, draw_rates, loss_rates = train_and_log(EPISODES, random_player_move, interval)

        filename_base = f"N{N}_K{K}_eps{EPSILON}_alpha{ALPHA}_gamma{GAMMA}_ep{EPISODES}"
        plot_learning_curve(win_rates, draw_rates, loss_rates, interval, f"{results_dir}/curve_{filename_base}.png")

        print("📦 Q-table size before saving:", len(Q))
        print("📦 Sample Q entries:", list(Q.items())[:3])
        print("🧩 Q-table keys:", list(Q.keys())[:5])

        with open(f"{results_dir}/qtable_{filename_base}.pkl", "wb") as f:
            pickle.dump(Q, f)

        print(f"Q-table size for config {filename_base}: {len(Q)} entries")

        win, draw, loss = evaluate_agent(random_player_move, games=1000)
        with open(f"{results_dir}/results_summary.txt", "a", encoding="utf-8") as f:
            f.write(f"Config alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, N={N}, K={K}, episodes={EPISODES} -> Win: {win}, Draw: {draw}, Loss: {loss}\n")

# =========================
# 5) Κύρια Εκτέλεση
# =========================
if __name__ == "__main__":
    experiment_configs = [
        {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.2, "episodes": 1000, "N": 3, "K": 3},
        {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.2, "episodes": 1000, "N": 3, "K": 3},
        {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.2, "episodes": 1000, "N": 3, "K": 3},
    ]
    run_experiments(experiment_configs, interval=1000)
