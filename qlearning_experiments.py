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
                action = opponent_func(state, N, K)
                next_state, done, reward = make_move(state, action, N, K)
            state = next_state

        # Update win, draw, and loss counts
        if reward == REWARD_WIN:
            if last_player == 1:
                win_count += 1
            else:
                loss_count += 1
        elif reward == REWARD_DRAW:
            draw_count += 1

        # At the end of the interval, store the stats
        if (episode + 1) % print_interval == 0:
            total = win_count + draw_count + loss_count
            if total > 0:
                win_rates.append(win_count / total)
                draw_rates.append(draw_count / total)
                loss_rates.append(loss_count / total)

                # Debug: print the current win, draw, and loss stats for this interval
                print(f"Interval {episode + 1}:")
                print(f"  win_count = {win_count}, draw_count = {draw_count}, loss_count = {loss_count}")
                print(f"  win_rate = {win_count / total:.2f}, draw_rate = {draw_count / total:.2f}, loss_rate = {loss_count / total:.2f}")
            else:
                print(f"No data collected for this interval at episode {episode+1}.")
            win_count = draw_count = loss_count = 0

    print(f"Final win_rates: {win_rates}")
    print(f"Final draw_rates: {draw_rates}")
    print(f"Final loss_rates: {loss_rates}")

    return win_rates, draw_rates, loss_rates

# =========================
# 2) Γράφημα Μάθησης
# =========================
def plot_learning_curve(win_rates, draw_rates, loss_rates, interval, filename):
    """
    Plots the learning curve of the agent.

    win_rates, draw_rates, loss_rates: Lists containing the win, draw, and loss rates.
    interval: The number of episodes in each interval.
    filename: The filename to save the plot image.
    """
    x = [i * interval for i in range(len(win_rates))]  # x-axis: Episode intervals
    plt.plot(x, win_rates, label='Win Rate')
    plt.plot(x, draw_rates, label='Draw Rate')
    plt.plot(x, loss_rates, label='Loss Rate')
    
    plt.xlabel('Episodes')
    plt.ylabel('Rate')
    plt.title('Agent Learning Curve')
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved as {filename}")

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
                action = opponent_func(state, N, K)  # Pass both N and K
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

        # Open the file to log training rates
        with open(f"{results_dir}/training_rates.txt", "a", encoding="utf-8") as f:
            f.write(f"\nRunning config: α={ALPHA}, γ={GAMMA}, ε={EPSILON}, N={N}, K={K}, episodes={EPISODES}\n")

        # Train the agent and get win, draw, and loss rates
        win_rates, draw_rates, loss_rates = train_and_log(EPISODES, random_player_move, interval)

        # Log the win, draw, and loss rates into the new training_rates.txt file
        with open(f"{results_dir}/training_rates.txt", "a", encoding="utf-8") as f:
            f.write(f"Final win_rates: {win_rates}\n")
            f.write(f"Final draw_rates: {draw_rates}\n")
            f.write(f"Final loss_rates: {loss_rates}\n")

        # Plot the learning curve and save it as a PNG file
        filename_base = f"N{N}_K{K}_eps{EPSILON}_alpha{ALPHA}_gamma{GAMMA}_ep{EPISODES}"
        plot_learning_curve(win_rates, draw_rates, loss_rates, interval, f"{results_dir}/curve_{filename_base}.png")

        print("📦 Q-table size before saving:", len(Q))
        print("📦 Sample Q entries:", list(Q.items())[:3])
        print("🧩 Q-table keys:", list(Q.keys())[:5])

        # Save the Q-table to a pickle file
        with open(f"{results_dir}/qtable_{filename_base}.pkl", "wb") as f:
            pickle.dump(Q, f)

        print(f"Q-table size for config {filename_base}: {len(Q)} entries")

        # Evaluate the agent against random player after training
        win, draw, loss = evaluate_agent(random_player_move, games=1000)

        # Log evaluation results
        with open(f"{results_dir}/evaluation_results_summary.txt", "a", encoding="utf-8") as f:
            f.write(f"Config alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, N={N}, K={K}, episodes={EPISODES} -> Win: {win}, Draw: {draw}, Loss: {loss}\n")

# =========================
# 5) Πειράματα με Minimax
# =========================
# Run experiments with Minimax as the opponent
# Run experiments with Minimax as the opponent
def run_experiments_Minimax(configs, interval=1000):
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
        
        # Open the file to log training rates for Minimax experiments
        with open(f"{results_dir}/training_rates_minimax.txt", "a", encoding="utf-8") as f:
            f.write(f"\nRunning config: α={ALPHA}, γ={GAMMA}, ε={EPSILON}, N={N}, K={K}, episodes={EPISODES}\n")

        # Train the agent with Minimax as the opponent
        win_rates, draw_rates, loss_rates = train_and_log(EPISODES, minimax_move, interval)  # Use Minimax as opponent

        # Log the win, draw, and loss rates into the new training_rates_minimax.txt file
        with open(f"{results_dir}/training_rates_minimax.txt", "a", encoding="utf-8") as f:
            f.write(f"Final win_rates: {win_rates}\n")
            f.write(f"Final draw_rates: {draw_rates}\n")
            f.write(f"Final loss_rates: {loss_rates}\n")

        # Plot the learning curve and save it as a PNG file
        filename_base = f"Minimax: N{N}_K{K}_eps{EPSILON}_alpha{ALPHA}_gamma{GAMMA}_ep{EPISODES}"
        plot_learning_curve(win_rates, draw_rates, loss_rates, interval, f"{results_dir}/curve_{filename_base}.png")

        # Saving the Q-table after the Minimax training
        with open(f"{results_dir}/qtable_{filename_base}.pkl", "wb") as f:
            pickle.dump(Q, f)

        print(f"Q-table size for config {filename_base}: {len(Q)} entries")

        # Evaluate the trained agent against Minimax after training
        win, draw, loss = evaluate_agent(minimax_move, games=1000)  # Evaluate against Minimax
        
        # Log the evaluation results to the summary file
        with open(f"{results_dir}/evaluation_results_summary_minimax.txt", "a", encoding="utf-8") as f:
            f.write(f"Config alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, N={N}, K={K}, episodes={EPISODES} -> Win: {win}, Draw: {draw}, Loss: {loss}\n")



# =========================
# 6) Πειράματα με self_play
# =========================
# Run experiments with Self-play
# Run experiments with Self-play
def run_experiments_self_play(configs, interval=1000):
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
        
        # Open the file to log self-play training rates
        with open(f"{results_dir}/training_rates_self_play.txt", "a", encoding="utf-8") as f:
            f.write(f"\nRunning config: α={ALPHA}, γ={GAMMA}, ε={EPSILON}, N={N}, K={K}, episodes={EPISODES}\n")

        # Train via self-play and get win, draw, loss rates
        win_rates, draw_rates, loss_rates = train_and_log(EPISODES, random_player_move, interval)

        # Log the win, draw, and loss rates into the new training_rates_self_play.txt file
        with open(f"{results_dir}/training_rates_self_play.txt", "a", encoding="utf-8") as f:
            f.write(f"Final win_rates: {win_rates}\n")
            f.write(f"Final draw_rates: {draw_rates}\n")
            f.write(f"Final loss_rates: {loss_rates}\n")

        filename_base = f"self_play: N{N}_K{K}_eps{EPSILON}_alpha{ALPHA}_gamma{GAMMA}_ep{EPISODES}"
        
        # Save the Q-table after self-play training
        with open(f"{results_dir}/selfplay_qtable_{filename_base}.pkl", "wb") as f:
            pickle.dump(Q, f)

        print(f"Q-table size for config {filename_base}: {len(Q)} entries")

        # Optionally, you can evaluate after self-play training
        win, draw, loss = evaluate_agent(random_player_move, games=1000)  # Evaluate against random opponent
        with open(f"{results_dir}/evaluation_results_summary_selfplay.txt", "a", encoding="utf-8") as f:
            f.write(f"Config alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, N={N}, K={K}, episodes={EPISODES} -> Win: {win}, Draw: {draw}, Loss: {loss}\n")


# =========================
# 7) Κύρια Εκτέλεση
# =========================
if __name__ == "__main__":
    experiment_configs = [
        {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.2, "episodes": 1000, "N": 3, "K": 3},
        {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.2, "episodes": 1000, "N": 3, "K": 3},
        {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.2, "episodes": 1000, "N": 3, "K": 3},
    ]

    experiment_configs_test = [
        {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.2, "episodes": 1000, "N": 3, "K": 3},
    ]

    run_experiments(experiment_configs, interval=1000)
    #run_experiments_Minimax(experiment_configs_test, interval=1000)
    #run_experiments_self_play(experiment_configs, interval=1000)
