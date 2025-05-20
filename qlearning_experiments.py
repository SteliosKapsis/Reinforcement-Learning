# qlearning_experiments.py
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from QLearningTicTacToe import *
from shared_q import Q  # ✅ shared Q-table
import json
import time

# =======================================
def log_experiment_to_json(filename_base, config, win, draw, loss, qtable_size, elapsed_seconds, results_dir="results"):
    """
    Αποθηκεύει τα αποτελέσματα ενός πειράματος σε JSON αρχείο με βάση το filename_base.
    """
    os.makedirs(results_dir, exist_ok=True)

    log_data = {
        "alpha": config["alpha"],
        "gamma": config["gamma"],
        "epsilon": config["epsilon"],
        "N": config.get("N", 3),
        "K": config.get("K", 3),
        "episodes": config.get("episodes", 30000),
        "evaluation_win": win,
        "evaluation_draw": draw,
        "evaluation_loss": loss,
        "qtable_size": qtable_size,
        "duration_seconds": round(elapsed_seconds, 2)
    }

    json_path = os.path.join(results_dir, f"log_{filename_base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"📄 JSON log saved to: {json_path}")


# =======================================


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

    print("Plot x:", x)
    print("Plot win:", win_rates)

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

    for i,config in enumerate(configs):
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
        
        start_time = time.time()
        # Train the agent and get win, draw, and loss rates
        win_rates, draw_rates, loss_rates = train_and_log(EPISODES, random_player_move, interval)

        # Log the win, draw, and loss rates into the new training_rates.txt file
        with open(f"{results_dir}/training_rates.txt", "a", encoding="utf-8") as f:
            f.write(f"Final win_rates: {win_rates}\n")
            f.write(f"Final draw_rates: {draw_rates}\n")
            f.write(f"Final loss_rates: {loss_rates}\n")

        # Plot the learning curve and save it as a PNG file
        filename_base = f"N{N}_K{K}_eps{EPSILON}_alpha{ALPHA}_gamma{GAMMA}_ep{EPISODES}".replace(".", "_")
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

        elapsed = time.time() - start_time

        log_experiment_to_json(
            filename_base=filename_base,
            config=config,
            win=win,
            draw=draw,
            loss=loss,
            qtable_size=len(Q),
            elapsed_seconds=elapsed,
            results_dir="results"
        )

# =========================
# 5) Πειράματα με Minimax
# =========================
# Run experiments with Minimax as the opponent
def run_experiments_Minimax(configs, interval=1000):
    global ALPHA, GAMMA, EPSILON, N, K
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for i,config in enumerate(configs):
        ALPHA = config['alpha']
        GAMMA = config['gamma']
        EPSILON = config['epsilon']
        N = config.get('N', 3)
        K = config.get('K', 3)
        EPISODES = config.get('episodes', 30000)
        Q.clear()  # ✅ properly clear shared Q-table

        print(f"\nMini_max_Running config: α={ALPHA}, γ={GAMMA}, ε={EPSILON}, N={N}, K={K}, episodes={EPISODES}")
        
        # Open the file to log training rates for Minimax experiments
        with open(f"{results_dir}/training_rates_minimax.txt", "a", encoding="utf-8") as f:
            f.write(f"\nRunning config: α={ALPHA}, γ={GAMMA}, ε={EPSILON}, N={N}, K={K}, episodes={EPISODES}\n")

        start_time = time.time()
        # Train the agent with Minimax as the opponent
        #win_rates, draw_rates, loss_rates = train_and_log(EPISODES, minimax_move, interval)  # Use Minimax as opponent
        win_rates, draw_rates, loss_rates = train_and_log(EPISODES, lambda s, n, k: minimax_move(s, n, k, max_depth=4), interval)

        # Log the win, draw, and loss rates into the new training_rates_minimax.txt file
        with open(f"{results_dir}/training_rates_minimax.txt", "a", encoding="utf-8") as f:
            f.write(f"Final win_rates: {win_rates}\n")
            f.write(f"Final draw_rates: {draw_rates}\n")
            f.write(f"Final loss_rates: {loss_rates}\n")

        # Plot the learning curve and save it as a PNG file
        filename_base = f"Minimax_N{N}_K{K}_eps{EPSILON}_alpha{ALPHA}_gamma{GAMMA}_ep{EPISODES}".replace(".", "_")
        plot_learning_curve(win_rates, draw_rates, loss_rates, interval, f"{results_dir}/curve_{filename_base}.png")

        print("📦 Q-table size before saving:", len(Q))
        print("📦 Sample Q entries:", list(Q.items())[:3])
        print("🧩 Q-table keys:", list(Q.keys())[:5])

        # Save the Q-table to a pickle file
        with open(f"{results_dir}/qtable_{filename_base}.pkl", "wb") as f:
            pickle.dump(Q, f)

        print(f"Q-table size for config {filename_base}: {len(Q)} entries")

        # Evaluate the trained agent against Minimax after training
        win, draw, loss = evaluate_agent(minimax_move, games=1000)  # Evaluate against Minimax
        
        # Log the evaluation results to the summary file
        with open(f"{results_dir}/evaluation_results_summary_minimax.txt", "a", encoding="utf-8") as f:
            f.write(f"Config alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, N={N}, K={K}, episodes={EPISODES} -> Win: {win}, Draw: {draw}, Loss: {loss}\n")

        elapsed = time.time() - start_time

        log_experiment_to_json(
            filename_base=filename_base,
            config=config,
            win=win,
            draw=draw,
            loss=loss,
            qtable_size=len(Q),
            elapsed_seconds=elapsed,
            results_dir="results"
        )


# =========================
# 6) Πειράματα με self_play
# =========================
# Run experiments with Self-play
def run_experiments_self_play(configs, interval=1000):
    global ALPHA, GAMMA, EPSILON, N, K
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for i,config in enumerate(configs):
        ALPHA = config['alpha']
        GAMMA = config['gamma']
        EPSILON = config['epsilon']
        N = config.get('N', 3)
        K = config.get('K', 3)
        EPISODES = config.get('episodes', 30000)
        Q.clear()  # ✅ properly clear shared Q-table

        print(f"\nSelf_play_Running config: α={ALPHA}, γ={GAMMA}, ε={EPSILON}, N={N}, K={K}, episodes={EPISODES}")
        
        # Open the file to log self-play training rates
        with open(f"{results_dir}/training_rates_self_play.txt", "a", encoding="utf-8") as f:
            f.write(f"\nRunning config: α={ALPHA}, γ={GAMMA}, ε={EPSILON}, N={N}, K={K}, episodes={EPISODES}\n")

        start_time = time.time()
        # Train via self-play and get win, draw, loss rates
        win_rates, draw_rates, loss_rates = train_and_log(EPISODES, random_player_move, interval)

        # Log the win, draw, and loss rates into the new training_rates_self_play.txt file
        with open(f"{results_dir}/training_rates_self_play.txt", "a", encoding="utf-8") as f:
            f.write(f"Final win_rates: {win_rates}\n")
            f.write(f"Final draw_rates: {draw_rates}\n")
            f.write(f"Final loss_rates: {loss_rates}\n")
        
        # Plot the learning curve and save it as a PNG file
        filename_base = f"Self_Play_N{N}_K{K}_eps{EPSILON}_alpha{ALPHA}_gamma{GAMMA}_ep{EPISODES}".replace(".", "_")
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
        with open(f"{results_dir}/evaluation_results_summary_self_play.txt", "a", encoding="utf-8") as f:
            f.write(f"Config alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, N={N}, K={K}, episodes={EPISODES} -> Win: {win}, Draw: {draw}, Loss: {loss}\n")

        elapsed = time.time() - start_time

        log_experiment_to_json(
            filename_base=filename_base,
            config=config,
            win=win,
            draw=draw,
            loss=loss,
            qtable_size=len(Q),
            elapsed_seconds=elapsed,
            results_dir="results"
        )

################################
def analyze_experiment_results(results_dir="results", save_plot=True):
    """
    Φορτώνει όλα τα .json logs από τον φάκελο και εμφανίζει/αποθηκεύει γράφημα με τα win rates.
    
    Args:
        results_dir (str): Ο φάκελος όπου βρίσκονται τα log_*.json.
        save_plot (bool): Αν True, αποθηκεύει το γράφημα σε PNG.
    """
    logs = []

    for filename in os.listdir(results_dir):
        if filename.startswith("log_") and filename.endswith(".json"):
            path = os.path.join(results_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logs.append(data)

    if not logs:
        print("❌ Δεν βρέθηκαν αρχεία log_*.json στον φάκελο:", results_dir)
        return

    # Δημιουργία DataFrame
    df = pd.DataFrame(logs)
    df["total_games"] = df["evaluation_win"] + df["evaluation_draw"] + df["evaluation_loss"]
    df["win_rate"] = df["evaluation_win"] / df["total_games"]
    df["draw_rate"] = df["evaluation_draw"] / df["total_games"]
    df["loss_rate"] = df["evaluation_loss"] / df["total_games"]
    df["label"] = df.apply(lambda row: f"N{row['N']}_K{row['K']}_α{row['alpha']}_γ{row['gamma']}_ε{row['epsilon']}", axis=1)

    # Ταξινόμηση κατά win rate
    df_sorted = df.sort_values(by="win_rate", ascending=False)

    # Οπτικοποίηση
    plt.figure(figsize=(12, 6))
    plt.bar(df_sorted["label"], df_sorted["win_rate"], color='green')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Win Rate")
    plt.title("🔎 Σύγκριση Configurations με βάση το Win Rate")
    plt.grid(axis='y')
    plt.tight_layout()

    # ✅ Αποθήκευση PNG
    if save_plot:
        plot_path = os.path.join(results_dir, "summary_win_rate_plot.png")
        plt.savefig(plot_path)
        print(f"📊 Το γράφημα αποθηκεύτηκε στο: {plot_path}")

    # Εμφάνιση
    plt.show()

    # Εκτύπωση κορυφαίων configs
    print("\n🏆 Καλύτερα Configs με βάση το Win Rate:")
    print(df_sorted[["label", "win_rate", "draw_rate", "loss_rate", "qtable_size", "duration_seconds"]].head(10))
################################

# =========================
# 7) Κύρια Εκτέλεση
# =========================
if __name__ == "__main__":

    experiment_configs = [
    # === Base Case: 3x3, K=3 ===
    {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.1, "episodes": 5000, "N": 3, "K": 3},  # Low epsilon, fast convergence
    {"alpha": 0.7, "gamma": 0.9, "epsilon": 0.3, "episodes": 5000, "N": 3, "K": 3},  # More exploration
    {"alpha": 0.5, "gamma": 0.6, "epsilon": 0.2, "episodes": 5000, "N": 3, "K": 3},  # Slower learning

    # === Increased challenge: K=4 (harder to win) ===
    {"alpha": 0.8, "gamma": 0.8, "epsilon": 0.2, "episodes": 7000, "N": 3, "K": 4},  # Technically impossible, test draw handling

    # === Larger board: 4x4 ===
    {"alpha": 0.9, "gamma": 0.95, "epsilon": 0.2, "episodes": 8000, "N": 4, "K": 3},  # More space, more Q-values
    {"alpha": 0.7, "gamma": 0.85, "epsilon": 0.3, "episodes": 8000, "N": 4, "K": 4},  # Diagonal strategies

    # === Even Larger: 5x5 ===
    {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.15, "episodes": 10000, "N": 5, "K": 4},  # Harder planning
    {"alpha": 0.6, "gamma": 0.95, "epsilon": 0.25, "episodes": 10000, "N": 5, "K": 5},  # Full board win

    # === Fast-learning but short memory ===
    {"alpha": 1.0, "gamma": 0.5, "epsilon": 0.2, "episodes": 6000, "N": 3, "K": 3},

    # === Long-term thinker but slow learner ===
    {"alpha": 0.4, "gamma": 0.99, "epsilon": 0.1, "episodes": 6000, "N": 3, "K": 3},
    ]

    experiment_configs_test = [
        {"alpha": 0.9, "gamma": 0.9, "epsilon": 0.2, "episodes": 5000, "N": 3, "K": 3},
    ]

    #run_experiments(experiment_configs, interval=500)
    #run_experiments_Minimax(experiment_configs, interval=500) #(If you try to train Minimax In this: # === Even Larger: 5x5 === , you will have to be very patient)
    #run_experiments_self_play(experiment_configs, interval=500)

    analyze_experiment_results()
