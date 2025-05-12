import random
import numpy as np

# ======================
# 1) Hyperparameters
# ======================
N = 10           # Board size (e.g., 10 -> 10x10 board)
K = 5            # Number of consecutive marks needed to win (e.g., 5 in a row)
ALPHA = 0        # Learning rate
GAMMA = 9        # Discount factor
EPSILON = 2      # Epsilon for epsilon-greedy exploration
EPISODES = 30000 # Number of training episodes

# Rewards
REWARD_WIN = 1.0
REWARD_DRAW = 0.0
REWARD_STEP = 0.0

# ======================
# 2) Game Logic
# ======================
def create_empty_board(n):
    """
    Returns a tuple of length n*n filled with 0, denoting an empty board.
    0 = empty, 1 = X, 2 = O.
    """
    return tuple([0] * (n*n))

def get_legal_actions(state, n):
    """
    Return a list of valid actions (board indices) for the current state,
    i.e., the positions that are still empty (0).
    """
    board, current_player = state
    return [i for i, val in enumerate(board) if val == 0]

def is_winner_k(board, player, n, k):
    """
    Checks if 'player' (1 or 2) has at least K consecutive marks in any row,
    column, or diagonal on an n x n board.
    """
    matrix = np.array(board).reshape(n, n)

    # Check rows
    for row in range(n):
        for col in range(n - k + 1):
            segment = matrix[row, col:col + k]
            if all(cell == player for cell in segment):
                return True

    # Check columns
    for col in range(n):
        for row in range(n - k + 1):
            segment = matrix[row:row + k, col]
            if all(cell == player for cell in segment):
                return True

    # Check main diagonals
    for row in range(n - k + 1):
        for col in range(n - k + 1):
            if all(matrix[row + d, col + d] == player for d in range(k)):
                return True

    # Check anti-diagonals
    for row in range(n - k + 1):
        for col in range(k - 1, n):
            if all(matrix[row + d, col - d] == player for d in range(k)):
                return True

    return False

def make_move(state, action, n, k):
    """
    Given a state (board, current_player) and an action (index),
    return (next_state, done, reward).
    """
    board, current_player = state
    board_list = list(board)
    board_list[action] = current_player
    next_board = tuple(board_list)

    # Check if this move caused a k-in-a-row win
    if is_winner_k(next_board, current_player, n, k):
        return (next_board, current_player), True, REWARD_WIN

    # Check if draw (no empty spaces left)
    if all(x != 0 for x in next_board):
        return (next_board, current_player), True, REWARD_DRAW

    # Otherwise, game continues; switch player
    next_player = 1 if current_player == 2 else 2
    return (next_board, next_player), False, REWARD_STEP

# ======================
# 3) Q-Learning Storage
# ======================
Q = {}

def get_Q_value(state, action):
    """Return Q-value for (state, action) or 0.0 if not present."""
    return Q.get((state, action), 0.0)

def set_Q_value(state, action, value):
    """Set Q-value for (state, action)."""
    Q[(state, action)] = value

def choose_action_epsilon_greedy(state, n):
    """
    Choose an action using an epsilon-greedy policy based on Q-values.
    """
    global EPSILON

    legal_actions = get_legal_actions(state, n)

    # With probability EPSILON, choose a random action
    if random.random() < EPSILON:
        return random.choice(legal_actions)

    # Otherwise choose the action with the highest Q-value
    q_values = [get_Q_value(state, a) for a in legal_actions]
    max_q = max(q_values)
    best_actions = [a for (a, q) in zip(legal_actions, q_values) if q == max_q]
    return random.choice(best_actions)

def update_Q(state, action, reward, next_state, done):
    """
    Q-learning update rule:
        Q(s,a) = Q(s,a) + ALPHA * (reward + GAMMA * maxQ(s',.) - Q(s,a))
    If 'done' is True, we consider maxQ(s',.) = 0.
    """
    global GAMMA, ALPHA

    old_value = get_Q_value(state, action)
    if done:
        td_target = reward
    else:
        next_legal_actions = get_legal_actions(next_state, N)
        if len(next_legal_actions) == 0:
            td_target = reward
        else:
            next_q_values = [get_Q_value(next_state, a) for a in next_legal_actions]
            td_target = reward + GAMMA * max(next_q_values)

    new_value = old_value + ALPHA * (td_target - old_value)
    set_Q_value(state, action, new_value)

# ======================
# 4) Opponent Logic
# ======================
def random_player_move(state, n):
    """
    Given the current state and board size n, returns a random legal move.
    This function encapsulates the random opponent's strategy.
    """
    legal_actions = get_legal_actions(state, n)
    return random.choice(legal_actions)

# ======================
# 5) Training Loop Against a Specified Opponent
# ======================
def train_q_learning_against_opponent(episodes, opponent_move_func, print_interval=1000):
    """
    Train the Q-learning agent (always playing as player 1) against a fixed opponent.
    """
    win_count = 0
    loss_count = 0
    draw_count = 0

    for episode in range(episodes):
        board = create_empty_board(N)
        # Q-learning agent always starts as player 1.
        state = (board, 1)
        done = False
        last_player = None

        while not done:
            board, current_player = state
            last_player = current_player
            if current_player == 1:
                # Q-learning agent's turn
                action = choose_action_epsilon_greedy(state, N)
                next_state, done, reward = make_move(state, action, N, K)
                update_Q(state, action, reward, next_state, done)
            else:
                # Opponent's turn (e.g., random, etc.)
                action = opponent_move_func(state, N)
                next_state, done, reward = make_move(state, action, N, K)
            state = next_state

            if done:
                if reward == REWARD_WIN:
                    if last_player == 1:
                        win_count += 1
                    else:
                        loss_count += 1
                elif reward == REWARD_DRAW:
                    draw_count += 1
                break

        if (episode + 1) % print_interval == 0:
            total = win_count + loss_count + draw_count
            if total == 0:
                win_rate = loss_rate = draw_rate = 0.0
            else:
                win_rate = win_count / total
                loss_rate = loss_count / total
                draw_rate = draw_count / total

            print(f"Episode {episode+1}: win_rate={win_rate:.2f}, loss_rate={loss_rate:.2f}, draw_rate={draw_rate:.2f} "
                  f"(over last {print_interval} episodes)")
            win_count = 0
            loss_count = 0
            draw_count = 0

# ======================
# 6) Demo / Play
# ======================
def print_board(board, n):
    """
    Utility to print the board with an outline and dashed borders.
    0 -> ' ' (empty), 1 -> 'X', 2 -> 'O'
    """
    symbols = {0: '   ', 1: ' X ', 2: ' O '}
    matrix = np.array(board).reshape(n, n)
    horizontal_line = '+' + '---+' * n
    print(horizontal_line)
    for row in range(n):
        row_str = '|'
        for cell in matrix[row]:
            row_str += f"{symbols[cell]}|"
        print(row_str)
        print(horizontal_line)
    print()


def play_vs_opponent(opponent_move_func, n=N, k=K):
    """
    Have the trained agent (X=1) play against a given opponent (O=2).
    The opponent's strategy is implemented by 'opponent_move_func'.
    """
    board = create_empty_board(n)
    current_player = 1  # Agent (X) starts
    state = (board, current_player)

    print(f"\nStarting a new {n}x{n} game, needing {k} in a row to win.")
    print_board(board, n)

    done = False
    while not done:
        board, current_player = state

        if current_player == 1:
            # Agent's turn: pick best action greedily (epsilon=0 in test)
            legal_actions = get_legal_actions(state, n)
            q_values = [get_Q_value(state, a) for a in legal_actions]
            max_q = max(q_values)
            best_actions = [a for (a, q) in zip(legal_actions, q_values) if q == max_q]
            action = random.choice(best_actions)
        else:
            # Opponent's turn (could be random or any other strategy function)
            action = opponent_move_func(state, n)

        next_state, done, reward = make_move(state, action, n, k)
        state = next_state

        print_board(state[0], n)

        if done:
            if reward == REWARD_WIN:
                if current_player == 1:
                    print("X (Agent) wins!")
                else:
                    print("O (Opponent) wins!")
            else:
                print("It's a draw!")
            break

# ======================
# 7) Main Execution
# ======================
if __name__ == "__main__":
    # 1. Train Q-learning agent against random opponent.
    print("Training the agent against a random opponent...")
    train_q_learning_against_opponent(EPISODES, random_player_move)
    print("Training complete.")

    # 2. Play a few demo games against the random opponent strategy.
    for _ in range(3):
        play_vs_opponent(random_player_move, N, K)