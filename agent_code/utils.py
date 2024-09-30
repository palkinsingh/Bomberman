import numpy as np
from collections import deque
import random

# Define actions for the game
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

class ReplayBuffer:
    """Replay Buffer to store transitions for experience replay."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)

def state_to_input(game_state):
    """Convert the game state to a format suitable for the DQN model."""
    field = game_state['field'].flatten()
    
    # Bombs position encoded as a binary grid
    bombs = np.zeros_like(field)
    for (x, y), _ in game_state['bombs']:
        bombs[x * len(game_state['field']) + y] = 1

    # Coins position encoded as a binary grid
    coins = np.zeros_like(field)
    for x, y in game_state['coins']:
        coins[x * len(game_state['field']) + y] = 1

    # Self position and other agents' positions encoded
    self_info = game_state['self'][3]  # (x, y) coordinates of the agent
    others = np.zeros_like(field)
    for _, _, _, (x, y) in game_state['others']:
        others[x * len(game_state['field']) + y] = 1

    # Crates position encoded as a binary grid
    crates = np.zeros_like(field)
    for x in range(game_state['field'].shape[0]):
        for y in range(game_state['field'].shape[1]):
            if game_state['field'][x, y] == 1:
                crates[x * len(game_state['field']) + y] = 1

    # Concatenate all these features into a single input vector
    return np.concatenate((field, bombs, coins, others, crates, self_info))

def reward_from_events(events):
    """Assign rewards based on game events."""
    reward = 0
    if 'COIN_COLLECTED' in events:
        reward += 1
    if 'BOMB_DROPPED' in events:
        reward += 0.5
    if 'KILLED_OPPONENT' in events:
        reward += 2
    if 'INVALID_ACTION' in events:
        reward -= 1
    if 'KILLED_SELF' in events:
        reward -= 5
    return reward
