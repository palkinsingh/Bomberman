import events as e
import numpy as np

def state_to_features(game_state: dict):
    """
    Convert the game state into a feature representation that the Q-learning algorithm can use.
    This version includes more detailed information.
    """
    if game_state is None:
        return None

    # Extract the relevant features
    position = np.array(game_state['self'][3])
    coins = np.array(game_state['coins'])
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = np.array([other[3] for other in game_state['others']])

    # Distance to the nearest coin
    nearest_coin_dist = min([np.sum(np.abs(coin - position)) for coin in coins]) if coins.size > 0 else 0
    
    # Is the agent in danger from a bomb?
    bomb_danger = 0
    for bomb_position, bomb_timer in bombs:
        if bomb_timer <= 2 and np.sum(np.abs(np.array(bomb_position) - position)) <= 3:
            bomb_danger = 1
            break
    
    # Distance to the nearest opponent
    nearest_opponent_dist = min([np.sum(np.abs(opponent - position)) for opponent in others]) if others.size > 0 else 0
    
    # Presence of obstacles around the agent
    left = game_state['field'][position[0] - 1, position[1]] if position[0] > 0 else -1
    right = game_state['field'][position[0] + 1, position[1]] if position[0] < game_state['field'].shape[0] - 1 else -1
    up = game_state['field'][position[0], position[1] - 1] if position[1] > 0 else -1
    down = game_state['field'][position[0], position[1] + 1] if position[1] < game_state['field'].shape[1] - 1 else -1

    # Include a feature to detect if the current position is a dead end (no escape routes)
    dead_end = int(left != 0 and right != 0 and up != 0 and down != 0)

    # Combine all features into a single tuple
    features = (position[0], position[1], nearest_coin_dist, bomb_danger, nearest_opponent_dist, left, right, up, down, dead_end)
    return features

import numpy as np
import events as e

def setup_training(self):
    """
    Setup for training the agent. Initialize Q-table and other necessary parameters.
    """
    self.q_table = {}
    self.alpha = 0.1  # Learning rate
    self.gamma = 0.9  # Discount factor
    self.epsilon = 1.0  # Start with full exploration
    self.epsilon_min = 0.1  # Minimum exploration rate
    self.epsilon_decay = 0.995  # Decay rate per episode

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
    """
    Called once per step to allow the agent to learn from the experience.
    """
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    if old_state not in self.q_table:
        self.q_table[old_state] = {action: 0 for action in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']}
    
    if new_state not in self.q_table:
        self.q_table[new_state] = {action: 0 for action in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']}
    
    reward = reward_from_events(events)
    
    old_q_value = self.q_table[old_state][self_action]
    future_reward = max(self.q_table[new_state].values())
    new_q_value = old_q_value + self.alpha * (reward + self.gamma * future_reward - old_q_value)
    self.q_table[old_state][self_action] = new_q_value

def end_of_round(self, last_game_state: dict, last_action: str, events):
    """
    Called at the end of each round to allow final updates.
    """
    game_events_occurred(self, last_game_state, last_action, None, events)
    
    total_reward = reward_from_events(events)
    
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

def reward_from_events(events) -> int:
    """
    Calculate the reward based on the events.
    """
    reward = 0
    for event in events:
        if event == e.COIN_COLLECTED:
            reward += 1
        elif event == e.KILLED_OPPONENT:
            reward += 5
        elif event == e.KILLED_SELF:
            reward -= 5
        elif event == e.MOVED_LEFT or event == e.MOVED_RIGHT or event == e.MOVED_UP or event == e.MOVED_DOWN:
            reward += 0.1  # Small reward for moving
        elif event == e.INVALID_ACTION:
            reward -= 1  # Penalize invalid actions
        elif event == e.BOMB_DROPPED:
            reward += 0.5  # Encourage bomb usage
        elif event == e.CRATE_DESTROYED:
            reward += 2  # Reward for destroying a crate
        # Comment out or remove this line
        # elif event == e.SAFE_ESCAPE_FROM_BOMB:
        #     reward += 3  # Reward for successfully escaping a bomb
    
    return reward





