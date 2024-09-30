import torch
import time
import numpy as np
import random
from collections import deque
from .utils import ReplayBuffer, ACTIONS, state_to_input, reward_from_events
from .train import DQN  # Import your DQN model from train.py

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

def setup(self):
    """Called once before a set of games to initialize data structures etc."""
    self.logger.debug('Successfully entered setup code')
    
    # Initialize bomb history and coordinate history
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    
    # Initialize timers and round counters
    self.ignore_others_timer = 0
    self.current_round = 0

    # Initialize the DQN agent (including networks, optimizer, etc.)
    self.gamma = 0.99
    self.epsilon = 1.0  # Exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.001
    self.batch_size = 64
    self.target_update_freq = 10  # Update target network every 10 steps

    # Initialize the Q-network and target network
    state_size = 15 * 15 * 4  # Based on your input size (game state)
    action_size = len(ACTIONS)
    self.q_network = DQN(state_size, action_size)
    self.target_network = DQN(state_size, action_size)
    self.target_network.load_state_dict(self.q_network.state_dict())

    # Optimizer and replay buffer
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    self.memory = ReplayBuffer(10000)

    # Set training mode
    self.train = True  # True for training, False for evaluation mode
    self.steps_done = 0

    # Timing mechanism for agent actions
    self.last_move_time = time.time()  # Keep track of the last move time
    self.move_interval = 1 / 10  # Ensure the agent moves 10 times per second

def reset_self(self):
    """Reset agent's state per round."""
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0

def is_near_crate(game_state):
    """Check if the agent is near any crates."""
    x, y = game_state['self'][3]  # Agent's coordinates
    field = game_state['field']  # The game field

    # Check surrounding tiles for crates (field value 1 represents crates)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if field[x + dx, y + dy] == 1:
            return True
    return False


def act(self, game_state):
    """This function is called every game step to determine the next action."""
    self.logger.info('Picking action according to DQN')

    # Check if enough time has passed for the next move (10 steps per second)
    current_time = time.time()
    if current_time - self.last_move_time < self.move_interval:
        return 'WAIT'

    # Update the last move time
    self.last_move_time = current_time

    # Convert game state to a format suitable for the DQN
    state = state_to_input(game_state)

    # Exploration vs. Exploitation decision
    if random.random() < self.epsilon and self.train:
        # Explore: choose a random action
        action_idx = random.randint(0, len(ACTIONS) - 1)
    else:
        # Exploit: choose the best action using the Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_idx = self.q_network(state_tensor).argmax().item()

    # Ensure bombs are only dropped near crates
    if ACTIONS[action_idx] == 'BOMB':
        if not is_near_crate(game_state):
            # If not near a crate, choose a different action
            action_idx = random.choice([i for i in range(len(ACTIONS)) if ACTIONS[i] != 'BOMB'])

    # Decay epsilon (if training)
    if self.train:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Return the chosen action
    return ACTIONS[action_idx]


def learn_from_experience(self):
    """Perform a learning step for the DQN agent using experience replay."""
    if len(self.memory) < self.batch_size:
        return  # Don't start training until we have enough samples

    # Sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

    # Convert to tensors
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    # Get current Q-values
    q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values
    with torch.no_grad():
        max_next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

    # Compute loss and update the model
    loss = torch.nn.MSELoss()(q_values, target_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Update the target network periodically
    if self.steps_done % self.target_update_freq == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())

def end_of_episode(self, last_game_state, last_action, events):
    """This function is called at the end of each episode."""
    # Calculate reward from events
    reward = reward_from_events(self, events)

    # Learn from the final experience of the episode
    next_state = np.zeros_like(last_game_state)  # Final state is terminal
    done = True  # The episode is over
    learn_from_experience(self, last_game_state, last_action, reward, next_state, done)

    # Print out statistics at the end of the episode
    self.logger.info(f"End of episode. Total steps: {self.steps_done}, Epsilon: {self.epsilon:.4f}")
