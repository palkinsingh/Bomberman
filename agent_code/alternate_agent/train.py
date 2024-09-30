import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import ReplayBuffer, ACTIONS, state_to_input, reward_from_events

class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Change input_size to 1447
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



def setup_training(self):
    """Initialize the training process with all hyperparameters."""
    self.gamma = 0.99  # Discount factor
    self.epsilon = 1.0  # Exploration rate
    self.epsilon_min = 0.01  # Minimum exploration rate
    self.epsilon_decay = 0.995  # Exploration decay rate
    self.learning_rate = 0.001
    self.batch_size = 64
    self.target_update_freq = 10  # How often to update the target network
    state_size = 1447  # Set the correct input size based on state_to_input

    # Initialize DQN models
    state_size = 15 * 15 * 4  # Example state input size (15x15 board with 4 channels)
    action_size = len(ACTIONS)
    self.q_network = DQN(state_size, action_size)
    self.target_network = DQN(state_size, action_size)
    self.target_network.load_state_dict(self.q_network.state_dict())

    # Optimizer and replay buffer
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    self.memory = ReplayBuffer(10000)

    # Initialize variables
    self.steps_done = 0

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Store experience after each step and learn from it."""
    # Convert game states to neural network input format
    state = state_to_input(old_game_state)
    next_state = state_to_input(new_game_state)

    # Get the reward from events
    reward = reward_from_events(self, events)
    
    # Find index of the action
    action_idx = ACTIONS.index(self_action)
    
    # Store the experience in replay buffer
    self.memory.push(state, action_idx, reward, next_state, False)
    
    # Learn from experience
    if len(self.memory) > self.batch_size:
        learn_from_experience(self)

def end_of_round(self, last_game_state, last_action, events):
    """Called at the end of each round to finalize the learning process."""
    # Calculate the final reward
    reward = reward_from_events(self, events)

    # Convert state to input format
    state = state_to_input(last_game_state)
    
    # Store the final experience as terminal
    action_idx = ACTIONS.index(last_action)
    self.memory.push(state, action_idx, reward, np.zeros_like(state), True)

    # Perform learning step after the episode ends
    learn_from_experience(self)

def learn_from_experience(self):
    """Sample from replay buffer and train the network."""
    if len(self.memory) < self.batch_size:
        return  # Not enough samples to learn

    # Sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
    
    # Convert to torch tensors
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    # Compute Q-values for current states
    current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values
    with torch.no_grad():
        next_max_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_max_q_values * (1 - dones)
    
    # Compute loss
    loss = torch.nn.MSELoss()(current_q_values, target_q_values)
    
    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Update target network periodically
    self.steps_done += 1
    if self.steps_done % self.target_update_freq == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())

def reward_from_events(self, events):
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
