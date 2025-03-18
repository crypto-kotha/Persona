import numpy as np
import random
from collections import namedtuple, deque
import torch

# Define the Experience tuple structure
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples with prioritized sampling."""

    def __init__(self, action_size, buffer_size=100000, batch_size=32, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5, device="cpu", n_step=1, gamma=0.99):
        """Initialize a PrioritizedReplayBuffer object.
        
        Parameters:
        -----------
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        alpha (float): determines how much prioritization is used (0 = no prioritization, 1 = full prioritization)
        beta (float): importance-sampling correction factor (0 = no correction, 1 = full correction)
        beta_increment (float): increment for beta parameter
        epsilon (float): small constant to avoid zero probabilities
        device (str): device to use for tensor operations
        n_step (int): number of steps for n-step returns
        gamma (float): discount factor
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        
        # Initialize buffers
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # For n-step returns
        if n_step > 1:
            self.n_step_buffer = deque(maxlen=n_step)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Convert everything to numpy arrays if they aren't already
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()
            
        # For n-step returns
        if self.n_step > 1:
            self.n_step_buffer.append((state, action, reward, next_state, done))
            
            if len(self.n_step_buffer) < self.n_step:
                return
                
            # Get n-step return
            state, action = self.n_step_buffer[0][0:2]
            
            # Calculate n-step discounted reward
            n_reward = 0
            for i in range(self.n_step):
                n_reward += self.gamma**i * self.n_step_buffer[i][2]
                
            next_state = self.n_step_buffer[-1][3]
            done = self.n_step_buffer[-1][4]
            
            # Only proceed to add to memory if the earliest experience in buffer is done
            if done and len(self.n_step_buffer) < self.n_step:
                # Clear buffer
                self.n_step_buffer.clear()
                return
                
            experience = Experience(state, action, n_reward, next_state, done)
        else:
            experience = Experience(state, action, reward, next_state, done)
        
        # Get max priority for new experience
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        # Add experience to memory
        if len(self.memory) < self.buffer_size:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            
        # Update priority
        self.priorities[self.position] = max_priority
        
        # Update position and size
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self):
        """Sample a batch of experiences from memory with prioritization."""
        if self.size < self.batch_size:
            return None
            
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)
        
        # Sample indices according to probabilities
        indices = np.random.choice(self.size, self.batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        experiences = [self.memory[idx] for idx in indices]
        
        # Convert to PyTorch tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        """Return the current size of memory."""
        return self.size
