import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import matplotlib.pyplot as plt
from memory import PrioritizedReplayBuffer

print("🔧 Importing libraries and setting up environment...")

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    print(f"🔒 Random seeds set to {seed}")

set_seeds()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        print(f"📐 Creating NoisyLinear layer with {in_features} inputs and {out_features} outputs")
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        print(f"🔄 Resetting parameters for NoisyLinear layer")
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
        
    def reset_noise(self):
        print(f"🎲 Generating new noise for NoisyLinear layer")
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(torch.ger(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
        
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, noisy=True, use_lstm=True):
        super(DuelingDQN, self).__init__()
        print(f"🏗️ Building DuelingDQN with state_size={state_size}, action_size={action_size}, hidden_size={hidden_size}")
        
        self.noisy = noisy
        self.use_lstm = use_lstm
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        print("✅ Feature extraction layers created")
        
        if use_lstm:
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            print("✅ LSTM layer created")
            
        if noisy:
            self.advantage_hidden = NoisyLinear(hidden_size, hidden_size)
            self.advantage = NoisyLinear(hidden_size, action_size)
            self.value_hidden = NoisyLinear(hidden_size, hidden_size)
            self.value = NoisyLinear(hidden_size, 1)
            print("✅ Advantage and Value streams created with NoisyLinear layers")
        else:
            self.advantage_hidden = nn.Linear(hidden_size, hidden_size)
            self.advantage = nn.Linear(hidden_size, action_size)
            
            self.value_hidden = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, 1)
            print("✅ Advantage and Value streams created with standard Linear layers")
            
    def reset_noise(self):
        if not self.noisy:
            return
        print("🎲 Resetting noise in all NoisyLinear layers")
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        self.value_hidden.reset_noise()
        self.value.reset_noise()
        
    def forward(self, state, hidden=None):
        x = self.features(state)
        print(f"👁️ Features extracted with shape: {x.shape}")
        
        if self.use_lstm:
            if len(x.shape) == 2: 
                x = x.unsqueeze(1)
            x, hidden = self.lstm(x, hidden)
            x = x.squeeze(1) 
            print(f"🧠 LSTM processed with output shape: {x.shape}")
        
        advantage_hidden = F.relu(self.advantage_hidden(x))
        value_hidden = F.relu(self.value_hidden(x))
        
        advantage = self.advantage(advantage_hidden)
        value = self.value(value_hidden)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        print(f"🎯 Q-values calculated with shape: {q_values.shape}")
        
        if self.use_lstm:
            return q_values, hidden
        return q_values

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        print(f"📦 Creating PrioritizedReplayBuffer with capacity={capacity}, alpha={alpha}, beta={beta}")
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = np.ones(capacity, dtype=np.float32) 
        self.position = 0
        self.size = 0
        self.alpha = alpha 
        self.beta = beta   
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        print(f"📝 Adding experience to PER - Action: {action}, Reward: {reward}, Done: {done}")
        
        if self.size < self.capacity:
            self.memory.append(None) 
            self.size += 1
        
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        print(f"🎲 Sampling batch of size {batch_size} from PER")
        self.beta = min(1.0, self.beta + self.beta_increment)
        n_samples = min(self.size, self.capacity)
        priorities = self.priorities[:n_samples]
        p = priorities ** self.alpha
        p = p / np.sum(p)
        indices = np.random.choice(n_samples, batch_size, p=p, replace=False)
        weights = (n_samples * p[indices]) ** (-self.beta)
        weights = weights / weights.max()
        samples = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        print(f"📊 Sampled with importance weights: min={weights.min():.4f}, max={weights.max():.4f}")
        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, indices, td_errors):
        print(f"🔄 Updating priorities for {len(indices)} samples")
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-5)
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.size

class RainbowDQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.0003, 
                 gamma=0.99, tau=0.001, batch_size=64, update_every=4, 
                 double_dqn=True, dueling_dqn=True, prioritized_replay=True, 
                 noisy_nets=True, use_lstm=True, n_step=3):
        
        print(f"🤖 Initializing RainbowDQNAgent with state_size={state_size}, action_size={action_size}")
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.noisy_nets = noisy_nets
        self.n_step = n_step
        self.use_lstm = use_lstm
        """
        self.memory = PrioritizedReplayBuffer(
            action_size=action_size,
            buffer_size=100000, 
            batch_size=batch_size,
            alpha=0.6,  
            beta=0.4,  
            beta_increment=0.001,
            device=device,
            n_step=n_step,  
            gamma=gamma 
        )
        """
        self.memory = PrioritizedReplayBuffer(
            capacity=100000,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001
        )

        print(f"🌈 Rainbow features:")
        print(f"  - Double DQN: {'✅' if double_dqn else '❌'}")
        print(f"  - Dueling DQN: {'✅' if dueling_dqn else '❌'}")
        print(f"  - Prioritized Replay: {'✅' if prioritized_replay else '❌'}")
        print(f"  - Noisy Networks: {'✅' if noisy_nets else '❌'}")
        print(f"  - N-step Learning (n={n_step}): ✅")
        print(f"  - LSTM Network: {'✅' if use_lstm else '❌'}")
        
        self.q_network = DuelingDQN(state_size, action_size, hidden_size, noisy_nets, use_lstm).to(device)
        self.target_network = DuelingDQN(state_size, action_size, hidden_size, noisy_nets, use_lstm).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() 
        
        print("✅ Q-network and Target network created")
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(capacity=100000)
        else:
            self.memory = deque(maxlen=100000)
            print("📦 Created standard replay buffer with capacity=100000")
            
        self.n_step_buffer = deque(maxlen=n_step)
        self.loss_list = []
        self.t_step = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.hidden = None
        
    def reset_episode(self):
        print("🔄 Resetting episode state")
        self.current_episode_reward = 0
        self.n_step_buffer.clear()
        if self.use_lstm:
            self.hidden = None
        
    def step(self, state, action, reward, next_state, done):
        print(f"👣 Step - Action: {action}, Reward: {reward}, Done: {done}")
        
        self.current_episode_reward += reward
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step and not done:
            return
            
        state, action, cumulative_reward, next_state, done = self.get_n_step_info()
        
        if self.prioritized_replay:
            self.memory.add(state, action, cumulative_reward, next_state, done)
        else:
            self.memory.append((state, action, cumulative_reward, next_state, done))
            
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            print(f"📊 Episode completed with total reward: {self.current_episode_reward}")
            self.reset_episode()
            
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.learn()
            
    def get_n_step_info(self):
        """Get the n-step return information"""
        print(f"📏 Calculating {self.n_step}-step return")
        
        state, action, cumulative_reward, _, _ = self.n_step_buffer[0]
        
        for idx in range(1, len(self.n_step_buffer)):
            r = self.n_step_buffer[idx][2]  
            cumulative_reward += r * (self.gamma ** idx)
            
        next_state, _, _, _, done = self.n_step_buffer[-1]
        
        return state, action, cumulative_reward, next_state, done
        
    def act(self, state, epsilon=0.0):
        """Choose an action given state using epsilon-greedy or just network if noisy"""
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.q_network.eval()
        
        with torch.no_grad():
            if self.use_lstm:
                q_values, self.hidden = self.q_network(state, self.hidden)
            else:
                q_values = self.q_network(state)
                
        self.q_network.train()
        
        if self.noisy_nets:
            action = np.argmax(q_values.cpu().data.numpy())
            print(f"🎯 Selected action {action} using noisy network")
        else:
            if random.random() > epsilon:
                action = np.argmax(q_values.cpu().data.numpy())
                print(f"🎯 Selected action {action} using Q-network (epsilon-greedy, exploit)")
            else:
                action = random.choice(np.arange(self.action_size))
                print(f"🎲 Selected random action {action} (epsilon-greedy, explore)")
                
        return action
        
    def learn(self):
        """Update policy network parameters using batch of experiences"""
        print("\n📚 Learning from experiences...")

        if len(self.memory) < self.batch_size:
            return
    
        if self.noisy_nets:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
    
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, indices, is_weights = self.memory.sample(self.batch_size)
            is_weights = torch.FloatTensor(is_weights).to(device)
        else:
            print(f"🎲 Sampling batch of size {self.batch_size} from standard replay buffer")
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            indices = None
            is_weights = 1.0
        
        state_batch = torch.FloatTensor(np.array(states)).to(device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(device)
        done_batch = torch.BoolTensor(dones).unsqueeze(1).to(device)
        
        print(f"📊 Batch shapes - States: {state_batch.shape}, Actions: {action_batch.shape}, Rewards: {reward_batch.shape}")
        
        if self.use_lstm:
            current_q_values, _ = self.q_network(state_batch)
        else:
            current_q_values = self.q_network(state_batch)
            
        current_q_values = current_q_values.gather(1, action_batch)
        
        with torch.no_grad():
            if self.double_dqn:
                if self.use_lstm:
                    next_q_values_online, _ = self.q_network(next_state_batch)
                else:
                    next_q_values_online = self.q_network(next_state_batch)
                    
                next_actions = next_q_values_online.max(1)[1].unsqueeze(1)
                
                if self.use_lstm:
                    next_q_values_target, _ = self.target_network(next_state_batch)
                else:
                    next_q_values_target = self.target_network(next_state_batch)
                    
                max_next_q_values = next_q_values_target.gather(1, next_actions)
                print("🔄 Using Double DQN for target computation")
            else:
                if self.use_lstm:
                    next_q_values, _ = self.target_network(next_state_batch)
                else:
                    next_q_values = self.target_network(next_state_batch)
                    
                max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
                print("🎯 Using standard DQN for target computation")
                
            gamma_n = self.gamma ** self.n_step
            target_q_values = reward_batch + (~done_batch) * gamma_n * max_next_q_values
            
 
        td_error = target_q_values - current_q_values
        
        if self.prioritized_replay:
            loss = (td_error.pow(2) * is_weights).mean()
            print(f"📊 Loss with importance sampling: {loss.item():.4f}")
            
            self.memory.update_priorities(indices, td_error.abs().detach().cpu().numpy())
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
            print(f"📊 Loss: {loss.item():.4f}")

        self.memory.update_priorities(indices, td_error.abs().detach().cpu().numpy())
        self.loss_list.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        self.soft_update()
        
    def soft_update(self):
        """Soft update of target network from policy network"""
        print(f"🔄 Soft updating target network with τ={self.tau}")
        for target_param, policy_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
            
    def save(self, filename):
        """Save the model"""
        print(f"💾 Saving model to {filename}")
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_list': self.loss_list,
            'episode_rewards': self.episode_rewards
        }
        torch.save(checkpoint, filename)
        
    def load(self, filename):
        """Load the model"""
        print(f"📂 Loading model from {filename}")
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_list = checkpoint['loss_list']
        self.episode_rewards = checkpoint['episode_rewards']
        
    def plot_metrics(self):
        """Plot training metrics"""
        print("📊 Plotting training metrics...")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_list)
        plt.title('Loss over time')
        plt.xlabel('Training steps')
        plt.ylabel('Loss')
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_rewards)
        plt.title('Rewards per episode')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.tight_layout()
        plt.show()

class SimpleEnv:
    def __init__(self, state_size=4, action_size=2):
        print(f"🌍 Creating SimpleEnv with state_size={state_size}, action_size={action_size}")
        self.state_size = state_size
        self.action_size = action_size
        self.reset()
        
    def reset(self):
        print("🔄 Resetting environment")
        self.state = np.random.rand(self.state_size)
        self.steps = 0
        return self.state
        
    def step(self, action):
        print(f"👣 Environment step with action {action}")
        self.state = np.clip(self.state + 0.1 * (np.random.rand(self.state_size) - 0.5) + 0.1 * (action / self.action_size), 0, 1)
        reward = -np.sum((self.state - 0.5)**2)
        self.steps += 1
        done = self.steps >= 100
        
        if done:
            print("🏁 Episode complete")
            
        return self.state, reward, done, {}

if __name__ == "__main__":
    print("\n🚀 Starting Advanced DQN Test...")
    
    env = SimpleEnv(state_size=4, action_size=2)
    
    agent = RainbowDQNAgent(
        state_size=4,
        action_size=2,
        hidden_size=64,
        learning_rate=0.0005,
        gamma=0.99,
        tau=0.001,
        batch_size=32,
        update_every=4,
        double_dqn=True,
        dueling_dqn=True,
        prioritized_replay=True,
        noisy_nets=True,
        use_lstm=True,
        n_step=3
    )
    
    num_episodes = 3
    epsilon = 0.1 
    
    print(f"\n🏋️‍♀️ Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_episode()
        score = 0
        
        print(f"\n🎮 Episode {episode+1}/{num_episodes} starting...")
        
        while True:
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
                
        print(f"🏁 Episode {episode+1} completed with score: {score:.2f}")
    
    print("\n✅ Training complete!")
    print(f"📊 Final model statistics:")
    print(f"  - Episodes completed: {len(agent.episode_rewards)}")
    print(f"  - Training steps: {len(agent.loss_list)}")
    print(f"  - Final loss: {agent.loss_list[-1] if agent.loss_list else 'N/A'}")
    print(f"  - Average reward (last 3 episodes): {np.mean(agent.episode_rewards[-3:]) if len(agent.episode_rewards) >= 3 else 'N/A'}")
