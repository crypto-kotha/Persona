import torch
import random
import numpy as np
from deap import base, creator, tools, algorithms
from dqn_agent import RainbowDQNAgent
from memory import PrioritizedReplayBuffer

# Create fitness class and individual type for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

class NeuralEvolution:
    def __init__(self, state_size, action_size, population_size=20, generations=10,
                hidden_size=64, learning_rate=0.0005, gamma=0.99, tau=0.001,
                double_dqn=True, dueling_dqn=True, prioritized_replay=True,
                noisy_nets=True, use_lstm=True, n_step=3):
        
        # Environment and DQN parameters
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.prioritized_replay = prioritized_replay
        self.noisy_nets = noisy_nets
        self.use_lstm = use_lstm
        self.n_step = n_step
        
        # Evolution parameters
        self.population_size = population_size
        self.generations = generations
        
        # Create template agent for architecture
        self.template_agent = RainbowDQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            batch_size=32,
            update_every=4,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn,
            prioritized_replay=prioritized_replay,
            noisy_nets=noisy_nets,
            use_lstm=use_lstm,
            n_step=n_step
        )
        
        # Setup DEAP toolbox
        self.toolbox = base.Toolbox()
        
        # Register parameter generation function
        self.param_ranges = self._get_parameter_ranges()
        self.param_sizes = self._get_parameter_sizes()
        self.total_params = sum(self.param_sizes.values())
        
        # Register genetic operators
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_float, n=self.total_params)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operations
        self.toolbox.register("evaluate", self.evaluate_agent)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self.experience_guided_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        self.population = self.toolbox.population(n=population_size)
        
        # Keep track of the best agent
        self.best_agent = None
        self.best_fitness = float('-inf')
        
        # Shared experience buffer
        self.shared_memory = PrioritizedReplayBuffer(
            action_size=action_size,
            buffer_size=100000,
            batch_size=32,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
            device="cpu",  # Adjust as needed
            n_step=n_step,
            gamma=gamma
        )
    
    def _get_parameter_ranges(self):
        """Define parameter ranges for mutation guidance"""
        return {
            'fc1_weight': (-0.1, 0.1),
            'fc1_bias': (-0.05, 0.05),
            'fc2_weight': (-0.1, 0.1),
            'fc2_bias': (-0.05, 0.05),
            'output_weight': (-0.1, 0.1),
            'output_bias': (-0.05, 0.05),
            'lstm_weight': (-0.1, 0.1) if self.use_lstm else None,
            'lstm_bias': (-0.05, 0.05) if self.use_lstm else None
        }
    
    def _get_parameter_sizes(self):
        """Get sizes of each parameter group (simplified example)"""
        # In a real implementation, this would reflect the actual network architecture
        sizes = {
            'fc1_weight': self.state_size * self.hidden_size,
            'fc1_bias': self.hidden_size,
            'fc2_weight': self.hidden_size * self.hidden_size,
            'fc2_bias': self.hidden_size,
            'output_weight': self.hidden_size * self.action_size,
            'output_bias': self.action_size
        }
        
        if self.use_lstm:
            sizes['lstm_weight'] = 4 * self.hidden_size * (self.hidden_size + self.state_size)
            sizes['lstm_bias'] = 4 * self.hidden_size
            
        return sizes
    
    def individual_to_agent(self, individual):
        """Convert an individual's genes to agent parameters"""
        agent = RainbowDQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            tau=self.tau,
            batch_size=32,
            update_every=4,
            double_dqn=self.double_dqn,
            dueling_dqn=self.dueling_dqn,
            prioritized_replay=self.prioritized_replay,
            noisy_nets=self.noisy_nets,
            use_lstm=self.use_lstm,
            n_step=self.n_step
        )
        
        # Load parameters from individual into agent
        # This is a simplified example - actual implementation would depend on agent's architecture
        index = 0
        param_dict = {}
        
        for name, size in self.param_sizes.items():
            param_dict[name] = np.array(individual[index:index+size]).reshape(
                self._get_param_shape(name)
            )
            index += size
        
        # Apply parameters to agent's network
        self._set_agent_params(agent, param_dict)
        
        # Share memory across agents
        agent.memory = self.shared_memory
        
        return agent
    
    def _get_param_shape(self, name):
        """Get the shape of a parameter tensor"""
        # Simplified - in real implementation, get actual shapes from model
        if name == 'fc1_weight':
            return (self.hidden_size, self.state_size)
        elif name == 'fc1_bias':
            return (self.hidden_size,)
        elif name == 'fc2_weight':
            return (self.hidden_size, self.hidden_size)
        elif name == 'fc2_bias':
            return (self.hidden_size,)
        elif name == 'output_weight':
            return (self.action_size, self.hidden_size)
        elif name == 'output_bias':
            return (self.action_size,)
        elif name == 'lstm_weight':
            return (4 * self.hidden_size, self.hidden_size + self.state_size)
        elif name == 'lstm_bias':
            return (4 * self.hidden_size,)
    
    def _set_agent_params(self, agent, param_dict):
        """Set parameters in the agent's networks"""
        # This is a simplified example
        # You would need to adapt this to the actual structure of your RainbowDQNAgent
        
        # Example implementation:
        with torch.no_grad():
            # Main network
            # FC1 layer
            agent.qnetwork_local.fc1.weight.data = torch.tensor(
                param_dict['fc1_weight'], dtype=torch.float32)
            agent.qnetwork_local.fc1.bias.data = torch.tensor(
                param_dict['fc1_bias'], dtype=torch.float32)
            
            # FC2 layer
            agent.qnetwork_local.fc2.weight.data = torch.tensor(
                param_dict['fc2_weight'], dtype=torch.float32)
            agent.qnetwork_local.fc2.bias.data = torch.tensor(
                param_dict['fc2_bias'], dtype=torch.float32)
            
            # Output layer
            agent.qnetwork_local.output.weight.data = torch.tensor(
                param_dict['output_weight'], dtype=torch.float32)
            agent.qnetwork_local.output.bias.data = torch.tensor(
                param_dict['output_bias'], dtype=torch.float32)
            
            # LSTM if used
            if self.use_lstm and hasattr(agent.qnetwork_local, 'lstm'):
                # LSTM parameters
                agent.qnetwork_local.lstm.weight_ih_l0.data = torch.tensor(
                    param_dict['lstm_weight'][:, :self.state_size], dtype=torch.float32)
                agent.qnetwork_local.lstm.weight_hh_l0.data = torch.tensor(
                    param_dict['lstm_weight'][:, self.state_size:], dtype=torch.float32)
                agent.qnetwork_local.lstm.bias_ih_l0.data = torch.tensor(
                    param_dict['lstm_bias'][:self.hidden_size*2], dtype=torch.float32)
                agent.qnetwork_local.lstm.bias_hh_l0.data = torch.tensor(
                    param_dict['lstm_bias'][self.hidden_size*2:], dtype=torch.float32)
            
            # Copy to target network
            agent.hard_update()
    
    def evaluate_agent(self, individual, env, n_episodes=3, max_steps=1000):
        """Evaluate an individual by testing its agent in the environment"""
        agent = self.individual_to_agent(individual)
        total_reward = 0
        
        for _ in range(n_episodes):
            state = env.reset()
            agent.reset_episode()
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.act(state, epsilon=0.05)  # Small exploration
                next_state, reward, done, _ = env.step(action)
                
                # Add experience to shared memory
                agent.step(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        avg_reward = total_reward / n_episodes
        return (avg_reward,)
    
    def extract_significant_experiences(self, n=100):
        """Extract most significant experiences from shared memory"""
        if len(self.shared_memory) < n:
            return [], []
        
        # Get indices of experiences with highest priority
        indices = np.argsort(self.shared_memory.priorities)[-n:]
        
        # Extract those experiences
        experiences = []
        weights = []
        
        for idx in indices:
            exp = self.shared_memory.memory[idx]
            priority = self.shared_memory.priorities[idx]
            experiences.append(exp)
            # Normalize weights
            weights.append(priority / sum(self.shared_memory.priorities))
        
        return experiences, weights
    
    def experience_guided_mutation(self, individual, indpb=0.1):
        """
        Mutate individual using guidance from experience replay buffer
        
        This mutation combines:
        1. Random mutations (standard approach)
        2. Experience-guided mutations based on TD errors
        """
        # Get significant experiences
        experiences, weights = self.extract_significant_experiences(n=50)
        
        if not experiences:
            # Fall back to standard mutation if no experiences
            for i in range(len(individual)):
                if random.random() < indpb:
                    individual[i] += random.gauss(0, 0.1)
            return individual,
        
        # Extract state transition information to guide mutation
        state_diffs = []
        reward_signals = []
        
        for exp in experiences:
            state, action, reward, next_state, done = exp
            # Calculate state transition difference
            state_diff = next_state - state
            state_diffs.append(state_diff)
            reward_signals.append(reward)
        
        # Calculate mutation direction from experiences
        mutation_direction = {}
        param_index = 0
        
        for param_name, size in self.param_sizes.items():
            # Skip if we don't want to use this parameter
            if self.param_ranges[param_name] is None:
                param_index += size
                continue
            
            # For each parameter group
            mutation_values = np.zeros(size)
            
            # Weight more by experiences with higher rewards
            for i, (state_diff, reward) in enumerate(zip(state_diffs, reward_signals)):
                influence = weights[i] * reward
                
                # Simple heuristic: use state differences to influence weight changes
                # This is a simplified approach - you might want to use more sophisticated methods
                if param_name.endswith('weight'):
                    for j in range(min(size, len(state_diff))):
                        mutation_values[j] += influence * state_diff[j % len(state_diff)]
                else:  # bias
                    for j in range(min(size, len(state_diff))):
                        mutation_values[j] += influence * 0.01 * state_diff[j % len(state_diff)]
            
            # Scale mutations to stay within parameter ranges
            low, high = self.param_ranges[param_name]
            scale = min(high - low, 0.1)  # Limit to 0.1 or range
            mutation_values = np.clip(mutation_values, -scale, scale)
            
            mutation_direction[param_name] = (param_index, size, mutation_values)
            param_index += size

        # Apply mutations with experience guidance
        for param_name, (start_idx, size, values) in mutation_direction.items():
            for i in range(size):
                if random.random() < indpb:
                    # Mix random mutation with experience-guided mutation
                    random_part = random.gauss(0, 0.05)
                    guided_part = values[i]
                    mutation = 0.5 * random_part + 0.5 * guided_part
                    
                    # Apply mutation
                    individual[start_idx + i] += mutation
                    
                    # Keep within reasonable bounds
                    individual[start_idx + i] = max(-1.0, min(1.0, individual[start_idx + i]))
        
        return individual,
    
    def evolve(self, env, n_generations):
        """Run the evolutionary process"""
        import torch  # Add this import
        
        # Initialize toolbox with evaluation function that includes environment
        self.toolbox.register("evaluate", lambda ind: self.evaluate_agent(ind, env))
        
        for gen in range(1, n_generations + 1):
            print(f"\n🧬 Generation {gen}/{n_generations}")
            
            # Select and clone individuals
            offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=0.5, mutpb=0.2)
            
            # Evaluate all individuals
            fits = self.toolbox.map(self.toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            # Select next generation
            self.population = self.toolbox.select(offspring, k=len(self.population))
            
            # Track best individual
            best_ind = tools.selBest(self.population, k=1)[0]
            best_fit = best_ind.fitness.values[0]
            
            if best_fit > self.best_fitness:
                self.best_fitness = best_fit
                self.best_agent = self.individual_to_agent(best_ind)
                print(f"🏆 New best fitness: {best_fit:.4f}")
            else:
                print(f"🔄 Best fitness: {best_fit:.4f}")
            
            # Print generation statistics
            fits = [ind.fitness.values[0] for ind in self.population]
            print(f"📊 Min: {min(fits):.4f}, Avg: {sum(fits)/len(fits):.4f}, Max: {max(fits):.4f}")
            
            # Occasionally train the best agent with standard DQN updates
            if gen % 3 == 0 and self.best_agent is not None:
                print("🔄 Fine-tuning best agent with standard DQN updates...")
                self._fine_tune_best_agent(env, episodes=5)
        
        print("\n✅ Evolution complete!")
        return self.best_agent
    
    def _fine_tune_best_agent(self, env, episodes=5):
        """Fine-tune best agent using standard DQN updates"""
        agent = self.best_agent
        
        for _ in range(episodes):
            state = env.reset()
            agent.reset_episode()
            
            while True:
                action = agent.act(state, epsilon=0.1)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                
                if done:
                    break
