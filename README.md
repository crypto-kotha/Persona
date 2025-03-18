# Persona

Persona is an advanced implementation of reinforcement learning techniques, utilizing Deep Q-Networks (DQN) and neural evolution strategies to create intelligent agents capable of learning and adapting to complex environments. This repository includes various modules that work together to provide a comprehensive framework for training and evolving neural networks.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
  - [DQN Agent](#dqn-agent)
  - [Memory](#memory)
  - [Neural Evolution](#neural-evolution)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project focuses on leveraging advanced reinforcement learning techniques to develop agents capable of mastering a variety of tasks. It combines Dueling DQN, Prioritized Experience Replay, Noisy Networks, and LSTM for temporal dependencies, along with a neural evolution strategy to optimize hyperparameters and network architectures.

## Features

- **Dueling DQN Architecture**: Separates value and advantage streams to improve learning efficiency.
- **Prioritized Experience Replay**: Prioritizes important experiences for more effective learning.
- **Noisy Networks**: Introduces stochasticity into the network for better exploration.
- **LSTM Integration**: Handles temporal dependencies in the input data.
- **Neural Evolution**: Uses genetic algorithms to evolve better-performing agents.
- **Extensive Logging and Visualization**: Provides detailed logs and visualizations of training metrics.
- 
## RainbowDQNAgent Agent Usage:

The `dqn_agent.py` defines the `RainbowDQNAgent` class, which implements the Rainbow DQN algorithm with various enhancements:
- **NoisyLinear**: A custom linear layer with added noise for exploration.
- **DuelingDQN**: Extends the DQN architecture to separate value and advantage streams.
- **PrioritizedReplayBuffer**: Manages experience replay with prioritized sampling.

### 1. **Train with DQN**:
```bash
import numpy as np
from datetime import datetime
from dqn_agent import RainbowDQNAgent, SimpleEnv

def train_agent():
    # Initialize environment and agent
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

if __name__ == "__main__":
    print(f"{datetime.utcnow()} - Starting training process")
    train_agent()
    print(f"{datetime.utcnow()} - Training process completed")
```

## 2. **Neural Evolution**: evolve agents and mutate last RainbowDQNAgent.

```bash
import torch
from dqn_agent import RainbowDQNAgent, SimpleEnv
from neural_evolution import NeuralEvolution

def genesis(state_size, action_size):
    """Run neuroevolution on a given environment"""
    
    # Create environment
    env = SimpleEnv(state_size=4, action_size=2)
    
    # Create neuroevolution framework
    neuro_evo = NeuralEvolution(
        state_size=state_size,
        action_size=action_size,
        population_size=20,
        generations=10,
        hidden_size=64,
        learning_rate=0.0005,
        gamma=0.99,
        double_dqn=True,
        dueling_dqn=True,
        prioritized_replay=True,
        noisy_nets=True,
        use_lstm=False  # Simplified for example
    )
    
    # Run evolution
    best_agent = neuro_evo.evolve(env, n_generations=10)
    
    # Test best agent
    state = env.reset()
    best_agent.reset_episode()
    total_reward = 0
    print("\n🎮 Testing best agent...")
    
    while True:
        action = best_agent.act(state, epsilon=0)  # No exploration during testing
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"🏆 Final score: {total_reward}")
    return best_agent
```

## Contributing

We welcome contributions to this project. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request. Make sure to follow the project's coding standards and guidelines.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
