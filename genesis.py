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
