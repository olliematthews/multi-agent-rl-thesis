import numpy as np
from environment import Cyclist
import pickle
from agent import Agent
import matplotlib.pyplot as plt
# Simulation Parameters
NUM_EPISODES = 4000
n_seeds = 10

env_params = {
    'vel_min' : 0,
    'vel_max' : 4,
    'cD': 0.2,
    'race_length': 250,
    'time_limit' : 200
    }

def normalise_state(state, env_params):
    new_state = np.ones(state.size + 1)
    range = np.array([env_params['race_length'], 4, 100, env_params['time_limit']])
    new_state[0:-1] = (2 * state - range) / range
    return new_state



# Hyperparameters
LEARNING_RATE = [1e-2]
GAMMA = 0.99
epsilon = 0.98
epsilon_final = 0.01
epsilon_decay = (epsilon / epsilon_final) ** (-10000/NUM_EPISODES)
alpha_mean = 0.1
alpha_std = 0.1
iterator_rewards = [None] * len(LEARNING_RATE)
seed_entropies = [None] * n_seeds
seed_velocities = [None] * n_seeds
batch_size = 4


for i in range(len(LEARNING_RATE)):
    seed_rewards = [None] * n_seeds

    for seed in range(n_seeds):
        # Create gym and seed numpy
        cyclist = Cyclist(env_params)
        nA = cyclist.action_space
        nx = cyclist.state_space + 1
        np.random.seed(seed)
        agent = Agent(lr = LEARNING_RATE[i], env_params = env_params, batch_size = batch_size, seed = seed)
        
        # Keep stats for final print of graph
        episode_rewards = []
        episode_entropies = []
        episode_velocities = []
        # Main loop 
        # Make sure you update your weights AFTER each episode
        for e in range(int(NUM_EPISODES//batch_size)):
            for batch in range(batch_size):
                state = normalise_state(cyclist.reset(),env_params)
            
            	# Keep track of game score to print
                score = 0
                done = False
                step = 0
                mean_entropy = 0
                mean_velocity = 0
                while not done:
            
            		# Sample from policy and take action in environment
                    alpha = 1/(step + 1)
                    action, entropy = agent.choose_action(state)
                    next_state,reward,done = cyclist.step(action - 1,env_params)
                    mean_velocity = mean_velocity * (1 - alpha) + next_state[1] * alpha
                    next_state = normalise_state(next_state, env_params)
                    agent.store_transition(state, action, reward)
                    state = next_state
                    score+=reward
                    step += 1
                    mean_entropy = mean_entropy * (1 - alpha) + entropy * alpha
                    if done:
                        break
                
                agent.learn()
            	# Append for logging and print
                episode_rewards.append(score) 
                episode_entropies.append(mean_entropy)
                episode_velocities.append(mean_velocity)

            print(f'Iteration: {i}, Seed: {seed}, EP: {e}, Score: {score}',flush = False)
            print()
        
        seed_rewards[seed] = np.array(episode_rewards)
        seed_entropies[seed] = episode_entropies
        seed_velocities[seed] = episode_velocities
    iterator_rewards[i] = seed_rewards.copy()
pickle.dump([iterator_rewards, seed_entropies, seed_velocities] , open('Boop_moop_sloop.p','wb'))