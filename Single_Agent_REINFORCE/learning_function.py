import numpy as np
import matplotlib.pyplot as plt
from environment import Cyclist
import pickle


# Simulation Parameters
sim_params = {
    'n_episodes' : 4000y,
    'n_seeds' : 10,
    'print_rewards' : True
    }

env_params = {
    'cD' : 0.2,
    'vel_max' : 4,
    'vel_min' : 0,
    'race_length' : 300,
    'time_limit' : 200
    }


# Hyperparameters
hyper_params = {
    'learning_rate' : 0.05,
    'batch_size' : 3,
    'gamma' : 0.99,
    # Epsilon starts at 0.98. It decays to epsilon_final in decay_episodes episodes
    'epsilon_final' : 0.01,
    'epsilon_0' : 0.98,
    'decay_episodes' : 100,
    'exploration' : False,
    'alpha_mean' : 0.1,
    'use_std' : True,
    'use_mean' : True,
    'true_mean' : False,
    'true_std' : False,
    'alpha_std' : 0.1
    }


env_params = {
        'cD' : 0.2,
        'vel_max' : 4,
        'vel_min' : 0,
        'race_length' : 400,
        'time_limit' : 200
        }

# Our policy that maps state to action parameterized by w
def policy(state,w):
	z = state.dot(w)
	exp = np.exp(z - np.max(z))
	return (exp/np.sum(exp))

# Vectorized softmax Jacobian
def softmax_grad(probs, action, state):
    probs = probs.reshape(1,-1)
    probs[0,action] -= 1
    probs = - probs
    return state.reshape(-1,1) @ probs


def normalise_state(state, env_params):
    range = np.array([env_params['race_length'], env_params['vel_max'], 100, env_params['time_limit']])
    return (2 * state - range) / range

def get_entropy(probs):
    return - np.sum(probs * np.log(probs)) 



# Initialisations
n_iterations = 1
labels = []

nearest_factor = (sim_params['n_episodes'] // hyper_params['batch_size']) * hyper_params['batch_size']
Seed_rewards = np.empty([n_iterations,sim_params['n_seeds'],nearest_factor])
Seed_entropies = np.empty([n_iterations,sim_params['n_seeds'],nearest_factor])
Seed_mean_velocities = np.empty([n_iterations,sim_params['n_seeds'],nearest_factor])

Average_rewards = np.empty([n_iterations,sim_params['n_episodes']])
Average_entropies = np.empty([n_iterations,sim_params['n_episodes']])
Std_rewards = np.zeros_like(Average_rewards)
Std_entropies = np.zeros_like(Average_entropies)
high_score = 0


for iterator in range(n_iterations):
    # Hyperparameters
    decay_episodes = hyper_params['decay_episodes']
    learning_rate = hyper_params['learning_rate']
    batch_size = hyper_params['batch_size']
    gamma = hyper_params['gamma']
    epsilon_0 = hyper_params['epsilon_0']
    epsilon_final = hyper_params['epsilon_final']
    alpha_mean = hyper_params['alpha_mean']
    alpha_std = hyper_params['alpha_std']
    use_std = hyper_params['use_std']
    use_mean = hyper_params['use_mean']
    epsilon_decay = (epsilon_0 / epsilon_final) ** (-1 / decay_episodes)
    labels.append('Batch Size of ' + str(batch_size))
    exploration = hyper_params['exploration']
    true_mean = hyper_params['true_mean']
    true_std = hyper_params['true_std']
    for seed in range(sim_params['n_seeds']):
        print(seed)
        # Create gym and seed numpy
        cyclist = Cyclist(env_params)
        nA = cyclist.action_space
        nx = cyclist.state_space + 1
        np.random.seed(seed)
        
        rolling_average_reward = np.zeros(201)
        rolling_std_dev = 0
        
        # Init weight
        w = np.random.normal(0,0.1,(nx,nA))
        delta_w = np.zeros_like(w)
        epsilon = epsilon_0
    
        # Keep stats for final print of graph
        episode_rewards = []
        mean_velocities = []
        entropies = []
        
        # Main loop 
        for e in range(int(sim_params['n_episodes'] / batch_size)):
        # Make sure you update your weights AFTER each episode
        
        	# Keep track of game score to print
            for i in range(batch_size):
                state = np.squeeze(np.append(normalise_state(cyclist.reset(),env_params),1))
                grads = []	
                rewards = []
                score = 0
                step = 0
                entropy = 0
                mean_velocity = 0
                while True:
            		# Sample from policy and take action in environment
                    probs = np.array(policy(state,w))
                    entropy += get_entropy(probs)
                    # With probability epsilon, draw from a uniform distribution
                    if np.random.rand() < epsilon and exploration:
                        action = np.random.choice(nA, p = [1 / nA] * nA)
                    # Else draw from the probability distribution
                    else:
                        action = np.random.choice(nA,p = np.squeeze(probs))
            
                    next_state,reward,done = cyclist.step(action - 1,env_params)
                    mean_velocity = mean_velocity * (1 - 1 / (1 + step)) + 1 / (1 + step) * next_state[1]
                    next_state = np.append(normalise_state(next_state,env_params),1)
                    next_state = next_state[None,:]
            
            		# Compute gradient and save with reward in memory for our weight updates
                    grad = softmax_grad(probs, action, state)
                    
                    grads.append(grad)
                    rewards.append(reward)		
                    score+=reward
                    step += 1
            		# Dont forget to update your old state to the new state
                    state = next_state

                    if done:
                        break
                
                # Save the best parameters
                if score > high_score:
                    high_score = score
                    w_best = w
                episode_rewards.append(score) 
                entropies.append(entropy / step)
                mean_velocities.append(mean_velocity)

                rewards = rewards[:step]
                state_reward = np.zeros_like(rewards)
                if true_mean:
                    alpha_mean = 1 / (1 + e)
                if true_std:
                    alpha_std = 1 / (1 + e)
                # Update parameters
                for i in range(len(rewards)):
            		# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
                    state_reward[i] = sum([ r * (hyper_params['gamma'] ** t) for t,r in enumerate(rewards[i:])])
                if use_mean:
                    rolling_average_reward[:len(state_reward)] = rolling_average_reward[:len(state_reward)] * (1 - alpha_mean) + alpha_mean * state_reward
                    advantage = state_reward - rolling_average_reward[:len(state_reward)]
                else:
                    advantage = state_reward
                if use_std:
                    rolling_std_dev = np.sqrt(rolling_std_dev ** 2 * (1 - alpha_std) + alpha_std * (np.mean(advantage)) ** 2)
                    advantage /= (rolling_std_dev + 0.01)
                
                # Subtract mean (baselining) and then divide by rolling std
                delta_w += np.sum(learning_rate * (grads * advantage[:,None,None]) / batch_size, axis = 0)
            w += delta_w
            delta_w = np.zeros_like(w)
            epsilon *= epsilon_decay
        	# Append for logging and print
            if sim_params['print_rewards']:
                print(f'Iteration: {iterator}, Seed: {seed}, EP: {e}, Score: {score}',flush = False)
                print()
        Seed_entropies[iterator,seed,:] = np.array(entropies)
        Seed_rewards[iterator,seed,:] = np.array(episode_rewards)
        Seed_mean_velocities[iterator,seed,:] = np.array(mean_velocities)

params = {'Env': env_params, 'Hyp' : hyper_params, 'Sim' : sim_params}

pickle.dump([w_best, Seed_rewards, Seed_entropies, Seed_mean_velocities, params], open('Optimal_PG.p','wb'))
