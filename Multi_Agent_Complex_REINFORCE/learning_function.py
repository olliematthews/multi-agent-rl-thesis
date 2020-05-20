import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
import pickle
from scipy.ndimage import gaussian_filter
from multiprocessing import Process, Pool
import sys

# Simulation Parameters
sim_params = {
    'n_episodes' : 3000,
    'n_seeds' : 10,
    'n_cyclists' : 4
    }

env_params = {
    'cD' : {
            'a': 2.2,
            'b': 1,
            'cDrag': 0.2
            },
    'vel_max' : 4,
    'vel_min' : 0,
    'race_length' : 300,
    'time_limit' : 200
    }


# Hyperparameters
hyper_params = {
    'learning_rate' : 5e-5,
    'batch_size' : 5,
    'gamma' : 0.99,
    # Epsilon starts at 0.98. It decays to epsilon_final in decay_episodes episodes
    'epsilon_final' : 0.01,
    'epsilon_0' : 0.98,
    'decay_episodes' : 500,
    'repeat_explore' : False,
    'use_std' : True,
    'alpha_mean' : 0.1,
    'alpha_std' : 0.1,
    'lam' : [0]
    }



# Our policy that maps state to action parameterized by w
def policy(state,w):
	z = state.dot(w)
	exp = np.exp(z - np.max(z))
	return (exp/np.sum(exp))

# Vectorized softmax Jacobian
def softmax_grad(probs, action, state, model, lam):
    probs = probs.reshape(1,-1)
    probs[0,action] -= 1
    probs = - probs
    return state.reshape(-1,1) @ probs - lam * model


def normalise_state(state, env_params):
    out_state = np.ones([state.size + 1])
    out_state[:state.size] = state / np.array([env_params['race_length'], env_params['vel_max'], 100,env_params['race_length'] / 4,env_params['vel_max']])
    return out_state

def get_entropy(probs):
    return - np.sum(probs * np.log(probs + 1e-4)) 


def run_seed(seed_params):
    # Create gym and seed numpy
    seed = seed_params['seed']
    env_params = seed_params['Env']
    high_score = 0
    actions = [0] * sim_params['n_cyclists']
    entropy_kick = 0.4
    env = Environment(env_params, n_cyclists = seed_params['n_cyclists'])
    action_space = env.action_space
    nA = len(action_space)
    nx = env.state_space + 1
    np.random.seed(seed)
    exploration = []
    epsilon = seed_params['epsilon_0']

    # Note this should be increased for longer runs
    rolling_average_reward = np.zeros(env_params['time_limit'] + 1)
    rolling_std_dev = np.flip(np.arange(env_params['time_limit'] + 1) * 0).astype(float)
    
    # Init weights
    model = np.random.normal(0,0.1,(nx,nA))
    model_delta = np.zeros([nx,nA])
    
    # Keep stats for final print of graph
    episode_rewards = []
    entropies = []
    
    # Main loop 
    for episode in range(seed_params['n_episodes']):
    	# Keep track of game score to print
        for i in range(seed_params['batch_size']):
            states = env.reset()
            for state in states:
                state['state'] = normalise_state(state['state'],env_params)
                
            grads = [np.zeros([env_params['time_limit'] + 1, nx, nA])] * (seed_params['n_cyclists'])
            rewards = np.zeros_like(rolling_average_reward)
            score = 0
            step = 0
            entropy = 0
            while True:
                for i in range(len(states)):
                    cyclist_number = states[i]['number']
            		# Sample from policy and take action in environment
                    probs = np.array(policy(states[i]['state'],model))
                    entropy += get_entropy(probs)
                
                    # With probability epsilon, draw from a uniform distribution
                    if np.random.rand() < epsilon:
                        action = np.random.choice(nA, p = [1 / nA] * nA)
                    # Else draw from the probability distribution
                    else:
                        try:
                            action = np.random.choice(nA,p = probs)
                        except:
                            break
                        
                    # Compute gradient and save with reward in memory for our weight updates
                    grad = softmax_grad(probs, action, states[i]['state'], model, seed_params['lam'])
                    grads[cyclist_number][step,:,:] = grad
                    actions[i] = action - 1
                    

                states,reward,done = env.step(actions)

                for state in states:
                    state['state'] = normalise_state(state['state'],env_params)                

                rewards[step] = reward	
                score+=reward
                step += 1
                if done:
                    break
            
            # Save the best parameters
            if score > high_score:
                high_score = score
                model_best = model
            
            rewards = rewards[:step]
            grads = [grad[:step,:,:] for grad in grads]
            state_reward = np.zeros_like(rewards)
            # Update parameters
            for i in range(len(rewards)):
        		# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
                state_reward[i] = sum([ r * (seed_params['gamma'] ** t) for t,r in enumerate(rewards[i:])])
            
            rolling_average_reward[:len(state_reward)] = rolling_average_reward[:len(state_reward)] * (1 - seed_params['alpha_mean']) + seed_params['alpha_mean'] * state_reward
            advantage = state_reward - rolling_average_reward[:len(state_reward)]
            rolling_std_dev[:len(state_reward)] = np.sqrt(rolling_std_dev[:len(state_reward)] ** 2 * (1 - seed_params['alpha_std']) + seed_params['alpha_std'] * (advantage) ** 2)
            advantage /= (rolling_std_dev[:len(state_reward)] + 0.1)
            # Subtract mean (baselining) and then divide by rolling std
            for i in range(len(grads)):
                model_delta += np.sum((seed_params['learning_rate'] * grads[i] * advantage[:grads[i].shape[0],None,None] / seed_params['gamma']),axis = 0)
                
        model += model_delta
        average_entropy = entropy / (4 * (step + 1))
        entropies.append(average_entropy)

        if average_entropy < entropy_kick and seed_params['repeat_explore']:
            epsilon = max(epsilon,hyper_params['epsilon_0'] / (episode/ 250))
            entropy_kick /= 2
            if entropy_kick < 0.1:
                entropy_kick = 0
        epsilon *= seed_params['epsilon_decay']
        exploration.append(epsilon)
    	# Append for logging and print
        episode_rewards.append(score) 
        print(f'Seed: {seed}, EP: {episode}, Score: {score}',flush = True)
        print()
#        sys.stdout.flush()

    Seed_entropies = np.array(entropies)
    Seed_rewards = np.array(episode_rewards)
    return Seed_rewards, Seed_entropies, exploration, model_best


if __name__ == '__main__':
    # Initialisations

    Seed_rewards = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    Seed_entropies = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    Seed_explore = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    model_best = [None] * sim_params['n_seeds']
    
    p = Pool(processes = sim_params['n_seeds'])
    
    n_iterations = len(hyper_params['lam'])
    seed_params = []
    labels = []

    for iterator in range(n_iterations):
        seed_params.append([])
#        decay_episodes = hyper_params['decay_episodes'][iterator]
        lam = hyper_params['lam'][iterator]
        labels.append(f'Lambda is {lam}')
        for seed in range(sim_params['n_seeds']):
            seed_params[iterator].append({'Env' : env_params})
            seed_params[iterator][seed]['seed'] = seed
            seed_params[iterator][seed].update(sim_params)
            seed_params[iterator][seed].update(hyper_params)
            seed_params[iterator][seed]['epsilon_decay'] = (hyper_params['epsilon_0'] / hyper_params['epsilon_final']) ** (-1 / hyper_params['decay_episodes'])
            seed_params[iterator][seed]['lam'] = lam
    # Flatten list to input it into multiprocessing, then restack it
    flat_params = [item for sublist in seed_params for item in sublist]
    output = p.map(run_seed, flat_params)
    output_stacked = []
    for iterator in range(n_iterations):
        output_stacked.append([])
        for seed in range(sim_params['n_seeds']):
            output_stacked[iterator].append(output[sim_params['n_seeds'] * iterator + seed])
            
        
#    Seed_rewards[seed,:], Seed_entropies[seed,:], Seed_explore[seed,:], model_best[seed] = run_seed(seed, seed_params)
    
    params = {'Env': env_params, 'Hyp' : hyper_params, 'Sim' : sim_params}
    pickle.dump([output_stacked,labels,params], open('Optimal.p','wb'))
