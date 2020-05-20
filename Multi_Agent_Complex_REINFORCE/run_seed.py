# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:02:51 2020

@author: Ollie
"""
import numpy as np
from environment import Environment
from learning_functions import get_entropy, softmax_grad, State_normaliser, policy

def run_seed(seed_params):
    # Create gym and seed numpy
    env_params = seed_params['Env']
    sim_params = seed_params['Sim']
    hyper_params = seed_params['Hyp']
    
    state_normaliser = State_normaliser(env_params)
    seed = seed_params['seed']
    # actions = [0] * sim_params['n_cyclists']
#    entropy_kick = 0.4
    env = Environment(env_params, n_cyclists = sim_params['n_cyclists'])
    action_space = env.action_space
    nA = len(action_space)
    nx = env.state_space + 1
    np.random.seed(seed)
    max_entropy = get_entropy(np.array([1/ nA] * nA))
    # Note this should be increased for longer runs
    reward_std = 0
    explore_std = 0
    
    
    # Init weights
    model = []
    model_delta = []
    high_score = 0
    for i in range(sim_params['n_cyclists']):
        model.append({
            'weights' : np.random.normal(0,0.1,(nx,nA)),
            'rolling_average_reward' : np.zeros(env_params['time_limit'] + 1),
            'rolling_average_explore' : np.zeros(env_params['time_limit'] + 1),
            'reward_std' : 0,
            'explore_std' : 0
            })
        model_delta.append(np.zeros([nx,nA]))
    
    # Keep stats for final print of graph
    episode_rewards = []
    episode_entropies = []
    
    # Main loop 
    for episode in range(sim_params['n_episodes']):
    	# Keep track of game score to print
        for i in range(hyper_params['batch_size']):
            states = env.reset()
            for state in states:
                state['state'] = state_normaliser.normalise_state(state['state'])
                
            step_grads = [[] for i in range(sim_params['n_cyclists'])]
            step_rewards = [[] for i in range(sim_params['n_cyclists'])]
            step_entropies = [[] for i in range(sim_params['n_cyclists'])]
            
            scores = [0] * sim_params['n_cyclists']
            step = 0
            while True:
                actions = []
                for i in range(len(states)):
                    cyclist_number = states[i]['number']
            		# Sample from policy and take action in environment
                    probs = np.array(policy(states[i]['state'],model[cyclist_number]['weights']))
                    step_entropies[cyclist_number].append(get_entropy(probs))
                
                    # With probability epsilon, draw from a uniform distribution
                    try:
                        action = np.random.choice(nA,p = probs)
                    except:
                        break
                        
                    # Compute gradient and save with reward in memory for our weight updates
                    grad = softmax_grad(probs, action, states[i]['state'], model)
                    step_grads[cyclist_number].append(grad)
                    actions.append(action - 1)
                     
                    
                
                states,rewards,done = env.step(actions)

                for state, reward in zip(states, rewards):
                    state['state'] = state_normaliser.normalise_state(state['state'])              
                    scores[state['number']] += reward
                    step_rewards[state['number']].append(reward)

                step += 1
                if done:
                    break
            
            # Save the best parameters
            if sum(scores) >= high_score:
                high_score = sum(scores)
                model_best = model
                
            for i in range(sim_params['n_cyclists']):
                rewards = step_rewards[i]
                entropies = step_entropies[i][:len(rewards)]
                grads = np.array(step_grads[i])
                state_reward = np.zeros_like(rewards)
                state_explore = np.zeros_like(entropies)
                state_max = np.zeros_like(entropies)
                max_entropies = np.ones_like(entropies) * max_entropy
                # Update parameters
                for j in range(len(rewards)):
            		# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
                    state_reward[i] = sum([ r * (hyper_params['gamma'] ** t) for t,r in enumerate(rewards[i:])])
                    state_explore[i] = sum([ r * (hyper_params['gamma_explore'] ** t) for t,r in enumerate(entropies[i:])])
                    state_max[i] = sum([ r * (hyper_params['gamma_explore'] ** t) for t,r in enumerate(max_entropies[i:])])
    
                model[i]['rolling_average_reward'][:len(state_reward)] = model[i]['rolling_average_reward'][:len(state_reward)] * (1 - hyper_params['alpha_mean']) + hyper_params['alpha_mean'] * state_reward
                model[i]['rolling_average_explore'][:len(state_explore)] = model[i]['rolling_average_explore'][:len(state_explore)] * (1 - hyper_params['alpha_mean']) + hyper_params['alpha_mean'] * state_explore
                reward_advantage = state_reward - model[i]['rolling_average_reward'][:len(state_reward)]
                # explore_advantage = state_explore - model[i]['rolling_average_explore'][:len(state_explore)]
                explore_advantage = state_explore - state_max
                
                model[i]['reward_std'] = np.sqrt(model[i]['reward_std'] ** 2 * (1 - hyper_params['alpha_std']) + hyper_params['alpha_std'] * (np.mean(reward_advantage ** 2)))
                model[i]['explore_std'] = np.sqrt(model[i]['explore_std'] ** 2 * (1 - hyper_params['alpha_std']) + hyper_params['alpha_std'] * (np.mean(explore_advantage ** 2)))
    
                
                reward_advantage /= (reward_std + 0.01)
    
                advantage = reward_advantage * hyper_params['lambda_reward'] + explore_advantage * (1 - hyper_params['lambda_reward'])
                # Subtract mean (baselining) and then divide by rolling std

                for j in range(len(grads)):
                    model_delta[i] += np.sum((hyper_params['learning_rate'] * grads[j] * advantage[:grads[j].shape[0],None,None]),axis = 0)

        for mod, delta in zip(model, model_delta):
            mod['weights'] += delta

        episode_entropies.append([np.mean(entropies) for entropies in step_entropies])
    	# Append for logging and print
        episode_rewards.append(scores) 
        if sim_params['print_rewards']:
            print(f'Seed: {seed}, EP: {episode}, Score: {scores}',flush = True)
            print()
#        sys.stdout.flush()
    
    Seed_entropies = np.array(episode_entropies)
    Seed_rewards = np.array(episode_rewards)
    return Seed_rewards, Seed_entropies, model_best

