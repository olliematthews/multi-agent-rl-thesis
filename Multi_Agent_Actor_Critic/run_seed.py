# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:02:51 2020

@author: Ollie
"""
import numpy as np
from environment import Environment
import pickle
import random

def run_seed(seed_params):
    '''
    Runs a learning simulation for the seed in question.

    Parameters
    ----------
    seed_params : dict

    Returns
    -------
    Seed_rewards : np.array
        Reward history for the seed.
    Seed_entropies : np.array
        Entropy history for the seed.
    Loss_history : list
        History of the losses of the critic's value function.

    '''
    from networks import Actor, Critic, State_normaliser

    # Create gym and seed numpy
    env_params = seed_params['Env']
    sim_params = seed_params['Sim']
    hyper_params = seed_params['Hyp'].copy()
    
    # Change relative points to episode points for explore profile
    for point in hyper_params['lambda_reward_profile']:
        point[0] = int(point[0] * sim_params['n_episodes'])
        
    
    seed = seed_params['seed']
    # actions = [0] * sim_params['n_cyclists']
#    entropy_kick = 0.4
    env = Environment(env_params, n_cyclists = sim_params['n_cyclists'], poses = [0,0,0,0])
    # action_space = env.action_space
    np.random.seed(seed)
    random.seed(seed)

    # nA = len(env.action_space)
    # Init weights
    if sim_params['weights'] == 'none':
        model = {'normaliser' : State_normaliser(env, hyper_params)}

        acs = []
        for i in range(sim_params['n_cyclists']):
            acs.append({
                'actor' : Actor(env, hyper_params, np.random.randint(1000)),
                'critic' : Critic(env, hyper_params, np.random.randint(1000))
                # 'actor' : Actor(env, hyper_params, seed),
                # 'critic' : Critic(env, hyper_params, seed)
                })
        model.update({'acs' : acs})
    else:
        model = sim_params['weights']
    
    # Keep stats for final print of graph
    episode_rewards = []
    episode_entropies = []
    episode_velocities = []
    episode_distances = []
    # Main loop 
    for episode in range(sim_params['n_episodes']):
    	# Keep track of game score to print
        pose_inits = [np.round(np.random.randn() * sim_params['random_std'], 1) for i in range(sim_params['n_cyclists'])] if sim_params['random_init'] else [0] * sim_params['n_cyclists']
        states = env.reset(poses = pose_inits)
        states = [model['normaliser'].normalise_state(state) for state in states]
        step_entropies = [[] for i in range(sim_params['n_cyclists'])]
        scores = [0] * sim_params['n_cyclists']
        group_score = 0
        group_reward = 0
        step = 0
        while True:
            actions = []
            for state in states:

                cyclist_number = state['number']

                probs, action, entropy = model['acs'][cyclist_number]['actor'].choose_action(state['state'])
                step_entropies[cyclist_number].append(entropy)
                actions.append(action)
                
                model['acs'][cyclist_number]['critic'].store_transition_1(state['state'], action, entropy, probs)
        
            next_states, rewards, done, info = env.step(actions)
            
            group_reward += sum(rewards) / sim_params['n_cyclists']
            group_score = sum(rewards) / sim_params['n_cyclists']
            next_states = [model['normaliser'].normalise_state(state) for state in next_states]             
            states = next_states.copy()

            for reward, score, state in zip(rewards, rewards, states):
                cyclist_number = state['number']
                model['acs'][cyclist_number]['critic'].store_transition_2((1 - sim_params['lambda_group']) * reward, state['state'], model['acs'][cyclist_number]['actor'], episode)
                scores[cyclist_number] += (1 - sim_params['lambda_group']) * score + group_score * sim_params['lambda_group']

            if done:
                dividors, subtractors = model['normaliser'].update_normalisation()
                [mod['critic'].update_dividor_subtractor(dividors, subtractors, mod['actor']) for mod in model['acs']]
                # Store the dead states
                for state in states:
                    model['acs'][state['number']]['critic'].store_dead(state['state'], model['acs'][state['number']]['actor'], episode)
                    actions.append(0)

                [m['critic'].store_final_reward(group_reward) for m in model['acs']]
                episode_velocities.append(info[0])
                episode_distances.append(info[1])
                [score + sim_params['lambda_group'] * group_score for score in scores]
                break
            else:
                step += 1  
                  
                
        episode_entropies.append([np.mean(entrops) for entrops in step_entropies])
    	# Append for logging and print
        episode_rewards.append(scores) 
        if sim_params['print_rewards']:
            print(f'Seed: {seed}, EP: {episode}, Score: {np.round(scores)}',flush = True)
            print()
    
    Seed_entropies = np.array(episode_entropies)
    Seed_rewards = np.array(episode_rewards)
    Seed_velocities = np.array(episode_velocities)
    Seed_distances = np.array(episode_distances)
    output = [Seed_rewards, Seed_entropies, seed_params, Seed_velocities, Seed_distances]
    print('Dumping output')
    pickle.dump(output, open(f'output_{seed}.p','wb'))
    return output

