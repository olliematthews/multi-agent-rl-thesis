# -*- coding: utf-8 -*-
"""
The equivalent of run. Will run episodes, but not in parallel (since the coach
framework itself uses multiprocessing).
"""

import numpy as np
from multiprocessing import Pool
import pickle
from run_seed import run_seed
import time
import cProfile

# model, _ = pickle.load(open('bm.p','rb'))

# Simulation Parameters
sim_params = {
    'n_episodes' : 5000,
    'n_seeds' : 1,
    'n_cyclists' : 4,
    'lambda_group' : 0.0, # Reward = lambda * group_reward + (1-lambda) * reward
    'print_rewards' : True,
    'random_init' : True, # If true, cyclists are initialised according to gaussian
    'random_std' : 0.5,
    'weights' : 'none' 
    }

env_params = {
    'cD' : {
            'a': 4,
            'b': 1.6,
            'cDrag': 0.2,
            'offset' : 0.61
            }, # Describes the cD equation
    'vel_max' : 4,
    'vel_min' : 0,
    'race_length' : 400,
    'time_limit' : 200,
    'granularity' : 0.1 # Granularity in velocity and poses
    }


# Hyperparameters
'''
Notable hyperparameters:
    gamma - discount factor for rewards
    gamma_explore - discount factor for entropy, usually at 0
    param_changes - base change to implement for that hyperparameter
    window_size - the size number of episodes run in a simulation to test
        hyperparameter values
    max_goes - the max number of times you can try a parameter before moving 
        onto another. max_goes = 0 => you only test each parameter once
    off_batch_size - the batch size for off-policy gradient updates
    significant prob - reduces computational complexity at the expense of 
        accuracy in the value function. Only actions who's probability >
        significant_prob will be fed through the value function when taking an 
        expectation over actions.
'''
hyper_params = {
    'lr_actor' : 0.0001,
    'lr_critic' : [0.01],
    'layers_critic' : [20,20],
    'TD_batch' : 60,
    'gamma' : 0.99,
    'gamma_explore' : 0,
    'use_std' : True,
    'iterate_over' : 'lr_critic',
    'std_init' : 10,
    'alpha_std' : 0.05,
    'state_normaliser_alpha' : 0.05,
    'param_changes' : {'lambda_reward' : 0.05,
                       'distance_penalty' : 0.2},
    'window_size' : 200,
    'param_inits' : {'lambda_reward' : 0.5,
                       'distance_penalty' : 0.0},
    'max_goes' : 1,
    'off_batch_size' : 300,
    'significant_prob' : 0.01 
    }



if __name__ == '__main__':
    # Initialisations

    Seed_rewards = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    Seed_entropies = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    Seed_explore = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    model_best = [None] * sim_params['n_seeds']
    
    
    n_iterations = len(hyper_params[hyper_params['iterate_over']])
    
    seed_params = []
    labels = []

    for iterator in range(n_iterations):
        seed_params.append([])
        labels.append(hyper_params['iterate_over'] + ' = ' + str(hyper_params[hyper_params['iterate_over']][iterator]))
        for seed in range(sim_params['n_seeds']):
            seed_params[iterator].append({
                'Env' : env_params.copy(),
                'Sim' : sim_params.copy(),
                'Hyp' : hyper_params.copy(),
                'seed' : seed + 4
                })
            
            seed_params[iterator][seed]['Hyp'][hyper_params['iterate_over']] = hyper_params[hyper_params['iterate_over']][iterator]
    # Flatten list to input it into multiprocessing, then restack it
    flat_params = [item for sublist in seed_params for item in sublist]
    # cProfile.run('re.compile("run_seed(flat_params[0])")','restats')
    output = []
    params = {'sim' : sim_params, 'hyp': hyper_params, 'env' : env_params}
    for i in range(sim_params['n_seeds']):
        output.append(run_seed(flat_params[i]))
        # Save after each episode
        pickle.dump([output, params],open('no_off_4.p','wb'))

