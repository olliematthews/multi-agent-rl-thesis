"""
Same as 'run', but just runs a single episode. Used for debugging.
"""

import numpy as np
from multiprocessing import Pool
import pickle
from run_seed import run_seed
import time
import cProfile


# Simulation Parameters
sim_params = {
    'n_episodes' : 10000,
    'n_seeds' : 1,
    'n_cyclists' : 4,
    'lambda_group' : 0.0,
    'print_rewards' : True,
    'random_init' : True,
    'random_std' : 0.5,
    'weights' : 'none'
    }

env_params = {
    'cD' : {
            'a': 4,
            'b': 1.6,
            'cDrag': 0.2,
            'offset' : 0.61
            },
    'vel_max' : 4,
    'vel_min' : 0,
    'race_length' : 400,
    'time_limit' : 200,
    'granularity' : 0.1,
    'distance_penalty' : 0
    }


# Hyperparameters
hyper_params = {
    'lr_actor' : 0.0001,
    'lr_critic' : 0.01,
    'layers_critic' : [10,10],
    'TD_batch' : 60,
    'gamma' : 0.99,
    'gamma_explore' : 0,
    'use_std' : True,
    'lambda_reward_profile' : [[[0.0,1.0],[1.0,1.0]]],
    'iterate_over' : 'lambda_reward_profile',
    'std_init' : 10,
    'alpha_std' : 0.05,
    'state_normaliser_alpha' : 0.01
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
                'seed' : seed
                })
            
            seed_params[iterator][seed]['Hyp'][hyper_params['iterate_over']] = hyper_params[hyper_params['iterate_over']][iterator]
    # Flatten list to input it into multiprocessing, then restack it
    flat_params = [item for sublist in seed_params for item in sublist]
    # cProfile.run('re.compile("run_seed(flat_params[0])")','restats')
    Seed_rewards, Seed_entropies, seed_params, Seed_velocities, Seed_distances = run_seed(flat_params[0])
