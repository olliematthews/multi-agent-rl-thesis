# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:28:33 2019

@author: Ollie
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
    'off_batch_size' : 1e99
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
        pickle.dump([output, params],open('no_off_4.p','wb'))

    # pickle.dump([Seed_velocities, Seed_distances], open('vd___.p','wb'))
    # pickle.dump(Seed_rewards, open('rewards___.p', 'wb'))