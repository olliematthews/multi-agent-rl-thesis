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
# Simulation Parameters
sim_params = {
    'n_episodes' : 100,
    'n_seeds' : 1,
    'n_cyclists' : 4,
    'print_rewards' : True
    }

env_params = {
    'cD' : {
            'a': 4,
            'b': 1.6,
            'cDrag': 0.2
            },
    'vel_max' : 4,
    'vel_min' : 0,
    'race_length' : 400,
    'time_limit' : 200
    }


# Hyperparameters
hyper_params = {
    'lr_actor' : [0.005],
    'p_order_actor' : 1,
    'lr_critic' : 0.01,
    'p_order_critic' : 2,
    'TD_batch' : 60,
    'gamma' : 0.99,
    'gamma_explore' : 0,
    'use_std' : True,
    'lambda_reward' : 0.6,
    'iterate_over' : 'lr_actor',
    'std_init' : 10,
    'alpha_std' : 0.05
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
#        decay_episodes = hyper_params['decay_episodes'][iterator]
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
    Seed_rewards, Seed_entropies, weights = run_seed(flat_params[0])
    # pickle.dump([Seed_rewards, Seed_entropies, model_best], open('ExploreYes.p','wb'))
#     output_stacked = []
#     for iterator in range(n_iterations):
#         output_stacked.append([])
#         for seed in range(sim_params['n_seeds']):
#             output_stacked[iterator].append(output[sim_params['n_seeds'] * iterator + seed])
            
        
# #    Seed_rewards[seed,:], Seed_entropies[seed,:], Seed_explore[seed,:], model_best[seed] = run_seed(seed, seed_params)
    
#     params = {'Env': env_params, 'Hyp' : hyper_params, 'Sim' : sim_params}
#     pickle.dump([output_stacked,labels,params], open('Exploring.p','wb'))
