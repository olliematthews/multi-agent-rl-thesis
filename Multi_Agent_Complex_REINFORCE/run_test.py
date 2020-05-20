# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:28:33 2019

@author: Ollie
"""

import numpy as np
import pickle
from run_seed_homogenous import run_seed
import time
# import cProfile
# import re
# Simulation Parameters
sim_params = {
    'n_episodes' : 100,
    'n_seeds' : 1,
    'n_cyclists' : 4,
    'print_rewards' : True
    }

env_params = {
    'cD' : {
            'a': 2.2,
            'b': 1,
            'cDrag': 0.2
            },
    'vel_max' : 4,
    'vel_min' : 0,
    'race_length' : 400,
    'time_limit' : 200
    }


# Hyperparameters
hyper_params = {
    'learning_rate' : 5e-5,
    'batch_size' : 2,
    'gamma' : 0.99,
    'gamma_explore' : 0,
    # Epsilon starts at 0.98. It decays to epsilon_final in decay_episodes episodes
    'use_std' : True,
    'alpha_mean' : 0.1,
    'alpha_std' : 0.1,
    'lambda_reward' : [0],
    'com_frequency' : 50
    }



if __name__ == '__main__':
    # Initialisations

    Seed_rewards = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    Seed_entropies = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    Seed_explore = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    model_best = [None] * sim_params['n_seeds']
    
    
    n_iterations = len(hyper_params['lambda_reward'])
    
    seed_params = []
    labels = []

    for iterator in range(n_iterations):
        seed_params.append([])
#        decay_episodes = hyper_params['decay_episodes'][iterator]
        for seed in range(sim_params['n_seeds']):
            seed_params[iterator].append({
                'Env' : env_params.copy(),
                'Sim' : sim_params.copy(),
                'Hyp' : hyper_params.copy(),
                'seed' : seed
                })
            
            seed_params[iterator][seed]['Hyp']['lambda_reward'] = hyper_params['lambda_reward'][iterator]
    # Flatten list to input it into multiprocessing, then restack it
    flat_params = [item for sublist in seed_params for item in sublist]
    # cProfile.run('re.compile("run_seed(flat_params[0])")','restats')
    time1 = time.time()
    Seed_rewards, Seed_entropies = run_seed(flat_params[0])
    time2 = time.time()
    print(time2 - time1)
    # pickle.dump([Seed_rewards, Seed_entropies, model_best], open('ExploreYes.p','wb'))
#     output_stacked = []
#     for iterator in range(n_iterations):
#         output_stacked.append([])
#         for seed in range(sim_params['n_seeds']):
#             output_stacked[iterator].append(output[sim_params['n_seeds'] * iterator + seed])
            
        
# #    Seed_rewards[seed,:], Seed_entropies[seed,:], Seed_explore[seed,:], model_best[seed] = run_seed(seed, seed_params)
    
#     params = {'Env': env_params, 'Hyp' : hyper_params, 'Sim' : sim_params}
#     pickle.dump([output_stacked,labels,params], open('Exploring.p','wb'))
