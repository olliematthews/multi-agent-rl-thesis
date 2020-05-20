# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:28:33 2019

@author: Ollie
"""

import numpy as np
from multiprocessing import Pool
import pickle
from run_seed import run_seed

# Simulation Parameters
sim_params = {
    'n_episodes' : 4000,
    'n_seeds' : 10,
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
    'learning_rate' : 1e-7,
    'batch_size' : 2,
    'gamma' : 0.99,
    'gamma_explore' : 0,
    'use_std' : True,
    'alpha_mean' : 0.1,
    'alpha_std' : 0.05,
    'lambda_reward' : [1]
    }



if __name__ == '__main__':
    # Initialisations

    Seed_rewards = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    Seed_entropies = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    Seed_explore = np.empty([sim_params['n_seeds'],sim_params['n_episodes']])
    model_best = [None] * sim_params['n_seeds']
    
    
    n_iterations = len(hyper_params['lambda_reward'])
    
    p = Pool(processes = sim_params['n_seeds'] * n_iterations)

    seed_params = []
    labels = []

    for iterator in range(n_iterations):
        seed_params.append([])
#        decay_episodes = hyper_params['decay_episodes'][iterator]
        labels.append(hyper_params['lambda_reward'][iterator])
        for seed in range(sim_params['n_seeds']):
            seed_params[iterator].append({
                'Env' : env_params.copy(),
                'Sim' : sim_params.copy(),
                'Hyp' : hyper_params.copy(),
                'seed' : seed + 1
                })
            
            seed_params[iterator][seed]['Hyp']['lambda_reward'] = hyper_params['lambda_reward'][iterator]
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
    pickle.dump([output_stacked,labels,params], open('PG_Multi.p','wb'))
