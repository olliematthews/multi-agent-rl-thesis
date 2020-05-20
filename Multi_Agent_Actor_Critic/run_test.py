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
    'n_episodes' : 10,
    'n_seeds' : 1,
    'n_cyclists' : 2,
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
    'lr_actor' : [0.001],
    'p_order_actor' : 1,
    'lr_critic' : 0.01,
    'layers_critic' : [10,10],
    'TD_batch' : 60,
    'gamma' : 0.99,
    'gamma_explore' : 0,
    'use_std' : True,
    'lambda_reward_profile' : [[0,1.0],[1.0,1.0]],
    'iterate_over' : 'lr_actor',
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
    Seed_rewards, Seed_entropies, seed_params, Seed_velocities, Seed_distances, final_model = run_seed(flat_params[0])
    pickle.dump([final_model, seed_params],open('bm.p','wb'))
    # pickle.dump([Seed_velocities, Seed_distances], open('vd___.p','wb'))
    # pickle.dump(Seed_rewards, open('rewards___.p', 'wb'))