# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 17:14:28 2019

@author: Ollie
"""
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d
import numpy as np
from matplotlib import gridspec
# import matplotlib

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'sans-serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })


plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'figure.autolayout': True})

w_best, Seed_rewards, Seed_entropies, Seed_mean_velocities, params = pickle.load(open('Baselining_2.p','rb'))

sigma = 10

Seed_rewards = np.squeeze(Seed_rewards)
Seed_entropies = np.squeeze(Seed_entropies)
Seed_mean_velocities = np.squeeze(Seed_mean_velocities)

Average_rewards = gaussian_filter1d(np.mean(Seed_rewards,axis = 1), sigma = sigma, axis = 1)
Average_entropies = gaussian_filter1d(np.mean(Seed_entropies, axis = 1), sigma = sigma, axis = 1)
Average_mean_velocities = gaussian_filter1d(np.mean(Seed_mean_velocities, axis = 1), sigma = sigma, axis = 1)

Std_rewards = gaussian_filter1d(np.std(Seed_rewards, axis = 1), sigma = sigma, axis = 1)
Std_entropies = gaussian_filter1d(np.std(Seed_entropies, axis = 1), sigma = sigma, axis = 1)
Std_mean_velocities = gaussian_filter1d(np.std(Seed_mean_velocities, axis = 1), sigma = sigma, axis = 1)


figure = plt.figure(figsize = (6.5,3.2))
ax = plt.gca()
labels = ['Baselining and Normalisation', 'Baselining','No Baselining']
for i in range(Seed_rewards.shape[0]-1,-1,-1):
    color = next(ax._get_lines.prop_cycler)['color'] 

    plt.plot(Average_rewards[i,:], label = labels[i], color = color)
    plt.fill_between(range(Std_rewards.shape[1]), Average_rewards[i,:] + Std_rewards[i,:], Average_rewards[i,:] - Std_rewards[i,:], alpha = 0.3, color = color)
plt.legend()
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.savefig('Baselining.pgf', bbox = "tight")
plt.show()
