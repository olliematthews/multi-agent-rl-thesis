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
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'sans-serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'figure.autolayout': True})


def plot_rewards(average, std, label, sigma, color):
    plt.plot(gaussian_filter1d(average, sigma), color = color, label = label)
    plt.fill_between(x = range(average.size), y1 = gaussian_filter1d(average + std,sigma), y2 = np.clip(gaussian_filter1d(average - std, sigma),0,None), color = color, alpha = 0.3)


# files = ['1_Poly.p','no_std.p','no_std_3.p']
# labels = ['Linear','Quadratic','Cubic']
# figure = plt.figure(figsize = (4,2.5))
# ax = plt.gca()
# for file, label in zip(files, labels):
#     w_best, seed_rewards, seed_losses , w_critics, params, policies = pickle.load(open(file,'rb'))
    
#     plt.plot
#     seed_rewards = np.array(seed_rewards)
#     averages = np.mean(seed_rewards, axis = 1)
#     stds = np.std(seed_rewards, axis = 1)
#     color = next(ax._get_lines.prop_cycler)['color']
#     for i in range(averages.shape[0]):
#         plt.plot(gaussian_filter(averages[i,:], sigma), label = label, color = color)
#         plt.fill_between(range(averages.shape[1]), gaussian_filter(averages[i,:] + stds[i,:], sigma), gaussian_filter(averages[i,:] - stds[i,:], sigma),alpha = 0.3, color = color)
# plt.tight_layout()
# plt.xlabel('Number of Episodes')
# plt.ylabel('Reward')
# plt.legend(loc = 'lower right')
# plt.savefig('polynomial_order.png', bbox_inches = "tight")

w_best, seed_rewards, seed_losses , w_critics, params, policies = pickle.load(open('batches.p','rb'))
sigma= 10

figure = plt.figure(figsize = (6.5,3.2))
ax = plt.gca()

labels = [1,5,10,20,50,100,500]
for seed_reward, label in zip(seed_rewards, labels):
    if label == 10 or label == 50:
        continue
    seed_reward = np.array(seed_reward)
    averages = np.mean(seed_reward, axis = 0)[:2000]
    stds = np.std(seed_reward, axis = 0)[:2000]
    color = next(ax._get_lines.prop_cycler)['color']
    plot_rewards(averages, stds, label, sigma, color)
    
plt.tight_layout()
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.legend()
plt.savefig('ac_batches.pgf', bbox_inches = "tight")


# w_best, seed_rewards, seed_losses , w_critics, params, policies = pickle.load(open('Estimate_Optimal.p','rb'))

# seed_rewards = np.squeeze(seed_rewards)
# averages__ = np.mean(seed_rewards, axis = 0)[:1960]
# stds__ = np.std(seed_rewards, axis = 0)[:1960]


# labels = [0.1,0.05,0.01,0.005]

# for i in range(len(labels)):
#     plt.plot(gaussian_filter(averages[i,:], sigma = sigma), label = labels[i])
# plt.legend()
# plt.show()
# figure = plt.figure()
# ax = plt.gca()
# color = next(ax._get_lines.prop_cycler)['color']
# plt.plot(gaussian_filter(averages[1,:], sigma = sigma), label = labels[0], color = color)
# plt.fill_between(range(averages.shape[1]), gaussian_filter(averages[0,:] - stds[0,:], sigma = sigma), gaussian_filter(averages[0,:] + stds[0,:], sigma = sigma), color = color, alpha = 0.2)


# color = next(ax._get_lines.prop_cycler)['color']

# plt.plot(40 + np.array(range(averages__.size)),gaussian_filter(averages__, sigma = sigma), color = color, label = 'AC_Estimates')
# plt.fill_between(40 + np.array(range(averages__.size)), gaussian_filter(averages__ - stds__, sigma = sigma), gaussian_filter(averages__ + stds__, sigma = sigma), color = color, alpha = 0.2)



# plt.legend()
