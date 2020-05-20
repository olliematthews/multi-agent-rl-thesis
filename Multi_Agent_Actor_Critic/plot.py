# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 17:14:28 2019

@author: Ollie
"""
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter, minimum_filter1d, maximum_filter1d
import numpy as np
from matplotlib import gridspec


def parse_data(results):
    rewards = []
    entropies = []
    params = []
    velocities = []
    distances = []
    models = []

    for i in range(len(results)):
        rewards.append(results[i][0])
        # r = []
        # for j in range(results[i][0][4:].size):
        #     r.append(results[i][0][4 + j])
        # rewards.append(np.array(r))
        entropies.append(results[i][1])
        params.append(results[i][2])
        velocities.append(results[i][3])
        distances.append(results[i][4])
        # models.append(results[i][5])        
        
        # model_best.append(results[i][2])
    rewards = np.array(rewards)
        
    return rewards, entropies, params, velocities, distances

def plot_episode(rewards, sigma):
    for i in range(rewards.shape[0]):
        for j in range(rewards[i].shape[1]):
            plt.plot(gaussian_filter(rewards[i,:,j], sigma = sigma))
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title(f'Seed {i}')
        plt.show()

def plot_episodes(rewards, sigma,save = False, filename = '', title = ''):
    for i in range(rewards.shape[0]):
        plt.plot(gaussian_filter(np.mean(rewards,axis = 2)[i,:], sigma = sigma), label = f'Seed {i}')
        
#    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    if save:
        plt.savefig(filename)
    plt.savefig('Episodes')
    plt.title(title)
    plt.show()

def plot_iterations(rewards, labels, sigma, save = False, filename = ''):
    plt.figure(figsize = figure_size)
    ax = plt.gca()
    for i in range(len(rewards)):
        color = next(ax._get_lines.prop_cycler)['color']
        rewards_mean = np.mean(rewards[i],axis = (0,2))
        rewards_std = np.std(rewards[i],axis = (0,2))
        rewards_mean = gaussian_filter(rewards_mean,sigma = sigma)
        rewards_std = gaussian_filter(rewards_std,sigma = sigma)
        plt.plot(rewards_mean, color = color, label = labels[i])
        plt.fill_between(range(rewards_mean.size),rewards_mean + rewards_std, rewards_mean - rewards_std, alpha = 0.3)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    if save:
        plt.savefig(filename)
    plt.show()
    
def plot_max_min(rewards, labels, sigma, save = False, filename = ''):
    plt.figure(figsize = figure_size)
    ax = plt.gca()
    for i in range(len(rewards)):
        rewards[i] = rewards[i].squeeze()
        color = next(ax._get_lines.prop_cycler)['color']
        rewards_mean = np.mean(rewards[i],axis = (0,2))
        rewards_mean = gaussian_filter(rewards_mean,sigma = sigma)
        rewards_min = np.min(rewards[i], axis = (0,2))
        rewards_min = minimum_filter1d(rewards_min, sigma)
        
        rewards_max = np.max(rewards[i], axis = (0,2))
        rewards_max = maximum_filter1d(rewards_max, sigma)
        plt.plot(rewards_mean, color = color, label = labels[i])
        plt.fill_between(range(rewards_mean.size),rewards_min, rewards_max, alpha = 0.3)
        # plt.plot(gaussian_filter(rewards_min, 100), color = color, linestyle = 'dashed')
        # plt.plot(gaussian_filter(rewards_max, 100), color = color, linestyle = 'dashed')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    if save:
        plt.savefig(filename)
    plt.show()

    
    
def plot_entropy_rewards(reward,entropy,sigma, save = False, filename = ''):
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
    # the fisrt subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    line0, = ax0.plot(range(reward.shape[0]), reward)
    plt.ylabel('Average Reward')
    
    #the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1.plot(range(entropy.shape[0]), entropy)
    plt.ylabel('Average Entropy')
    plt.setp(ax0.get_xticklabels(), visible=False)
    #yticks[-1].label1.set_visible(False)
    plt.xlabel('Number of Episodes')
    
    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    if save:
        plt.savefig(filename)
    plt.show()
    

sigma = 10

plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'figure.autolayout': True})
figure_size = (4,2.8)




#_,_, _, Seed_rewards, _, _, _ =     pickle.load(open('BigTest.p','rb'))
# files = ['No_Normaliser_No_Coach.p', 'No_Normaliser.p', 'No_Normaliser_Std.p',  'No_Normaliser_No_Coach_Std.p']
# labels = ['No Coach', 'Coach', 'Coach STD', 'No Coach STD']


rewards = []
entropies = []
exploration = []
model_best = []
data, labels, params = pickle.load(open('acfull.p','rb'))
# labels = ['No Exploration', 'Exploration']
# labels = ['Exploration = ' + str(label) for label in labels]
for i in range(len(data)):
    r, e, l, v, d = parse_data(data[i])
    # To deal with error in saving
    # r = r[:,4:]
    # e = [a[4:] for a in e]
    rewards.append(r)
    entropies.append(e)
    # model_best.append(m)
    # plot_episode(r, sigma)
    plot_episodes(r,sigma)

# for i in range(len(entropies)):
#     plt.plot(entropies[i][0], label = labels[i])
# plt.legend()
# plt.show()
# plot_max_min(rewards, labels, sigma, True, 'no_norm_learning.png')
    ##

#plot_episodes(rewards, sigma)
#rewards.append(np.squeeze(Seed_rewards[:10] * 4))
#labels = ['Multiple agents learning the same policy', 'Single agents learning a policy']
plot_iterations(rewards,labels,sigma, False, 'Optimal_Comparison.png')

