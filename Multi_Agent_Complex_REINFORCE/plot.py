# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 17:14:28 2019

@author: Ollie
"""
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter
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


def parse_data(results):
    rewards = []
    entropies = []
    model_best = []
    for i in range(len(results)):
        rewards.append(results[i][0])
        # r = []
        # for j in range(results[i][0][4:].size):
        #     r.append(results[i][0][4 + j])
        # rewards.append(np.array(r))
        entropies.append(results[i][1])
        # model_best.append(results[i][2])
    rewards = np.array(rewards)
        
    return rewards, entropies, model_best

def plot_episode(rewards, sigma):
    for i in range(rewards.shape[0]):
        if not i == 3:
            continue
        figure = plt.figure(figsize = (3.5,2.8))

        for j in range(rewards[i].shape[1]):
            plt.plot(gaussian_filter(rewards[i,:,j], sigma = sigma), label = f'Rider {j}')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Reward')
        plt.legend()
        # plt.title(f'Seed {i}')
        plt.savefig('initial_learning.pgf', bbox = "tight")
        plt.show()

def plot_episodes(rewards, sigma,save = False, filename = '', title = ''):
    for i in range(rewards.shape[0]):
        plt.plot(gaussian_filter(np.mean(rewards,axis = 2)[i,:], sigma = sigma), label = f'Seed {i}')
        
#    plt.legend()
    plt.xlabel('Number of Episodes')
    plt.ylabel('Reward')
    if save:
        plt.savefig(filename)
    plt.savefig('Episodes')
    plt.title(title)
    plt.show()

def plot_iterations(rewards, labels, sigma, save = False, filename = ''):
    plt.figure(figsize = (5,3))

    ax = plt.gca()
    for i in range(len(rewards)):
        color = next(ax._get_lines.prop_cycler)['color']
        rewards_mean = np.mean(rewards[i],axis = (0,2))
        rewards_std = np.std(rewards[i],axis = (0,2))
        rewards_mean = gaussian_filter(rewards_mean,sigma = sigma)
        rewards_std = gaussian_filter(rewards_std,sigma = sigma)
        plt.plot(rewards_mean, color = color, label = labels[i])
        plt.fill_between(range(rewards_mean.size),rewards_mean + rewards_std, rewards_mean - rewards_std, alpha = 0.3)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.legend(loc = 'lower center')
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
    

sigma = 5


data, labels, params = pickle.load(open('PG_Multi.p','rb'))
# data_, _, params = pickle.load(open('Exploring_com6.p','rb'))
# data__, _, _ = pickle.load(open('Exploring_com6.p', 'rb'))
#_,_, _, Seed_rewards, _, _, _ =     pickle.load(open('BigTest.p','rb'))



rewards = []
entropies = []
exploration = []
model_best = []

for i in range(len(data)):
    r, e, m = parse_data(data[i])
    # r_, _, _ = parse_data(data_[i])
    # # r__ , _, _ = parse_data(data__[i])
    # r = np.append(r[:,:4000,:],r_, axis = 0)
    # r = np.append(r,r__, axis = 0)
    
    # To deal with error in saving
    # r = r[:,4:]
    # e = [a[4:] for a in e]
    rewards.append(r)
    entropies.append(e)
    model_best.append(m)
    plot_episode(r, sigma)
    # plot_episodes(r,sigma)

# for i in range(len(entropies)):
#     plt.plot(entropies[i][0], label = labels[i])
# plt.legend()
# plt.show()
# labels = ['Lambda = ' + str(i) for i in labels]
# plot_iterations(rewards, labels, sigma)
##

#plot_episodes(rewards, sigma)
#rewards.append(np.squeeze(Seed_rewards[:10] * 4))
#labels = ['Multiple agents learning the same policy', 'Single agents learning a policy']
#plot_iterations(rewards,labels,sigma, True, 'Optimal_Comparison.png')

