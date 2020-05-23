"""
Functions for plotting the results of simulations.
"""
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import gridspec


def parse_data(results):
    '''
    Parse the results file.
    '''
    rewards = []
    entropies = []
    model_best = []
    for i in range(len(results)):
        rewards.append(results[i][0])
        entropies.append(results[i][1])
    rewards = np.array(rewards)
        
    return rewards, entropies, model_best

def plot_episode(rewards, sigma):
    '''
    Plot the individual agent performances for a simulation.
    '''
    for i in range(rewards.shape[0]):
        for j in range(rewards[i].shape[1]):
            plt.plot(gaussian_filter(rewards[i,:,j], sigma = sigma))
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title(f'Seed {i}')
        plt.show()

def plot_episodes(rewards, sigma,save = False, filename = '', title = ''):
    '''
    Will plot the individual seeds for a set of simulations.
    '''
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
    '''
    Main plotting function. Will plot average and std rewards for different 
    iterations.
    '''
    plt.figure()
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
    plt.legend(loc = 'lower center')
    if save:
        plt.savefig(filename)
    plt.show()
    
    
def plot_entropy_rewards(reward,entropy,sigma, save = False, filename = ''):
    '''
    Used to plot stacked plots of reward and entropy.
    '''
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
    
def join_compare(files, nums):
    '''
    Can be used to join the results of two different sets of simulations.

    Parameters
    ----------
    files : list
        The p files to be joined.
    nums : list
        The iteration number in each p file to be taken.

    Returns
    -------
    data : list
        A combined data file which can be fed into plot_iterations.

    '''
    data = []
    for file, num in zip(files, nums):
        data_, _, _ = pickle.load(open(file,'rb'))
        data.append(data_[num])
    return data

sigma = 1


data, labels, params = pickle.load(open('LR_c.p','rb'))


rewards = []
entropies = []
exploration = []
model_best = []
# labels = [None]
for i in range(len(data)):
    r, e, m = parse_data(data[i])
    rewards.append(r)
    entropies.append(e)
    model_best.append(m)

plot_iterations(rewards, labels, sigma)