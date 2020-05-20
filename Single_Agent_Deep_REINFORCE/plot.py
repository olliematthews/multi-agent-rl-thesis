# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 17:14:28 2019

@author: Ollie
"""
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter
import numpy as np


rewards = pickle.load(open('Rewards_8.p','rb'))
_, rewards_, entropies_, velocities_, _ = pickle.load(open('Whitened.p','rb'))
rewards_ = np.squeeze(rewards_)

fig = plt.figure(figsize = (12, 4.5))
for i in range(10):

    plt.plot(gaussian_filter(rewards[i],sigma = 20))
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Deep')
plt.savefig('Deep')
plt.show()


fig = plt.figure(figsize = (12, 4.5))

for i in range(10):

    plt.plot(gaussian_filter(rewards_[i],sigma = 20))
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Linear')
plt.savefig('Linear')

plt.show()


rewards = np.array(rewards)
average_rewards = gaussian_filter(np.mean(rewards, axis = 0), sigma = 20)
std_rewards = gaussian_filter(np.std(rewards, axis = 0), sigma = 20)


fig = plt.figure(figsize = (12, 8))

average_rewards_ = gaussian_filter(np.mean(rewards_, axis = 0), sigma = 20)
std_rewards_ = gaussian_filter(np.std(rewards_, axis= 0), sigma = 20)
ax = plt.gca()
color = next(ax._get_lines.prop_cycler)['color'] 
plt.fill_between(x = range(average_rewards.size),y1 =  average_rewards + std_rewards, y2 = average_rewards - std_rewards, color = color, alpha = 0.3)
plt.plot(average_rewards, label = 'Deep', color = color)
color = next(ax._get_lines.prop_cycler)['color'] 
plt.fill_between(range(average_rewards_.size), average_rewards_ + std_rewards_, average_rewards_ - std_rewards_, color = color, alpha = 0.3)
plt.plot(average_rewards_, label = 'Linear')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend(loc = 'lower right')
plt.savefig('Deep_Vs_Linear')
