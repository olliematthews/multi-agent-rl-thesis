"""
Created on Thu Oct 10 10:31:04 2019

@author: Ollie
"""
import numpy as np

class Cyclist:
    def __init__(self, env_params, velocity = 0, pose = 0, energy = 100):
        assert env_params['vel_min'] <= 0 
        self.velocity = velocity
        self.pose = pose
        self.energy = energy
        self.action_space = 3
        self.state_space = 4
        self.time = 0
        
    def reset(self, velocity = 0, pose = 0, energy = 100):
        self.velocity = velocity
        self.pose = pose
        self.energy = energy
        self.time = 0
        return self.get_state()
        
    def get_state(self):
        return np.array([self.pose,self.velocity,self.energy,self.time])

    def step(self, action, env_params):
        self.velocity += action
        self.velocity = np.clip(self.velocity, env_params['vel_min'], env_params['vel_max'])
        self.pose += self.velocity
        self.energy -= self.velocity ** 3 * env_params['cD']
        self.time += 1
            
        if self.energy <= 0 or self.time >= env_params['time_limit'] or self.pose >= env_params['race_length']:
            done = True
        else:
            done = False
        return self.get_state(), self.velocity if not done else 0, done
        
    def print_state(self):
        print(f'Pose is {self.pose}, velocity is {self.velocity}, energy is {self.energy}.')
        

