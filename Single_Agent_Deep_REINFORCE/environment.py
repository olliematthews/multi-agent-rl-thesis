"""
Created on Thu Oct 10 10:31:04 2019

@author: Ollie
"""
import numpy as np

class Cyclist:
    def __init__(self, _velocity = 0, _pose = 0, _energy = 100):
        self.velocity = _velocity
        self.pose = _pose
        self.energy = _energy
        self.action_space = 3
        self.state_space = 4
        self.time = 0
        
    def reset(self, _velocity = 0, _pose = 0, _energy = 100):
        self.velocity = _velocity
        self.pose = _pose
        self.energy = _energy
        self.time = 0
        return self.get_state()
        
    def get_state(self):
        return np.array([self.pose,self.velocity,self.energy, self.time])

    def step(self, action, env_params):
        self.velocity += action
        if self.velocity < 0:
            self.velocity = 0
        if self.velocity > 4:
            self.velocity = 4
        self.pose += self.velocity
        self.energy -= self.velocity ** 3 * env_params['cD']
        self.time += 1
            
        if self.energy <= 0 or self.time >= 200:
            done = True
        else:
            done = False
        return self.get_state(), self.velocity, done
        
    def print_state(self):
        print(f'Pose is {self.pose}, velocity is {self.velocity}, energy is {self.energy}.')
        

