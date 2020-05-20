"""
Created on Thu Oct 10 10:31:04 2019

@author: Ollie
"""
import numpy as np

class Cyclist:
    def __init__(self, env_params, number = 0, velocity = 0, pose = 0, energy = 100):
        assert env_params['vel_min'] <= 0 
        self.velocity = velocity
        self.pose = pose
        self.energy = energy
        self.pose_rel_next = 0
        self.vel_rel_next = 0
        self.number = number
        
    def reset(self, velocity = 0, pose = 0, energy = 100):
        self.velocity = 0
        self.pose = pose
        self.energy = energy
        self.pose_rel_next = 0
        self.vel_rel_next = 0
        
    def get_state(self):
        return np.array([self.pose,self.velocity,self.energy, self.pose_rel_next, self.vel_rel_next])

    def step(self, action, env_params):
        self.velocity += action
        self.velocity = max(env_params['vel_min'], min(self.velocity, env_params['vel_max']))
        self.pose += self.velocity
        self.energy -= self.velocity ** 3 * self.cD
        
        if self.pose >= env_params['race_length'] or self.energy <= 0:
            return True
        else:
            return False
        
    def print_state(self):
        print(f'Pose is {self.pose}, velocity is {self.velocity}, energy is {self.energy}.')
        
        
class Environment:
    def __init__(self, env_params, n_cyclists = 4):
        self.n_cyclists = n_cyclists
        self.cyclists = [None] * self.n_cyclists            
        for i in range(self.n_cyclists):
            self.cyclists[i] = Cyclist(env_params, number = i)
            
        self.time = 0
        self.env_params = env_params
        # State space is pose, velocity, energy and relative pose and velocity of other riders
        self.state_space = 5
        self.action_space = np.array([-1,0,1])

    def reset(self):
        self.cyclists = [None] * self.n_cyclists            
        for i in range(self.n_cyclists):
            self.cyclists[i] = Cyclist(self.env_params, number = i)
        self.time = 0
        return self.get_state()
    
    def step(self, action):
        self.time += 1
        
        # Set the drag coefficient
        [self.set_cD(cyclist) for cyclist in self.cyclists]

        # Step each cyclist forward
        self.cyclists = [cyclist for cyclist, act in zip(self.cyclists, action) if not cyclist.step(act, self.env_params)]
        # i = 0
        # while i < len(self.cyclists):
        #     if self.cyclists[i].step(action[i],self.env_params):
        #         self.cyclists.pop(i)
        #         action.pop(i)
        #     else:
        #         i += 1
        

        # print([cyclist.velocity for cyclist in self.cyclists])
        # Order the cyclists into their current order
        self.cyclists.sort(key=lambda x: x.pose, reverse=True)
        
        # Relative position and velocity for next rider for first cyclist is 0
        try:
            self.cyclists[0].pose_rel_next = 0
            self.cyclists[0].vel_rel_next = 0
        except IndexError:
            pass
        
        # Update other cyclists according to cyclist in front
        for i in range(1,len(self.cyclists)):
            self.cyclists[i].pose_rel_next = self.cyclists[i-1].pose - self.cyclists[i].pose 
            self.cyclists[i].vel_rel_next = self.cyclists[i-1].velocity- self.cyclists[i].velocity
                        
        return self.get_state(), self.get_reward(), len(self.cyclists) == 0 or self.time >= self.env_params['time_limit']
        
    def get_reward(self):
        # Reward is the total distance travelled by the cyclists            
        return [cyclist.velocity for cyclist in self.cyclists]
        
    def set_cD(self, cyclist):
        # cD is modelled for now on the function ax/e^(bx)
        cyclist.cD = self.cD_fun(cyclist.pose_rel_next, self.env_params['cD'])
            
    def cD_fun(self,x,cD):
        # cD is modelled for now on the function ax/be^x
        return cD['cDrag'] * (1-(cD['a'] * x / np.exp(x * cD['b'])))
    
    def get_state(self):
        return [{'number' : cyclist.number, 'state' : cyclist.get_state()} for cyclist in self.cyclists]
    
    def print_state(self):
        print(f'At time {self.time}s:')
        for i in range(len(self.cyclists)):
            print(f'Cyclist {self.cyclists[i].number}:')
            self.cyclists[i].print_state()
            print(self.cyclists[i].pose_rel_next)
        print()
        print()
        
        