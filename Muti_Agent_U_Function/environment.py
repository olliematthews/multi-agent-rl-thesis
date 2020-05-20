"""
Created on Thu Oct 10 10:31:04 2019

@author: Ollie
"""
import numpy as np

class Cyclist:
    def __init__(self, env_params, n_cyclists = 4, number = 0, velocity = 0, pose = 0, energy = 100):
        assert env_params['vel_min'] <= 0 
        self.velocity = velocity
        self.pose = pose
        self.energy = energy
        self.rel_poses = [0] * (n_cyclists - 1)
        self.rel_vels = [0] * (n_cyclists - 1)
        self.other_energies = [0] * (n_cyclists - 1)
        self.number = number
        self.done = False
        self.granularity = env_params['granularity']
        
    def reset(self, velocity = 0, pose = 0, energy = 100):
        self.velocity = 0
        self.pose = pose
        self.energy = energy
        self.rel_poses = [0 for i in self.rel_poses]
        self.rel_vels =  [0 for i in self.rel_vels]
        self.other_energies = [0 for i in self.other_energies]

        
    def get_state(self, time):
        state = [self.pose,self.velocity,self.energy]
        state.extend(self.rel_poses)
        state.extend(self.rel_vels)
        state.extend(self.other_energies)
        state.append(time)
        return np.array(state)

    def step(self, action, env_params):
        done_before = self.done
        self.velocity += (action - 1) * self.granularity
        self.velocity = max(env_params['vel_min'], min(self.velocity, env_params['vel_max']))
        self.pose += self.velocity
        self.energy -= self.velocity ** 3 * self.cD
        if self.pose >= env_params['race_length'] or self.energy <= 0:
            self.done = True 
        return done_before

        
    def print_state(self):
        print(f'Pose is {self.pose}, velocity is {self.velocity}, energy is {self.energy}.')
        
        
class Environment:
    def __init__(self, env_params, n_cyclists = 4, poses = [0] * 4):
        self.n_cyclists = n_cyclists
        self.cyclists = [None] * self.n_cyclists            
        for i in range(self.n_cyclists):
            self.cyclists[i] = Cyclist(env_params, number = i, pose = poses[i])
        self.time = 0
        self.env_params = env_params
        # State space is pose, velocity, energy and relative pose and velocity of other riders
        self.state_space = 3 * n_cyclists + 1
        self.action_space = np.array([0,1,2])
        self.mean_distance = 0
        self.mean_velocity = 0

    def reset(self, poses = [0] * 4):
        self.cyclists = [None] * self.n_cyclists            
        for i in range(self.n_cyclists):
            self.cyclists[i] = Cyclist(self.env_params, number = i, pose = poses[i])
        self.time = 0
        distance = self.order_update_cyclists()
        self.mean_velocity = 0 
        self.mean_distance = distance
        return self.get_state()
    
    def step(self, action):
        self.time += 1
        
        # Set the drag coefficient
        [self.set_cD(cyclist) for cyclist in self.cyclists]

        # Step each cyclist forward
        self.cyclists = [cyclist for cyclist, act in zip(self.cyclists, action) if not cyclist.step(act, self.env_params)]
        

        # print([cyclist.velocity for cyclist in self.cyclists])
        distance = self.order_update_cyclists()        
        race_done =  self.time >= self.env_params['time_limit'] or np.array([cyclist.done for cyclist in self.cyclists]).any()
        # print([e.energy for e in self.cyclists])
        # print(self.time)
        
        # self.mean_velocity = 1/(self.time + 1) * np.mean([cyclist.velocity for cyclist in self.cyclists]) + (1 - 1/(self.time + 1)) * self.mean_velocity
        self.mean_distance = 1/(self.time + 1) * (distance) + (1 - 1/(self.time + 1)) * self.mean_distance
        
        return self.get_state(), self.get_reward(), race_done, [distance, self.mean_velocity, self.mean_distance]

    def order_update_cyclists(self):
        # Order the cyclists into their current order
        self.cyclists.sort(key=lambda x: x.pose, reverse=True)
        
        distance = self.cyclists[0].pose - self.cyclists[-1].pose
        for cyclist in self.cyclists:
            cyclist.rel_poses = [c.pose - cyclist.pose for c in self.cyclists if not c == cyclist]
            cyclist.rel_vels = [c.velocity - cyclist.velocity for c in self.cyclists if not c == cyclist]
            cyclist.other_energies = [c.energy for c in self.cyclists if not c == cyclist]

        return distance
    
    def get_reward(self):
        return [cyclist.velocity for cyclist in self.cyclists]
        
    def set_cD(self, cyclist):
        # cD is modelled for now on the function ax/e^(bx)
        cD = self.env_params['cD']
        pos_poses = np.array([p for p in cyclist.rel_poses])
        calc = cD['cDrag'] * (1-(cD['a'] * (pos_poses+cD['offset']) / np.exp((pos_poses+cD['offset']) * cD['b'])))
        cyclist.cD = min([np.min(calc), cD['cDrag']])
                
    def get_state(self):
        return [{'number' : cyclist.number, 'state' : cyclist.get_state(self.time)} for cyclist in self.cyclists]

    
    def print_state(self):
        print(f'At time {self.time}s:')
        for i in range(len(self.cyclists)):
            print(f'Cyclist {self.cyclists[i].number}:')
            self.cyclists[i].print_state()
            print(self.cyclists[i].pose_rel_next)
        print()
        print()
        
        