"""
Created on Thu Oct 10 10:31:04 2019

@author: Ollie
"""
import numpy as np

class Cyclist:
    def __init__(self, env_params, number = 0, velocity = 0, pose = 0, energy = 100):
        '''
        Initialiser

        Parameters
        ----------
        env_params : dict
            A dictionary which contains the info needed to describe the 
            environment
        velocity : int, optional
            Initial velocity of the cyclist (usually 0)
        pose : int, optional
            Initial position of the cyclist. The default is 0.
        energy : float, optional
            Initial energy of the cyclist. The default is 100.

        Returns
        -------
        None.

        '''
        assert env_params['vel_min'] <= 0 
        self.velocity = velocity
        self.pose = pose
        self.energy = energy
        self.pose_rel_next = 0
        self.vel_rel_next = 0
        self.number = number
        self.done = False
        
    def reset(self, velocity = 0, pose = 0, energy = 100):
        '''
        Reset the cyclist

        Parameters
        ----------
        velocity : int, optional
            Initial velocity of the cyclist (usually 0)
        pose : int, optional
            Initial position of the cyclist. The default is 0.
        energy : float, optional
            Initial energy of the cyclist. The default is 100.

        Returns
        -------
        state : np.array
            The cyclist's state.

        '''
        self.velocity = 0
        self.pose = pose
        self.energy = energy
        self.pose_rel_next = 0
        self.vel_rel_next = 0
        
    def get_state(self, time):
        '''
        Returns an array describing the cyclist's state
        Parameters
        ----------
        time : int
        
        Returns
        -------
        state : np.array
            The state, given by [position, velocity, energy, next_pose, 
                                 next_vel, time].

        '''
        return np.array([self.pose,self.velocity,self.energy, self.pose_rel_next, self.vel_rel_next, time])

    def step(self, action, env_params):
        '''
        Steps the cyclist forward in an episode

        Parameters
        ----------
        action : int
            An integer between 0 and 2. 0 means slow down (by 1 velocity unit),
            2 means speed up (by 1 velocity unit), 1 means do nothing. 
        env_params : dict
            A dictionary which contains the info needed to describe the 
            environment

        Returns
        -------
        done_before : bool
            True when a cyclist has run out before the step

        '''
        done_before = True if self.pose >= env_params['race_length'] or self.energy <= 0 else False
        self.velocity += action - 1
        self.velocity = max(env_params['vel_min'], min(self.velocity, env_params['vel_max']))
        self.pose += self.velocity
        self.energy -= self.velocity ** 3 * self.cD
        if self.pose >= env_params['race_length'] or self.energy <= 0:
            self.done = True 
        return done_before

        
    def print_state(self):
        '''
        Print the cyclsit's state

        Returns
        -------
        None.

        '''
        print(f'Pose is {self.pose}, velocity is {self.velocity}, energy is {self.energy}.')
        
        
class Environment:
    def __init__(self, env_params, n_cyclists = 4):
        '''
        Initialiser.

        Parameters
        ----------
        env_params : dict
            A dictionary which contains the info needed to describe the 
            environment
        n_cyclists : int, optional
            Number of cyclists in the environment. The default is 4.

        Returns
        -------
        None.

        '''
        self.n_cyclists = n_cyclists
        self.cyclists = [None] * self.n_cyclists            
        for i in range(self.n_cyclists):
            self.cyclists[i] = Cyclist(env_params, number = i)
            
        self.time = 0
        self.env_params = env_params
        # State space is pose, velocity, energy and relative pose and velocity of other riders
        self.state_space = 6
        self.action_space = np.array([0,1,2])

    def reset(self):
        '''
        Reset the environment.

        Returns
        -------
        state : np.array
            The environment's state.

        '''
        self.cyclists = [None] * self.n_cyclists            
        for i in range(self.n_cyclists):
            self.cyclists[i] = Cyclist(self.env_params, number = i)
        self.time = 0
        return self.get_state()
    
    def step(self, action):
        '''
        Steps the cyclists forward in an episode.

        Parameters
        ----------
        action : list
            The set of actions taken by each agent.

        Returns
        -------
        state : np.array
            The environment state.
        rewards : np.array
            The individual reward for each cyclist.
        race_done : bool
            True the whole race is finished.
        cyclists_done : list
            A list of indexes of cyclist numbers for each cyclist which is 
            done.

        '''
        self.time += 1
        
        # Set the drag coefficient
        [self.set_cD(cyclist) for cyclist in self.cyclists]

        # Step each cyclist forward
        self.cyclists = [cyclist for cyclist, act in zip(self.cyclists, action) if not cyclist.step(act, self.env_params)]
        
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
            
        cyclists_done = [c.number for c in self.cyclists if c.done or self.time == self.env_params['time_limit']]
        race_done = True if len(self.cyclists) == len(cyclists_done) or self.time > self.env_params['time_limit'] else False
        return self.get_state(), self.get_reward(), race_done, cyclists_done
        
    def get_reward(self):
        '''
        The reward function.

        Returns
        -------
        rewards : list
            The reward for each cyclist.

        '''
        # Reward is the total distance travelled by the cyclists            
        return [cyclist.velocity for cyclist in self.cyclists]
        
    def set_cD(self, cyclist):
        '''
        Sets the coefficient of drag for each cyclist.

        Parameters
        ----------
        cyclist : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # cD is modelled for now on the function ax/e^(bx)
        cyclist.cD = self.cD_fun(cyclist.pose_rel_next, self.env_params['cD'])
            
    def cD_fun(self,x,cD):
        '''
        Calculates the drag coefficient

        Parameters
        ----------
        x : int
            Distance to rider in front.
        cD : float
            The coefficient of drag without slipstreaming.

        Returns
        -------
        cD
            The coefficient of drag.

        '''
        # cD is modelled for now on the function ax/be^x
        return cD['cDrag'] * (1-(cD['a'] * x / np.exp(x * cD['b'])))
    
    def get_state(self):
        '''
        Returns a list describing the environment's state

        Returns
        -------
        state : list
            A list of dictionaries, with the cyclist number and their state

        '''
        return [{'number' : cyclist.number, 'state' : cyclist.get_state(self.time)} for cyclist in self.cyclists]
    
    def print_state(self):
        '''
        Print the environment's state

        Returns
        -------
        None.

        '''
        print(f'At time {self.time}s:')
        for i in range(len(self.cyclists)):
            print(f'Cyclist {self.cyclists[i].number}:')
            self.cyclists[i].print_state()
            print(self.cyclists[i].pose_rel_next)
        print()
        print()
