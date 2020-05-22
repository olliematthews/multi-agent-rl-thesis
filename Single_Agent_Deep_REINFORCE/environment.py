"""
The single agent environment contains only one class - the cyclist.
"""
import numpy as np

class Cyclist:
    def __init__(self, env_params, velocity = 0, pose = 0, energy = 100):
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
        self.action_space = 3
        self.state_space = 4
        self.time = 0
        
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

        self.velocity = velocity
        self.pose = pose
        self.energy = energy
        self.time = 0
        return self.get_state()
        
    def get_state(self):
        '''
        Returns a list describing the cyclist's state

        Returns
        -------
        state : np.array
            The state, given by [position, velocity, energy, time].

        '''
        return np.array([self.pose,self.velocity,self.energy,self.time])

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
        state : np.array
            The cyclist's state.
        reward : int
            In this environment, the reward is simply the cyclist's velocity.
        done : bool
            True when a cyclist runs out of energy or time.

        '''
        self.velocity += action
        self.velocity = np.clip(self.velocity, env_params['vel_min'], env_params['vel_max'])
        self.pose += self.velocity
        self.energy -= self.velocity ** 3 * env_params['cD']
        self.time += 1
            
        if self.energy < 0 or self.time > env_params['time_limit'] or self.pose >= env_params['race_length']:
            done = True
        else:
            done = False
        return self.get_state(), self.velocity, done
        
    def print_state(self):
        '''
        Print the cyclsit's state

        Returns
        -------
        None.

        '''
        print(f'Pose is {self.pose}, velocity is {self.velocity}, energy is {self.energy}.')
