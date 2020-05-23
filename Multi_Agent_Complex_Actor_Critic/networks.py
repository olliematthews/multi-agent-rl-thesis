"""
Contains the networks which describe each agent.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import comb

class State_normaliser:
    '''
    Used to ensure inputs to networks are normalised.
    '''
    def __init__(self, env_params):
        self.state_subtractor = np.array([env_params['race_length'] / 2, env_params['vel_max'] / 2, 50,1,env_params['vel_max'] / 2, env_params['time_limit'] / 2])
        self.state_dividor = np.array([env_params['race_length']/2, env_params['vel_max']/2, 50,env_params['race_length'] / 4,env_params['vel_max'], env_params['time_limit'] / 2])

    def normalise_state(self, state):
        state['state'] = (state['state'] - self.state_subtractor) / self.state_dividor
        return state
class Actor:
    def __init__(self, environment, hyper_params, seed):
        '''
        Initialiser

        Parameters
        ----------
        environment : Environment
        hyper_params : dict
        seed : int
            Random int.

        Returns
        -------
        None.

        '''
        self.lr = hyper_params['lr_actor']
        self.nx = environment.state_space
        self.nA = len(environment.action_space) 
        self.action_space = environment.action_space
        self.seed = seed
        self.p_order = hyper_params['p_order_actor']
        self.poly = PolynomialFeatures(self.p_order)
        self.alpha_std = hyper_params['alpha_std']
        np.random.seed(seed)
        self.policy = self.build_policy_network()
        self.var = hyper_params['std_init']
        
    def build_policy_network(self):
        '''
        Initialise weight array.

        Returns
        -------
        w : np.array
            Policy weight matrix.

        '''
        w_size = int(comb(self.nx + self.p_order, self.p_order))
        w = np.random.normal(0,0.1,(w_size,self.nA))
        return w
    
    def predict(self, state):
        '''
        Get the probabilities for taking each action.

        Parameters
        ----------
        state : np.array

        Returns
        -------
        probs : np.array

        '''
        z = state.dot(self.policy)
        exp = np.exp(z - np.max(z))
        return (exp/np.sum(exp))
        
    def choose_action(self, state):
        '''
        Choose the action according to the policy and state.

        Parameters
        ----------
        state : np.array

        Returns
        -------
        probs : np.array
        action : int
        entropy : float

        '''
        poly_state = self.poly.fit_transform(state.reshape(1,-1))
        probs = self.predict(poly_state)
        entropy = self.get_entropy(probs)
        action = np.random.choice(self.action_space, p = np.squeeze(probs))
        return probs, action, entropy
    
    def get_entropy(self, probs):
        '''
        Calculates Shannon entropy.

        Parameters
        ----------
        probs : np.array
            The output probabilities from the policy.

        Returns
        -------
        entropy : float
            The entropy of the probabilities.

        '''
        return - np.sum(probs * np.log(probs + 1e-4)) 
    

    def learn(self, actions, states, advantages, probs):
        '''
        Update the policy based on state, action, advantage tuples.
        Parameters
        ----------
        actions : np.array
        states : np.array
        advantages : np.array
        probs : np.array
        
        Returns
        -------
        None.

        '''
        self.var = self.var * (1 - self.alpha_std) + self.alpha_std * np.mean(advantages ** 2)
        dLdx = actions - probs
        grads = self.poly.fit_transform(states).T @ (dLdx * advantages / np.sqrt(self.var))
        self.policy += grads * self.lr
        
class Critic:
    def __init__(self, environment, hyper_params, seed):
        '''
        Initialiser

        Parameters
        ----------
        environment : Environment
        hyper_params : dict
        seed : int
            Random int.

        Returns
        -------
        None.

        '''
        self.gamma = hyper_params['gamma']
        self.lr = hyper_params['lr_critic']
        self.nx = environment.state_space
        self.nA = len(environment.action_space)
        self.batch_size = hyper_params['TD_batch']
        self.lambda_reward = hyper_params['lambda_reward']
        self.action_space = environment.action_space
        self.seed = seed
        self.reward_memory = np.empty([self.batch_size,1]).astype(int)
        self.entropy_memory = np.empty([self.batch_size,1])
        self.state_memory = np.empty([self.batch_size + 1,self.nx])
        self.action_memory = np.empty([self.batch_size,]).astype(int)
        self.prob_memory = np.empty([self.batch_size,self.nA])
        self.n_cyclists = environment.n_cyclists
        self.advantages = np.empty([0,])
        self.p_order = hyper_params['p_order_critic']
        self.poly = PolynomialFeatures(self.p_order)
        self.counter = 0
        np.random.seed(seed)
        self.weights = self.build_policy_network()
        self.losses = []

    def build_policy_network(self):
        '''
        Initialise critic weight array.

        Returns
        -------
        w : np.array
            Critic weight matrix.

        '''
        w_size = int(comb(self.nx + self.p_order, self.p_order))
        w = np.random.normal(0,0.1,(w_size,1))
        return w
    
    def predict(self, states):
        '''
        Estimate the value of states.

        Parameters
        ----------
        states : np.array

        Returns
        -------
        value : np.array
            The value of the states.

        '''
        return states @ self.weights

    def store_dead_cyclist(self, state):
        '''
        Store the transitions for cyclists who are in dead states. These need
        special treatment

        Parameters
        ----------
        state : np.array

        Returns
        -------
        None.

        '''
        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = -1
        self.counter += 1
        
    def store_dead(self, state, actor):
        '''
        Store the transition after the whole simulation is dead.

        Parameters
        ----------
        state : np.array
        actor : Actor
            The actor for the associated agent.

        Returns
        -------
        None.

        '''
        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = -1
        self.counter += 1
        if self.counter >= self.batch_size:
            # Does not really matter what you put here
            self.state_memory[self.counter] = state
            self.learn(actor)
        
    def store_transition_1(self, state, action, entropy, probs):
        '''
        Store the transition, done before the step is taken.

        Parameters
        ----------
        state : np.array
        action : int
        entropy : float
        probs : np.array

        Returns
        -------
        None.

        '''
        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = action
        self.prob_memory[self.counter] = probs
        self.entropy_memory[self.counter] = entropy
        
    def store_transition_2(self, reward, next_state, actor):
        '''
        Store the transition, done after the step is taken in the environment.

        Parameters
        ----------
        reward : float
        next_state : np.array
        actor : Actor
            The actor for the associated agent.

        Returns
        -------
        None.

        '''
        self.reward_memory[self.counter] = reward
        self.counter += 1
        if self.counter >= self.batch_size:
            self.state_memory[self.counter] = next_state
            self.learn(actor)

    def learn(self, actor):
        '''
        Update the critic and actor policies

        Parameters
        ----------
        actor : Actor
            The actor for the associated agent.

        Returns
        -------
        None.

        '''

        policy_change = np.zeros_like(self.weights)
        loc_end = np.where(self.action_memory == -1)
        self.action_memory[loc_end] = 0

        # self.reward_memory += self.lambda_reward * self.entropy_memory
        actions = np.zeros([self.batch_size,self.nA])
        actions[np.arange(actions.shape[0]), self.action_memory] = 1
        states = np.array(self.state_memory).reshape(-1,self.nx)
        if actions[-1,0] == -1:
            states = np.append(states,np.zeros([1,self.nx]), axis = 0)
        
        polys = self.poly.fit_transform(states)
        rewards = self.reward_memory[:actions.shape[0]]
        values = self.predict(polys)
        # print(values)
        value_labels =  rewards + self.gamma * values[1:]
            

        # Ensure places corresponding to ends of episodes get 0 value
        value_labels[loc_end] = 0
        self.weights -= self.lr * polys[:-1].T @ (2 * (values[:-1] - value_labels))
        # print(np.linalg.norm(values[:-1] - value_labels))
        self.losses.append(np.linalg.norm(values[:-1] - value_labels))
        probs = np.squeeze(np.array(self.prob_memory))
        values = self.predict(polys)  
        rewards = self.reward_memory[:len(actions),0]
        advantages = rewards[:,None] + self.gamma * values[1:] - values[:-1]
        advantages[loc_end,:] = 0
        actor.learn(actions, states[:-1,:], advantages, probs)

        self.reward_memory = np.empty([self.batch_size,1]).astype(int)
        self.entropy_memory = np.empty([self.batch_size,1])
        self.state_memory = np.empty([self.batch_size + 1,self.nx])
        self.action_memory = np.empty([self.batch_size,]).astype(int)
        self.prob_memory = np.empty([self.batch_size,self.nA])

        self.counter = 0

