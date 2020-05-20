"""
Created on Sun Nov  3 16:43:48 2019

@author: Ollie
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import comb

class Actor:
    def __init__(self, environment, hyper_params, seed):
        self.lr = hyper_params['lr_actor']
        self.nx = environment.state_space
        self.nA = len(environment.action_space)
        self.batch_size = hyper_params['batch_size']
        self.action_space = environment.action_space
        self.seed = seed
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.entropy_memory = []
        np.random.seed(seed)
        self.policy = self.build_policy_network()

    def build_policy_network(self):
        w = np.random.normal(0,0.1,(self.nx,self.nA))
        return w
    
    def predict(self, state):
    	z = state.dot(self.policy)
    	exp = np.exp(z - np.max(z))
    	return (exp/np.sum(exp))
        
    def choose_action(self, state):
        probs = self.predict(state)
        entropy = self.get_entropy(probs)
        action = np.random.choice(self.action_space, p=probs)
        return probs, action, entropy
    
    def get_entropy(self, probs):
        return - np.sum(probs * np.log(probs + 1e-4)) 
    

    def learn(self, actions, states, advantages, probs):
        dLdx = actions - probs
        grads = states.T @ (dLdx * advantages)
        self.policy += grads * self.lr
        
class Individual_Critic:
    def __init__(self, environment, hyper_params, seed):
        self.gamma = hyper_params['gamma']
        self.lr = hyper_params['lr_critic']
        self.nx = environment.state_space
        self.nA = len(environment.action_space)
        self.batch_size = hyper_params['TD_batch']
        self.lambda_reward = hyper_params['lambda_reward']
        self.action_space = environment.action_space
        self.seed = seed
        self.state_memory = {}
        self.action_memory = {}
        self.prob_memory = {}
        self.n_cyclists = environment.n_cyclists
        self.reward_memory = np.empty([hyper_params['TD_batch'],1])
        self.entropy_memory = np.empty([hyper_params['TD_batch'],1])
        self.advantages = np.empty([0,])
        self.p_order = hyper_params['p_order']
        self.poly = PolynomialFeatures(hyper_params['p_order'])
        self.counter = 0
        np.random.seed(seed)
        self.policy = self.build_policy_network()

    def build_policy_network(self):
        w_size = int(comb(self.nx + self.p_order, self.p_order))
        w = np.random.normal(0,0.1,(w_size,1))
        return w
    
    def predict(self, states):
        return states @ self.policy

    def store_transition(self, states, actions, reward, entropy, probs, next_states, actor):
        
        for i in range(len(states)):
            if states[i]['number'] not in self.state_memory:
                self.state_memory[states[i]['number']] = [states[i]['state']]
                self.action_memory[states[i]['number']] = [actions[i]]
                self.prob_memory[states[i]['number']] = [probs[i]]
            else:
                self.state_memory[states[i]['number']].append(states[i]['state'])
                self.action_memory[states[i]['number']].append(actions[i])
                self.prob_memory[states[i]['number']].append(probs[i])
        self.reward_memory[self.counter] = reward
        self.entropy_memory[self.counter] = entropy
        self.counter += 1
        
        if self.counter >= self.batch_size:
            for i in range(len(next_states)):
                self.state_memory[next_states[i]['number']].append(next_states[i]['state'])
            self.learn(actor)

    def learn(self, actor):
        poly_states = []
        policy_change = np.zeros_like(self.policy)
        self.reward_memory += self.lambda_reward * self.entropy_memory
        # print(self.reward_memory.shape)
        for (_, states), (_, actions) in zip(self.state_memory.items(), self.action_memory.items()):
            # states = np.squeeze(np.array(states))
            # actions = np.squeeze(np.array(actions))
            actions = np.array(actions).reshape(-1,self.nA)
            states = np.array(states).reshape(-1,self.nx)
            if np.sum(actions[-1,:]) == 0:
                states = np.append(states,np.zeros([1,self.nx]), axis = 0)
            
            polys = self.poly.fit_transform(states)
            rewards = self.reward_memory[:actions.shape[0]]
            loc_end = np.where(np.sum(actions,axis = 1) == 0)
            values = self.predict(polys)
            value_labels =  rewards + self.gamma * values[1:]
                

            # Ensure places corresponding to ends of episodes get 0 value
            value_labels[loc_end] = 0
            poly_states.append(polys)
            policy_change += self.lr * polys[:-1].T @ (2 * (values[:-1] - value_labels)) / self.n_cyclists
        print(np.linalg.norm(policy_change))
        self.policy += policy_change
        i = 0
        for (_, states), (_, actions), (_,probs) in zip(self.state_memory.items(), self.action_memory.items(), self.prob_memory.items()):
            actions = np.array(actions).reshape(-1,self.nA)
            states = np.array(states).reshape(-1,self.nx)
            if np.sum(actions[-1,:]) == 0:
                states = np.append(states,np.zeros([1,self.nx]), axis = 0)
            probs = np.squeeze(np.array(probs))
            values = self.predict(poly_states[i])  
            rewards = self.reward_memory[:len(actions),0]
            advantages = rewards[:,None] + self.gamma * values[1:] - values[:-1]
            loc_end = np.where(np.sum(actions,axis = 1) == 0)
            advantages[loc_end,:] = 0
            actor.learn(actions, states[:-1], advantages, probs)
            i += 1

        self.reward_memory = np.empty([self.batch_size,1])
        self.entropy_memory = np.empty([self.batch_size,1])
        self.state_memory = {}
        self.action_memory = {}
        self.prob_memory = {}

        self.counter = 0
        
        