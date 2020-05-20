"""
Created on Sun Nov  3 16:43:48 2019

@author: Ollie
"""
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.losses import mean_squared_error

from copy import copy
import random
from numba import cuda
import gc

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


class State_normaliser:
    def __init__(self, env, hyper_params):
        self.nx = env.state_space
        self.state_subtractor = [env.env_params['race_length'] / 2, env.env_params['vel_max'] / 2, 50.0]
        self.state_dividor = [env.env_params['race_length']/2, env.env_params['vel_max']/2, 50.0]
        # Other pose normaliser
        self.state_subtractor.extend([0.0] * (env.n_cyclists - 1))
        self.state_dividor.extend([50.0] * (env.n_cyclists - 1))

        # Other vel normaliser
        self.state_subtractor.extend([0.0] * (env.n_cyclists - 1))
        self.state_dividor.extend([1.0] * (env.n_cyclists - 1))

        # Other energy normaliser
        self.state_subtractor.extend([50.0] * (env.n_cyclists - 1))
        self.state_dividor.extend([50.0] * (env.n_cyclists - 1))

        self.state_subtractor.append(env.env_params['time_limit'] / 2)
        self.state_dividor.append(env.env_params['time_limit'] / 2)
        self.state_subtractor = np.array(self.state_subtractor)
        self.state_dividor = np.array(self.state_dividor)

        self.reset_arrays()
        self.alpha = hyper_params['state_normaliser_alpha']

    def update_normalisation(self):
        state_stds = np.sqrt(self.state_mean_squareds - self.state_means ** 2)
        # Rounding errors can cause negative sqrts. Set these to 0.
        state_stds[np.isnan(state_stds)] = 0
        
        old_state_subtractor = self.state_subtractor.copy()
        self.state_subtractor = self.alpha * self.state_means + (1 - self.alpha) * self.state_subtractor
        old_state_dividor = self.state_dividor.copy()
        self.state_dividor = self.alpha * state_stds + (1 - self.alpha) * self.state_dividor
        self.reset_arrays()
        return [old_state_dividor, self.state_dividor], [old_state_subtractor, self.state_subtractor]
        
    def normalise_state(self, state):
        alpha = 1 / self.count
        self.state_means = (alpha) * state['state'] + (1 - alpha) * self.state_means
        self.state_mean_squareds = (alpha) * state['state'] ** 2 + (1 - alpha) * self.state_mean_squareds
        self.count += 1
        state['state'] = (state['state'] - self.state_subtractor) / self.state_dividor
        return state 
    
    def normalise_batch(self, states):
        # print('BAHHHH')
        # print(len(states))
        # print(self.state_subtractor.shape)
        # print(self.state_dividor.shape)
        return (states - self.state_subtractor) / self.state_dividor
        
    
    def reset_arrays(self):
        self.state_means = np.array([self.nx,])
        self.state_mean_squareds = np.array([self.nx,])
        self.count = 1

    def save_model(self):
        self.reset_arrays()
        pickle.dump([self.state_subtractor, self.state_dividor], open('normaliser_weights.p', 'wb'))
        
    def load_model(self):
        self.reset_arrays()
        self.state_subtractor, self.state_dividor= pickle.load(open('normaliser_weights.p', 'rb'))
    
class Actor:
    def __init__(self, environment, hyper_params, seed):
        self.lr = hyper_params['lr_actor']
        self.nx = environment.state_space
        self.nA = len(environment.action_space) 
        self.action_space = environment.action_space
        self.seed = seed
        # self.p_order = hyper_params['p_order_actor']
        # self.poly = PolynomialFeatures(self.p_order)
        np.random.seed(seed)
        random.seed(seed)
        self.policy = self.build_policy_network()
        
    def build_policy_network(self):
        w_size = self.nx + 1
        w = np.random.normal(0,0.1,(w_size,self.nA))
        return w
    
    def predict(self, state):
    	z = state.dot(self.policy)
    	exp = np.exp(z - np.max(z))
    	return (exp/np.sum(exp))
        
    def choose_action(self, state):
        # poly_state = self.poly.fit_transform(state.reshape(1,-1))
        poly_state = np.append(np.array([1]),state)
        probs = self.predict(poly_state)
        entropy = self.get_entropy(probs)
        action = random.choices(self.action_space, probs)[0]
        return probs, action, entropy
    
    def get_entropy(self, probs):
        return - np.sum(probs * np.log(probs + 1e-4)) 
    

    def learn(self, actions, states, advantages, lambda_explore, probs):
        poly_states = np.append(np.ones([states.shape[0],1]),states, axis = 1)
        x = np.identity(probs.shape[1])[None,:,:] - probs[:,None,:]
            # np.ones([probs.shape[1],1,]) @ probs.reshape([1,-1])
        dLdx = x[np.arange(x.shape[0]),actions.astype(int),:]
        log_grad = np.dot(poly_states.T, (dLdx * advantages[:,None]))
        
        log_probs = probs * (1 + np.log(probs + 1e-6)) * (1 - (advantages == 0))[:,None]
        entropy_grad = - np.einsum('ij,ikl,ik->jl',poly_states, x, log_probs)
            
        self.policy += self.lr * (log_grad + (1 - lambda_explore) * entropy_grad)
        # return np.linalg.norm(poly_states, axis = 0)
    
class Critic:
    def __init__(self, environment, hyper_params, seed):
        
        def get_entropy(probs):
            return - np.sum(probs * np.log(probs + 1e-4)) 
        
        self.gamma = hyper_params['gamma']
        self.lr = hyper_params['lr_critic']
        self.nx = environment.state_space
        self.nA = len(environment.action_space)
        self.batch_size = hyper_params['TD_batch']
        self.action_space = environment.action_space
        self.seed = seed
        self.reward_memory = np.zeros([self.batch_size,]).astype(float)
        self.entropy_memory = np.empty([self.batch_size,]).astype(float)
        self.state_memory = np.empty([self.batch_size + 1,self.nx]).astype(float)
        self.action_memory = np.empty([self.batch_size,]).astype(int)
        self.prob_memory = np.empty([self.batch_size,self.nA])
        self.n_cyclists = environment.n_cyclists
        self.maximum_entropy = get_entropy(np.array([1/self.nA] * self.nA))
        self.advantages = np.empty([0,])
        self.layers = hyper_params['layers_critic']
        self.alpha_std = hyper_params['alpha_std']
        self.lambda_reward_profile = hyper_params['lambda_reward_profile']
        self.lambda_profile_progress = 0
        self.counter = 0
        self.episode_counter = 0
        
        np.random.seed(seed)
        self.vars = [hyper_params['std_init'] ** 2, 0]
        self.advantage_hangover_sums = [0] * 2
        self.advantage_hangover_lens = [0] * 2
        self.alpha_std = hyper_params['alpha_std']
        self.loc_end = []
        self.losses = []
        self.norms = np.empty([0,8])

        self.model = self.build_policy_network()
        
    def build_policy_network(self):
        states = Input(shape=(self.nx,))
        dense_layers = []
        for layer in self.layers:
            if len(dense_layers) == 0:
                dense_layers.append(Dense(layer, activation='relu', kernel_initializer = glorot_normal(seed=self.seed))(states))
            else:
                dense_layers.append(Dense(layer, activation='relu', kernel_initializer = glorot_normal(seed=self.seed))(dense_layers[-1]))

        value = Dense(1, activation='linear', kernel_initializer = glorot_normal(seed=self.seed))(dense_layers[-1])

        model = Model(inputs=[states], outputs=[value])
        model.compile(optimizer=Adam(lr = self.lr), loss = mean_squared_error)

        return model
        
    def predict(self, states):
        return np.squeeze(self.model.predict(states))

        
    def store_dead(self, state, actor, episode_counter):
        self.state_memory[self.counter] = state
        self.loc_end.append(self.counter)
        # Set these to 1 to avoid erros when computing logs
        self.prob_memory[self.counter,:] = 1
        self.counter += 1
        if self.counter >= self.batch_size:
            # Does not really matter what you put here
            self.state_memory[self.counter] = state
            self.learn(actor, episode_counter)
        
    def store_transition_1(self, state, action, entropy, probs):
        
        self.state_memory[self.counter] = state
        self.action_memory[self.counter] = action
        self.prob_memory[self.counter] = probs
        self.entropy_memory[self.counter] = entropy
        
    def store_transition_2(self, reward, next_state, actor, episode_counter):
        self.reward_memory[self.counter] = reward
        self.counter += 1
        if self.counter >= self.batch_size:
            self.state_memory[self.counter] = next_state
            self.learn(actor, episode_counter)
            
    def store_final_reward(self, reward):
        self.reward_memory[self.counter - 1] = reward
    
    def get_lam(self, episode):
        last_point = self.lambda_reward_profile[self.lambda_profile_progress]
        next_point = self.lambda_reward_profile[self.lambda_profile_progress + 1]
        for i in range(self.lambda_profile_progress, len(self.lambda_reward_profile)):
            if episode <= next_point[0]:
                break
            else:
                self.lambda_profile_progress += 1
                last_point = next_point.copy()
                next_point = self.lambda_reward_profile[self.lambda_profile_progress + 1]
        return last_point[1] + (episode - last_point[0]) / (next_point[0] - last_point[0]) * (next_point[1] - last_point[1])

    def get_stds(self, advantages):
        stds = []
        new_hang_sums = []
        new_hang_lens = []
        new_vars = []
        for ads, s, l, var in zip(advantages, self.advantage_hangover_sums, self.advantage_hangover_lens, self.vars):
            if self.loc_end == []:
                s += np.sum(ads ** 2)
                l += len(ads)
                variances = np.ones_like(ads) * var
            else:
                variances = np.ones_like(ads)
                s += np.sum(ads[:self.loc_end[0]] ** 2)
                l += self.loc_end[0]
                init_ep_variance = s / l
                s = 0
                l = 0
                var = var * (1 - self.alpha_std) + self.alpha_std * init_ep_variance
                variances[:self.loc_end[0]] = var
                for start, end in zip(self.loc_end[:-1], self.loc_end[1:]):
                    var = var * (1 - self.alpha_std) + self.alpha_std * np.mean(ads[start + 1:end] ** 2)
                    variances[start:end] = var
                s += np.sum(ads[self.loc_end[-1] + 1:] ** 2)
                l += len(ads) - self.loc_end[-1] - 1
            new_hang_sums.append(s)
            new_hang_lens.append(l)
            new_vars.append(var)
            stds.append(np.sqrt(variances))
        self.advantage_hangover_sums = new_hang_sums
        self.advantage_hangover_lens = new_hang_lens
        self.vars = new_vars
        return stds
        
    def learn(self, actor, episode_counter):
        self.action_memory[self.loc_end] = 0

        states = np.array(self.state_memory).reshape(-1,self.nx)
        
        rewards = self.reward_memory[:self.action_memory.size]
        entropies = self.entropy_memory[:self.action_memory.size]
        values = self.predict(states)
        value_labels =  rewards + self.gamma * values[1:]

        # Ensure places corresponding to ends of episodes get 0 value
        value_labels[self.loc_end] = 0
        
        self.model.train_on_batch(states[:-1], value_labels)
        # print(np.linalg.norm(values[:-1] - value_labels))
        # self.losses.append(np.linalg.norm(values[:-1] - value_labels))
        probs = np.squeeze(np.array(self.prob_memory))
        values = self.predict(states)  
        
        reward_advantages = rewards + self.gamma * values[1:] - values[:-1]
        entropy_advantages = entropies - self.maximum_entropy
        reward_stds, entropy_stds = self.get_stds([reward_advantages, entropy_advantages])
        reward_advantages /= (reward_stds + 1)
        entropy_advantages /= (entropy_stds + 0.05)
        lambda_reward = self.get_lam(episode_counter)
        advantages = reward_advantages * lambda_reward + entropy_advantages * (1 - lambda_reward)
        advantages[self.loc_end] = 0
        
        # advantages = np.clip(advantages, None, 20)
        actor.learn(self.action_memory, states[:-1,:], advantages, lambda_reward, probs)
        # self.norms = np.append(self.norms, norms.reshape([1,-1]), axis = 0)
        self.reward_memory = np.zeros([self.batch_size,]).astype(float)
        self.entropy_memory = np.empty([self.batch_size,])
        self.state_memory = np.empty([self.batch_size + 1,self.nx])
        self.action_memory = np.empty([self.batch_size,]).astype(int)
        self.prob_memory = np.empty([self.batch_size,self.nA])
        self.loc_end = []
        self.counter = 0

    def update_dividor_subtractor(self, dividors, subtractors, actor): 

        delta_dividors = dividors[1] / dividors[0]
        W = self.model.layers[1].weights[0] * delta_dividors[:,None]
        self.model.layers[1].weights[0].assign(W)
        b = self.model.layers[1].weights[1] + tf.tensordot(W, ((subtractors[1] - subtractors[0]) / dividors[1]).astype('float32'), axes = (0,0))
        self.model.layers[1].weights[1].assign(b)
                
        actor.policy[1:] *= delta_dividors[:,None]
        actor.policy[0] += (np.dot((subtractors[1] - subtractors[0])/dividors[1],actor.policy[1:]))
        
    def clear(self):
        K.clear_session()
        gc.collect()
        del self.model
        for gpu in range(len(cuda.gpus)):
            cuda.select_device(gpu)
            cuda.close()
