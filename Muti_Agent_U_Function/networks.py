"""
Contains the networks which describe each agent.
"""
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.losses import mean_squared_error

from copy import copy
import random
 
# Restrict TensorFlow to only allocate 512 MB of memory on the first GPU

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
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
        '''
        This function updates the normalisation paramters at the end of an 
        episode. This helps to deal with problems occuring from state 
        distribution drift.

        Returns
        -------
        state_dividors : list
            A list with the old and new state dividors. The old value is needed
            to adjust the policy and value function.
        state_subtractors : list
            A list with the old and new state subtractors.

        '''
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
        '''
        We normalise the state and also add the state to the normaliser's 
        memory
        '''
        alpha = 1 / self.count
        self.state_means = (alpha) * state['state'] + (1 - alpha) * self.state_means
        self.state_mean_squareds = (alpha) * state['state'] ** 2 + (1 - alpha) * self.state_mean_squareds
        self.count += 1
        state['state'] = (state['state'] - self.state_subtractor) / self.state_dividor
        return state 
    
    def normalise_batch(self, states):
        '''
        Normalise an entire batch
        '''
        return (states - self.state_subtractor) / self.state_dividor
        
    
    def reset_arrays(self):
        self.state_means = np.array([self.nx,])
        self.state_mean_squareds = np.array([self.nx,])
        self.count = 1

    
class Actor:
    def __init__(self, environment, hyper_params, seed):
        self.lr = hyper_params['lr_actor']
        self.nx = environment.state_space
        self.nA = len(environment.action_space) 
        self.action_space = environment.action_space
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.policy = self.build_policy_network()
        
    def build_policy_network(self):
        w_size = self.nx + 1
        w = np.random.normal(0,0.1,(w_size,self.nA))
        return w
    
    def predict(self, state):
        '''
        Get the probability of taking each action
        '''
        z = state.dot(self.policy)
        exp = np.exp(z - np.max(z))
        return (exp/np.sum(exp))
        
    def choose_action(self, state):
        '''
        Choose you action according to a seeded generator and the policy.
        '''
        poly_state = np.append(np.array([1]),state)
        probs = self.predict(poly_state)
        entropy = self.get_entropy(probs)
        action = random.choices(self.action_space, probs)[0]
        return probs, action, entropy
    
    def get_entropy(self, probs):
        '''
        Calculate the Shannon entropy of a set of probabilities
        '''
        return - np.sum(probs * np.log(probs + 1e-4)) 
    

    def learn(self, actions, states, advantages, lambda_explore, probs):
        '''
        Update the policy. We introdue entropy regularisation, with lambda
        explore indicating the weight towards exploitation. 
        '''
        poly_states = np.append(np.ones([states.shape[0],1]),states, axis = 1)
        x = np.identity(probs.shape[1])[None,:,:] - probs[:,None,:]
            # np.ones([probs.shape[1],1,]) @ probs.reshape([1,-1])
        dLdx = x[np.arange(x.shape[0]),actions.astype(int),:]
        log_grad = np.dot(poly_states.T, (dLdx * advantages[:,None]))
        
        log_probs = probs * (1 + np.log(probs + 1e-6)) * (1 - (advantages == 0))[:,None]
        entropy_grad = - np.einsum('ij,ikl,ik->jl',poly_states, x, log_probs)
            
        self.policy += self.lr * (log_grad + (1 - lambda_explore) * entropy_grad)

    
class Critic:
    def __init__(self, environment, hyper_params, seed):
        
        def get_entropy(probs):
            return - np.sum(probs * np.log(probs + 1e-4)) 
        self.n_cyclists = environment.n_cyclists
        self.gamma = hyper_params['gamma']
        self.lr = hyper_params['lr_critic']
        self.nx = environment.state_space
        self.nA = len(environment.action_space)
        self.batch_size = hyper_params['TD_batch']
        self.action_space = environment.action_space
        self.seed = seed
        self.reset_arrays()
        self.n_cyclists = environment.n_cyclists
        self.maximum_entropy = get_entropy(np.array([1/self.nA] * self.nA))
        self.advantages = np.empty([0,])
        self.layers = hyper_params['layers_critic']
        self.alpha_std = hyper_params['alpha_std']
        self.lambda_reward_profile = hyper_params['lambda_reward_profile']
        self.lambda_profile_progress = 0
        self.counter = 0
        self.episode_counter = 0
        self.significant_prob = hyper_params['significant_prob']
        
        np.random.seed(seed)
        self.vars = [hyper_params['std_init'] ** 2, 0]
        self.advantage_hangover_sums = [0] * 2
        self.advantage_hangover_lens = [0] * 2
        self.alpha_std = hyper_params['alpha_std']
        self.loc_end = []
        self.losses = []
        self.norms = np.empty([0,8])
        
        # The tile is used when taking an expectation over actions later.
        small_tile = np.identity(self.nA)
        self.actions_tile = np.zeros([self.nA ** (self.n_cyclists - 1), 0])
        for i in range(self.n_cyclists - 1):
            n_reps = self.nA ** (self.n_cyclists - 2 - i)
            n_tiles = (self.nA) ** i
            rep_tile = np.repeat(small_tile, n_reps, axis = 0)
            tile_tile = np.tile(rep_tile, (n_tiles, 1))
            self.actions_tile = np.append(self.actions_tile, tile_tile, axis = 1)
        self.model = self.build_policy_network()
        
    def build_policy_network(self):
        '''
        Critic network is a neural network of arbitrary length and width.
        '''
        model = Sequential()
        
        in_dim = self.nx + (self.n_cyclists - 1) * self.nA
        model.add(Dense(self.layers[0], input_dim = in_dim, activation = 'relu', kernel_initializer = glorot_normal(seed=self.seed)))
        
        for lay in self.layers[1:]:
            model.add(Dense(lay, activation='relu', kernel_initializer = glorot_normal(seed=self.seed)))

        model.add(Dense(1, activation='linear', kernel_initializer = glorot_normal(seed=self.seed)))
        model.compile(optimizer=Adam(lr = self.lr), loss = mean_squared_error)
        return model
        
    def predict(self, states):
        '''
        Returns estimated value of states
        '''
        return np.squeeze(self.model.predict(states))

        
    def store_dead(self, dead_state, dead_actions, dead_probs, index, actor, episode_counter):
        '''
        Store transtions involving dead states.
        '''
        self.state_memory[self.counter] = dead_state
        actions = dead_actions.pop(index)
        probs = dead_probs.pop(index)
        other_actions = dead_actions
        other_probs = dead_probs
        for i in range(self.n_cyclists - 1):
            self.other_action_memory[i,self.counter] = other_actions[i]
            self.other_probs_memory[i,self.counter,:] = other_probs[i]
        self.loc_end.append(self.counter)
        # Set these to 1 to avoid erros when computing logs
        self.prob_memory[self.counter,:] = 1
        self.counter += 1
        if self.counter >= self.batch_size:
            # Does not matter what the next states are. The label will be zero anyways and there will be no advantage.
            self.learn(actor, episode_counter)
        
    def store_transition_1(self, state, entropy):
        '''
        Store the first transition after the environment is reset.
        '''
        self.state_memory[self.counter] = state
        self.entropy_memory[self.counter] = entropy
            
        
    def store_actions(self, actions, probss, index):
        '''
        Store the first set of actions.
        '''
        self.action_memory[self.counter] = actions.pop(index)
        self.prob_memory[self.counter] = probss.pop(index)
        other_actions = actions
        other_probs = probss
        for i in range(self.n_cyclists - 1):
            self.other_action_memory[i,self.counter] = other_actions[i]
            self.other_probs_memory[i,self.counter,:] = other_probs[i]

    def store_transition_loop(self, state, actions, index, entropy, probss, reward, actor, episode_counter):
        '''
        Store a set of transitions in the main loop.
        '''
        action = actions.pop(index)
        other_actions = actions
        probs = probss.pop(index)
        other_probs = probss
        
        self.reward_memory[self.counter] = reward
        self.counter += 1
        if self.counter >= self.batch_size:
            self.state_memory[self.counter] = state
            for i in range(self.n_cyclists - 1):
                self.other_action_memory[i,self.counter] = other_actions[i]
                self.other_probs_memory[i,self.counter,:] = other_probs[i]
            self.learn(actor, episode_counter)

        self.state_memory[self.counter] = state
        self.prob_memory[self.counter] = probs
        self.entropy_memory[self.counter] = entropy
        self.action_memory[self.counter] = action
        for i in range(self.n_cyclists - 1):
            self.other_action_memory[i,self.counter] = other_actions[i]
            self.other_probs_memory[i,self.counter,:] = other_probs[i]
        
        
    def store_final_reward(self, reward):
        '''
        Used to store a final sparse reward e.g. the group reward.
        '''
        self.reward_memory[self.counter - 1] = reward
    
    def get_lam(self, episode):
        '''
        Get the exploitation parameter value from a pre-defined profile. 
        '''
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
        '''
        Calculate the standard deviations for the advantages. This is used to 
        normalise the reward and entropy advantages so that the lambda parameter
        can effectively weight them.
        '''
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
        '''
        Updates the critic and actor. 
        '''
        rewards = self.reward_memory[:self.action_memory.size]
        entropies = self.entropy_memory[:self.action_memory.size]

        states_rep = np.repeat(self.state_memory,self.nA ** (self.n_cyclists - 1),axis = 0)
        actions_tile = np.tile(self.actions_tile, (self.batch_size + 1, 1))

        state_actions = np.append(states_rep, actions_tile, axis = 1)
        action_indexes = np.zeros([self.batch_size,])
        for i in range(self.n_cyclists - 1):
            action_indexes += self.other_action_memory[i,:-1] * self.nA ** (self.n_cyclists - 2 - i)
        true_indexes = (self.nA ** (self.n_cyclists - 1) * np.arange(self.batch_size) + action_indexes).astype(int)
        
        combined_probs = np.ones([(self.batch_size + 1),self.nA ** (self.n_cyclists - 1)])
        for i in range(self.n_cyclists - 1):
            n_reps = self.nA ** (self.n_cyclists - 2 - i)
            n_tiles = (self.nA) ** i
            rep = np.repeat(self.other_probs_memory[i,:,:], n_reps, axis = -1)
            rep = np.tile(rep, (1,n_tiles))
            combined_probs *= rep
            
        where_significant = np.where(combined_probs.ravel() > self.significant_prob)
        Qs = np.zeros([state_actions.shape[0],]) 
        Qs[where_significant] = self.predict(state_actions[where_significant])      
        Vs = np.einsum('ij,ij->i',Qs.reshape([-1,self.nA ** (self.n_cyclists - 1)]), combined_probs)[1:]
        # print(time() - tic)
        Q_labels = rewards + self.gamma * Vs

        # Ensure places corresponding to ends of episodes get 0 value
        Q_labels[self.loc_end] = 0
        # Q_labels[self.loc_end - 1] = rewards[self.loc_end - 1]
        self.model.train_on_batch(state_actions[true_indexes,:], Q_labels)

        Qs = np.zeros([state_actions.shape[0],]) 
        Qs[where_significant] = self.predict(state_actions[where_significant])      
        Vs = np.einsum('ij,ij->i',Qs.reshape([-1,self.nA ** (self.n_cyclists - 1)]), combined_probs)[1:]

        reward_advantages = rewards + self.gamma * Vs - Qs[true_indexes]
        entropy_advantages = entropies - self.maximum_entropy
        reward_stds, entropy_stds = self.get_stds([reward_advantages, entropy_advantages])
        reward_advantages /= (reward_stds + 1)
        entropy_advantages /= (entropy_stds + 0.05)
        lambda_reward = self.get_lam(episode_counter)
        advantages = reward_advantages * lambda_reward + entropy_advantages * (1 - lambda_reward)
        advantages[self.loc_end] = 0
        
        # advantages = np.clip(advantages, None, 20)
        actor.learn(self.action_memory, self.state_memory[:-1,:], advantages, lambda_reward, self.prob_memory)
        # self.norms = np.append(self.norms, norms.reshape([1,-1]), axis = 0)
        self.reset_arrays()

    def update_dividor_subtractor(self, dividors, subtractors, actor): 
        '''
        Update the policy and value functions to account for a change in the
        normalisation factors. Without this step, adjusting the normalisation
        factors at the ends of episodes effectively changes the functions.
        '''

        delta_dividors = dividors[1] / dividors[0]
        delta_dividors = np.append(delta_dividors, np.ones(self.nA * (self.n_cyclists - 1)))
        W = self.model.layers[0].weights[0] * delta_dividors[:,None]
        self.model.layers[0].weights[0].assign(W)
        delta_subs = np.append((subtractors[1] - subtractors[0]) / dividors[1], np.zeros(self.nA * (self.n_cyclists - 1)))
        b = self.model.layers[0].weights[1] + tf.tensordot(W, delta_subs.astype('float32'), axes = (0,0))
        self.model.layers[0].weights[1].assign(b)
                
        actor.policy[1:] *= delta_dividors[:self.nx,None]
        actor.policy[0] += (np.dot(delta_subs[:self.nx],actor.policy[1:]))
        
        
    def reset_arrays(self):
        self.reward_memory = np.zeros([self.batch_size,]).astype(float)
        self.distance_memory = np.zeros([self.batch_size,]).astype(float)
        self.entropy_memory = np.empty([self.batch_size,]).astype(float)
        self.state_memory = np.zeros([self.batch_size + 1,self.nx])
        self.action_memory = np.empty([self.batch_size,]).astype(int)
        self.other_action_memory = np.zeros([self.n_cyclists - 1, self.batch_size + 1,]).astype(int)
        self.other_probs_memory = np.zeros([self.n_cyclists - 1, self.batch_size + 1,self.nA]).astype(float)
        self.prob_memory = np.empty([self.batch_size,self.nA]).astype(float)
        self.loc_end = []
        self.counter = 0
