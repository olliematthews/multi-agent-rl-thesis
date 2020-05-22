'''
Runs learning.
'''

import numpy as np
import matplotlib.pyplot as plt
from environment import Cyclist
import pickle
import copy
from learning_functions import policy, critic, critic_grad, actor_grad
from learning_functions import normalise_state, get_entropy, estimate_critic
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import comb

# Simulation Parameters
sim_params = {
    'n_episodes' : 4000,
    'n_seeds' : 10,
    'print_rewards' : True
    }

env_params = {
    'cD' : 0.2,
    'vel_max' : 4,
    'vel_min' : 0,
    'race_length' : 250,
    'time_limit' : 200
    }


# Hyperparameters
hyper_params = {
    'learning_rate_actor' : 0.005,
    'learning_rate_critic' : 0.01,
    'momentum_critic': 0.9,
    'gamma' : 0.99,
    # Epsilon starts at 0.98. It decays to epsilon_final in decay_episodes episodes
    'epsilon_final' : 0.01,
    'epsilon_0' : 0.98,
    'decay_episodes' : 1000,
    'exploration' : False,
    'p_order' : 2,
    'use_std' : False,
    'alpha_std' : 0.01,
    'TD_batch' : [1,5,10,20,50,100,500]
    }


env_params = {
        'cD' : 0.2,
        'vel_max' : 4,
        'vel_min' : 0,
        'race_length' : 400,
        'time_limit' : 200
        }



# Initialisations
n_iterations = len(hyper_params['TD_batch'])
labels = []
Seed_rewards = np.empty([n_iterations,sim_params['n_seeds'],sim_params['n_episodes']])
Seed_entropies = np.empty([n_iterations,sim_params['n_seeds'],sim_params['n_episodes']])
Seed_mean_velocities = np.empty([n_iterations,sim_params['n_seeds'],sim_params['n_episodes']])
loss_norm = 0
high_score = 0
seed_losses = []
for i in range(sim_params['n_seeds']):
    seed_losses.append([])
iterator_rewards = []
iterator_losses = []
iterator_critics = []
abss = []


for iterator in range(n_iterations):
    # Hyperparameters
    alpha_p_change = 0.001
    TD_batch = hyper_params['TD_batch'][iterator]
    decay_episodes = hyper_params['decay_episodes']
    learning_rate_actor = hyper_params['learning_rate_actor']
    learning_rate_critic = hyper_params['learning_rate_critic']
    gamma = hyper_params['gamma']
    epsilon_0 = hyper_params['epsilon_0']
    epsilon_final = hyper_params['epsilon_final']
    epsilon_decay = (epsilon_0 / epsilon_final) ** (-1 / decay_episodes)
    p_order = hyper_params['p_order']
    poly = PolynomialFeatures(p_order)

    seed_rewards = np.empty([sim_params['n_seeds'], sim_params['n_episodes']])

    policy_changes = []
    exploration = hyper_params['exploration']
    seed_critics = [None] * sim_params['n_seeds']
    for seed in range(sim_params['n_seeds']):
        # Create enrironment and seed numpy
        cyclist = Cyclist(env_params)
        nA = cyclist.action_space
        nx = cyclist.state_space + 1
        np.random.seed(seed)
        Average_rewards = np.empty([n_iterations,sim_params['n_episodes']])
        Average_entropies = np.empty([n_iterations,sim_params['n_episodes']])
        Std_rewards = np.zeros_like(Average_rewards)
        Std_entropies = np.zeros_like(Average_entropies)

        rolling_std = 0

        # Init weight
        w_policy = np.random.normal(0,0.1,(nx,nA))
        policies = copy.copy(w_policy)
        policies = policies[None,:,:]
        delta_w_policy = np.zeros_like(w_policy)
        w_critic = estimate_critic(w_policy, 1, gamma, env_params, seed, poly)

        w_critic = np.random.normal(0,0.1,w_critic.shape)
        print(w_critic)
        w_critics = copy.copy(w_critic)
        epsilon = epsilon_0
    
        # Keep stats for final print of graph
        episode_rewards = []
        mean_velocities = []
        entropies = []
        policy_change = 10
        states = np.zeros([TD_batch + 1,nx])
        advantages = np.zeros([TD_batch,1])
        actions = np.zeros([TD_batch,nA])
        probs = np.empty([TD_batch, nA])
        rewards = np.empty([TD_batch,1])
        loc_exp = []
        losses = []
        batch_step = 0
        incremented = False
        episode_advantages = np.zeros([0,1])
        episode_states = np.zeros([0,nx])
        episode_actions = np.zeros([0,nA])
        stds = []
        # Main loop 
        for e in range(sim_params['n_episodes']):
            state_rewards = np.empty([0,1])
            state = np.squeeze(np.append(normalise_state(cyclist.reset(pose = np.random.randint(-100,101)),env_params),1))
            critic_rewards = np.empty([0,1])
            score = 0
            step = 0
            entropy = 0
            mean_velocity = 0
            while True:
        		# Choose your action
                prob = np.array(policy(state,w_policy))
                # With probability epsilon, draw from a uniform distribution
                if np.random.rand() < epsilon and exploration:
                    action = np.random.choice(nA, p = [1 / nA] * nA)
                    # Note which steps were random
                    loc_exp.append(step)
                # Else draw from the probability distribution
                else:
                    action = np.random.choice(nA,p = np.squeeze(prob))
                
                
                # Save your state and action
                states[batch_step,:] = state
                actions[batch_step,action] = 1
                probs[batch_step,:] = prob
                
                # Take the action
                state,reward,done = cyclist.step(action - 1,env_params)
                
                # Update the state
                mean_velocity = mean_velocity * (1 - 1 / (1 + step)) + 1 / (1 + step) * state[1]
                state = np.append(normalise_state(state,env_params),1)
                
                # Save the score
                rewards[batch_step] = reward
                score+=reward
                step += 1
                
                
                if done:
                    # Make sure you gracefully treat starts of new episodes -
                    # treat action row of all 0s as a flag for this
                    batch_step += 1
                    states[batch_step,:] = state
                    
                batch_step += 1
                # If we hit the batch size, update parameters!
                if batch_step >= TD_batch:
                    # Find any places where the episode ended
                    loc_end = np.where(np.sum(actions,axis = 1) == 0)

                    states[-1,:] = state
                    
                    # Calculate labels:
                    values = critic(states[:,:-1],w_critic, poly)
                    value_labels =  rewards + gamma * values[1:]
                    
                    # Ensure places corresponding to ends of episodes get 0 value
                    value_labels[loc_end] = 0
                    # Make sure not to learn from off-policy exploration actions
                    safe_values = np.delete(values[:-1], loc_exp, axis = 0)
                    safe_states = np.delete(states[:-1], loc_exp, axis = 0)
                    safe_value_labels = np.delete(value_labels, loc_exp, axis = 0)
                    
                    # Update the critic
                    loss_norm = (np.linalg.norm(values[:-1] - value_labels)) ** 2
                    w_critic -= learning_rate_critic * critic_grad(safe_values, safe_value_labels, safe_states[:,:-1], poly)
                    # Calculate the advantage
                    values = critic(states[:,:-1],w_critic, poly)
                    advantages = rewards + gamma * values[1:] - values[:-1]
                    
                    # Set 0 advantage at the end of episodes
                    advantages[loc_end,:] = 0

                    episode_advantages = np.append(episode_advantages, advantages, axis = 0)
                    if hyper_params['use_std']:
                        advantages /= (rolling_std + 1)
                    policy_changes.append(learning_rate_actor * actor_grad(probs, actions, states[:-1], advantages))
                    abss.append(np.sqrt(np.mean(advantages ** 2, axis = 0)))
                    w_policy += learning_rate_actor * actor_grad(probs, actions, states[:-1], advantages)
                    policies = np.append(policies, w_policy[None,:,:], axis = 0)
                    actions = np.zeros([TD_batch,nA])
                    batch_step = 0
                    

                if done:
                    break             
                
            if hyper_params['use_std']:
                if batch_step > 0:
                    vals = critic(states[:batch_step,:-1],w_critic, poly)
                    ads = rewards[: (batch_step - 1),:] + gamma * vals[1:] - vals[:-1]
                    episode_advantages = np.append(episode_advantages, ads)
                    
                rolling_std = np.sqrt(rolling_std ** 2 * (1 - hyper_params['alpha_std']) + hyper_params['alpha_std'] * (np.mean(episode_advantages ** 2)))
                episode_advantages = np.zeros([0,1])

            # Save the best parameters
            if score >= high_score:
                high_score = score
                w_best = w_policy
                w_critic_best = w_critic.reshape(-1,1)
            if sim_params['print_rewards']:
                print(f'Seed {seed}, Episode {e}: Reward of {score}')
                print()
            episode_rewards.append(score)
            losses.append(loss_norm)
            stds.append(rolling_std)
            w_critics = np.append(w_critics, w_critic, axis = 1)
        seed_losses[seed] = losses   

        plt.plot(episode_rewards, alpha = 0.3)
        plt.plot(gaussian_filter(episode_rewards,20), label = f'Seed is {seed}')
        plt.show
        seed_rewards[seed,:] = episode_rewards
        seed_critics[seed] = w_critic
        print(w_critic)
    iterator_rewards.append(seed_rewards)
    iterator_losses.append(seed_losses)
    iterator_critics.append(seed_critics)
    
params = {'Env': env_params, 'Hyp' : hyper_params, 'Sim' : sim_params}
plt.legend()
plt.show()

# Save results
pickle.dump([w_best, iterator_rewards, iterator_losses , iterator_critics, params, policies], open('batches.p','wb'))
