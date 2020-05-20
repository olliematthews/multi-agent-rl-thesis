import numpy as np
from environment import Cyclist
from sklearn.preprocessing import PolynomialFeatures
# Our policy that maps state to action parameterized by w

def policy(state,w):
	z = state.dot(w)
	exp = np.exp(z - np.max(z))
	return (exp/np.sum(exp))

def feature_map(states, poly):
    return poly.fit_transform(states)
    
def critic(states,w, poly):
    return feature_map(states, poly) @ w

def critic_grad(values, value_labels, x, poly):
    return feature_map(x, poly).T @ (2 * (values - value_labels))

# Vectorized softmax Jacobian
def actor_grad(probs, actions, states, advantages):
    dLdx = actions - probs
    return states.T @ (dLdx * advantages)

# Returns states normalised between 
def normalise_state(state, env_params):
    range = np.array([env_params['race_length'], env_params['vel_max'], 100, env_params['time_limit']])
    return (2 * state - range) / range

def get_entropy(probs):
    return - np.sum(probs * np.log(probs)) 

def update_critic_ls(F,r):
    return np.linalg.inv(F.T @ F) @ (F.T @ r)

def estimate_critic(w_policy, batch_size, gamma, env_params, seed, poly):
    cyclist = Cyclist(env_params)
    nA = cyclist.action_space
    nx = cyclist.state_space + 1
    np.random.seed(seed)
    state_rewards = np.empty([0,1])
    states = np.empty([0,nx])
    # Keep track of game score to print
    for i in range(batch_size):
        state = np.squeeze(np.append(normalise_state(cyclist.reset(pose = np.random.randint(-100,101)),env_params),1))
        rewards = []
        step = 0
        while True:
    		# Sample from policy and take action in environment
            probs = np.array(policy(state,w_policy))
            
            # Else draw from the probability distribution
            action = np.random.choice(nA,p = np.squeeze(probs))
    
            next_state,reward,done = cyclist.step(action - 1,env_params)
            next_state = np.append(normalise_state(next_state,env_params),1)
            next_state = next_state[None,:]
    
    		# Compute gradient and save with reward in memory for our weight updates
            states = np.append(states,state.reshape(1,-1),axis = 0)
            rewards.append(reward)		
            step += 1
    		# Dont forget to update your old state to the new state
            state = next_state
            if done:
                break
                            
        rewards = rewards[:step]
        state_reward = np.zeros_like(rewards).astype(float)
        # Update parameters
        for i in range(len(rewards)):
    		# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
            state_reward[i] = sum([ r * (gamma ** t) for t,r in enumerate(rewards[i:])])
        
        state_rewards = np.append(state_rewards,state_reward.reshape(-1,1),axis = 0) 
        poly_states = feature_map(states[:,:-1], poly)
    return update_critic_ls(poly_states, np.array(state_rewards).reshape(-1,1))

