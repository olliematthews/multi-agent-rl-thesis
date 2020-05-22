'''
File with theh functions required by the 'run' file.
'''
import numpy as np
from environment import Cyclist
from sklearn.preprocessing import PolynomialFeatures


# Our policy that maps state to action parameterized by w
def policy(state,w):
    '''
    Returns probabilities for taking each action, based on a logistic
    regression

    Parameters
    ----------
    state : np.array
        The cyclist's state.
    w : np.array
        The weights for the policy.

    Returns
    -------
    probs : np.array
        The probabilities for taking each action.

    '''
    z = state.dot(w)
    exp = np.exp(z - np.max(z))
    return (exp/np.sum(exp))

def feature_map(states, poly):
    '''
    Maps the states to a higher dimension according to the feature map.

    Parameters
    ----------
    states : np.array
        States.
    poly : PolynomialFeatures
        The polynomial feature map.

    Returns
    -------
    poly_states : np.array
        A higher dimension representation of the states.

    '''
    return poly.fit_transform(states)
    
def critic(states,w, poly):
    '''
    Evaluate the value of some states

    Parameters
    ----------
    states : np.array
        States.
    w : np.array
        Weight matrix.
    poly : PolynomialFeatures
        Polynomial feature map.

    Returns
    -------
    Value : float
        The value of the state.

    '''
    return feature_map(states, poly) @ w

def critic_grad(values, value_labels, x, poly):
    '''
    Evaluates the gradient with a mse loss function wrt the critic's weights.

    Parameters
    ----------
    values : np.array
        The estimated state values.
    value_labels : np.array
        The labels for the values.
    x : np.array
        The inputs (states).
    poly : PolynomialFeatures
        Polynomial feature map.

    Returns
    -------
    Gradients : np.array
        The gradients wrt the weight matrix.

    '''
    return feature_map(x, poly).T @ (2 * (values - value_labels))

# Vectorized softmax Jacobian
def actor_grad(probs, actions, states, advantages):
    '''
    Evaluates the policy gradient.

    Parameters
    ----------
    probs : np.array
        Array of probabilities for each action.
    actions : np.array
        Array of actions taken at each step.
    states : np.array
        Array of input states at each step.
    advantages : np.array
        Array of advantages.

    Returns
    -------
    gradients : np.array
        The gradients wrt the policy weight matrix.

    '''
    dLdx = actions - probs
    return states.T @ (dLdx * advantages)

def normalise_state(state, env_params):
    '''
    Normalises the states so that they lie roughly between -1 and 1

    Parameters
    ----------
    state : np.array
        The cyclist's state.
    env_params : dict
        Environment parameters.

    Returns
    -------
    normalised state : np.array
        The cyclsits state, normalised.

    '''
    range = np.array([env_params['race_length'], env_params['vel_max'], 100, env_params['time_limit']])
    return (2 * state - range) / range

def get_entropy(probs):
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
    return - np.sum(probs * np.log(probs)) 

def update_critic_ls(F,r):
    '''
    Update the critic to minimise the least squared loss.

    Parameters
    ----------
    F : np.array
        Input states.
    r : np.array
        Rewards.

    Returns
    -------
    w : np.array
        The policy which minimises the least squared loss over the examples.

    '''
    return np.linalg.inv(F.T @ F) @ (F.T @ r)

def estimate_critic(w_policy, batch_size, gamma, env_params, seed, poly):
    '''
    Here we run some episodes with the initial policy to initialise the critic.

    Parameters
    ----------
    w_policy : np.array
        Policy weight matrix.
    batch_size : int
        Number of episodes to run.
    gamma : float
        Discount rate.
    env_params : dict
        Environment parameters.
    seed : int
        Random seed.
    poly : PolynomialFeatures
        Polynomial feature map.

    Returns
    -------
    w : np.array
        The policy which minimises the least squared loss over the examples.

    '''
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

