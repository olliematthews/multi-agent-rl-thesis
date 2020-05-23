import numpy as np


# Our policy that maps state to action parameterized by w
def policy(state,w):
    '''
    Get the probabilities for taking each action.

    Parameters
    ----------
    state : np.array
    w : np.array
        The weight matrix for the policy.

    Returns
    -------
    probs : np.array

    '''
    z = state.dot(w)
    exp = np.exp(z - np.max(z))
    return (exp/np.sum(exp))

def softmax_grad(probs, action, state):
    '''
    Evaluates the softmax gradient.

    Parameters
    ----------
    probs : np.array
        Array of probabilities for each action.
    actions : np.array
        Array of actions taken at each step.
    states : np.array
        Array of input states at each step.

    Returns
    -------
    gradients : np.array
        The gradients wrt the policy weight matrix.

    '''
    probs = probs.reshape(1,-1)
    probs[0,action] -= 1
    probs = - probs
    return state.reshape(-1,1) @ probs

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
    return - np.sum(probs * np.log(probs + 1e-20)) 

class State_normaliser:
    '''
    Used to ensure inputs to networks are normalised.
    '''
    def __init__(self, env_params):
        self.state_subtractor = np.array([env_params['race_length'] / 3, env_params['vel_max'] / 2, 50,1,env_params['vel_max'] / 2])
        self.state_dividor = np.array([env_params['race_length'], env_params['vel_max'], 100,env_params['race_length'] / 4,env_params['vel_max']])

    def normalise_state(self, state):
        return np.append((state - self.state_subtractor) / self.state_dividor, 1)




