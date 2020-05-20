import numpy as np


# Our policy that maps state to action parameterized by w
def policy(state,w):
	z = state.dot(w)
	exp = np.exp(z - np.max(z))
	return (exp/np.sum(exp))

# Vectorized softmax Jacobian
def softmax_grad(probs, action, state, model):
    probs = probs.reshape(1,-1)
    probs[0,action] -= 1
    probs = - probs
    return state.reshape(-1,1) @ probs

class State_normaliser:
    def __init__(self, env_params):
        self.state_subtractor = np.array([env_params['race_length'] / 3, env_params['vel_max'] / 2, 50,1,env_params['vel_max'] / 2])
        self.state_dividor = np.array([env_params['race_length'], env_params['vel_max'], 100,env_params['race_length'] / 4,env_params['vel_max']])

    def normalise_state(self, state):
        return np.append((state - self.state_subtractor) / self.state_dividor, 1)

# def normalise_state(state, env_params):
    
#     # out_state = np.ones([state.size + 1])
#     # out_state[:state.size] = state 
#     # out_state[:state.size] -= 
#     # out_state[:state.size] /= 
#     return np.append((state - state_subtractor) / state_dividor, 1, axis = 1)

def get_entropy(probs):
    return - np.sum(probs * np.log(probs + 1e-20)) 


