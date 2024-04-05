import numpy as np
import tensorflow as tf
import pickle

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import random


class Actor:
    def __init__(self, environment, hyper_params, seed):
        self.lr = hyper_params["lr_actor"]
        self.nx = environment.state_space
        self.nA = len(environment.action_space)
        self.action_space = environment.action_space
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.policy = self.build_policy_network()

    def build_policy_network(self):
        w_size = self.nx + 1
        w = np.random.normal(0, 0.1, (w_size, self.nA))
        return w

    def predict(self, state):
        """
        Get the probability of taking each action
        """
        z = state.dot(self.policy)
        exp = np.exp(z - np.max(z))
        return exp / np.sum(exp)

    def choose_action(self, state):
        """
        Choose you action according to a seeded generator and the policy.
        """
        poly_state = np.append(np.array([1]), state)
        probs = self.predict(poly_state)
        entropy = self.get_entropy(probs)
        action = random.choices(self.action_space, probs)[0]
        return probs, action, entropy

    def get_entropy(self, probs):
        """
        Calculate the Shannon entropy of a set of probabilities
        """
        return -np.sum(probs * np.log(probs + 1e-4))

    def learn(self, actions, states, advantages, lambda_explore, probs):
        """
        Update the policy. We introduce entropy regularisation, with lambda
        explore indicating the weight towards exploitation.
        """
        poly_states = np.append(np.ones([states.shape[0], 1]), states, axis=1)
        x = np.identity(probs.shape[1])[None, :, :] - probs[:, None, :]
        # np.ones([probs.shape[1],1,]) @ probs.reshape([1,-1])
        dLdx = x[np.arange(x.shape[0]), actions.astype(int), :]
        log_grad = np.dot(poly_states.T, (dLdx * advantages[:, None]))

        log_probs = (
            probs * (1 + np.log(probs + 1e-6)) * (1 - (advantages == 0))[:, None]
        )
        entropy_grad = -np.einsum("ij,ikl,ik->jl", poly_states, x, log_probs)

        self.policy += self.lr * (log_grad + (1 - lambda_explore) * entropy_grad)

    def save_model(self, seed, number):
        pickle.dump(
            self.policy,
            open("actor_weights_" + str(seed) + "_" + str(number) + ".p", "wb"),
        )

    def load_model(self, seed, number):
        self.policy = pickle.load(
            open("actor_weights_" + str(seed) + "_" + str(number) + ".p", "rb")
        )
