"The state normaliser model"

import pickle

import numpy as np


class StateNormaliser:
    """The state normaliser model.

    This is used to normalise states before feeding them into each model.
    """

    def __init__(self, env, hyper_params):
        self.nx = env.state_space
        self.state_subtractor = [
            env.env_params["race_length"] / 2,
            env.env_params["vel_max"] / 2,
            50.0,
        ]
        self.state_dividor = [
            env.env_params["race_length"] / 2,
            env.env_params["vel_max"] / 2,
            50.0,
        ]
        # Other pose normaliser
        self.state_subtractor.extend([0.0] * (env.n_cyclists - 1))
        self.state_dividor.extend([50.0] * (env.n_cyclists - 1))

        # Other vel normaliser
        self.state_subtractor.extend([0.0] * (env.n_cyclists - 1))
        self.state_dividor.extend([1.0] * (env.n_cyclists - 1))

        # Other energy normaliser
        self.state_subtractor.extend([50.0] * (env.n_cyclists - 1))
        self.state_dividor.extend([50.0] * (env.n_cyclists - 1))

        self.state_subtractor.append(env.env_params["time_limit"] / 2)
        self.state_dividor.append(env.env_params["time_limit"] / 2)
        self.state_subtractor = np.array(self.state_subtractor)
        self.state_dividor = np.array(self.state_dividor)

        self.reset_arrays()
        self.alpha = hyper_params["state_normaliser_alpha"]

    def update_normalisation(self):
        """
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

        """
        state_stds = np.sqrt(self.state_mean_squareds - self.state_means**2)
        # Rounding errors can cause negative sqrts. Set these to 0.
        state_stds[np.isnan(state_stds)] = 0

        old_state_subtractor = self.state_subtractor.copy()
        self.state_subtractor = (
            self.alpha * self.state_means + (1 - self.alpha) * self.state_subtractor
        )
        old_state_dividor = self.state_dividor.copy()
        self.state_dividor = (
            self.alpha * state_stds + (1 - self.alpha) * self.state_dividor
        )
        self.reset_arrays()
        return [old_state_dividor, self.state_dividor], [
            old_state_subtractor,
            self.state_subtractor,
        ]

    def normalise_state(self, state):
        """
        We normalise the state and also add the state to the normaliser's
        memory
        """
        alpha = 1 / self.count
        self.state_means = (alpha) * state["state"] + (1 - alpha) * self.state_means
        self.state_mean_squareds = (alpha) * state["state"] ** 2 + (
            1 - alpha
        ) * self.state_mean_squareds
        self.count += 1
        state["state"] = (state["state"] - self.state_subtractor) / self.state_dividor
        return state

    def normalise_batch(self, states):
        """
        Normalise an entire batch
        """
        return (states - self.state_subtractor) / self.state_dividor

    def reset_arrays(self):
        self.state_means = np.array(
            [
                self.nx,
            ]
        )
        self.state_mean_squareds = np.array(
            [
                self.nx,
            ]
        )
        self.count = 1

    def save_model(self, seed):
        self.reset_arrays()
        pickle.dump(
            [self.state_subtractor, self.state_dividor],
            open("normaliser_weights_" + str(seed) + ".p", "wb"),
        )

    def load_model(self, seed):
        self.reset_arrays()
        self.state_subtractor, self.state_dividor = pickle.load(
            open("normaliser_weights_" + str(seed) + ".p", "rb")
        )
