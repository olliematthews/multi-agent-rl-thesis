"""The coordinator, which supervises learning for a number of agents.
"""

import copy
from itertools import count, cycle

import numpy as np


class Coordinator:
    """
    This is the 'coach'. It does all of the hyperparameter value optimising.
    It runs simulations in parallel with different hyperparameter values, and
    measures the performance of each based on a maximum entropy performance
    hueristic. It cycles through each of the available hyperparameter values,
    testing them one by one.
    """

    def __init__(self, coordinator_args):
        self.queue = coordinator_args["queue"]
        self.pipes = coordinator_args["pipes"]
        self.n_workers = coordinator_args["n_workers"]
        self.hyper_params = coordinator_args["hyper_params"]
        self.n_episodes = coordinator_args["n_episodes"]
        self.optimisable_params = cycle(self.hyper_params["param_changes"])

        self.current_params = self.hyper_params["param_inits"].copy()
        self.param_changes = self.hyper_params["param_changes"].copy()

        self.optimise_over = next(self.optimisable_params)
        self.param = self.hyper_params["param_inits"][self.optimise_over]

        self.search_count = 0

    def get_searcher_params(self):
        searcher_params = self.current_params.copy()
        searcher_params[self.optimise_over] += self.param_changes[self.optimise_over]
        return searcher_params

    def _initial_wait_for_workers(self):
        # Wait for all processes to be ready
        readies = []
        while len(readies) < self.n_workers:
            readies.append(self.queue.get())
            print(f"Received response {len(readies)}")
        print("All processes ready - begin!")

    def _await_workers(self):
        # Wait for each worker and get their scores
        scores = []
        while len(scores) < self.n_workers:
            scores.append(self.queue.get())
            print(f"Received response {len(scores)}")
        return scores

    def _scores_to_performances(self, scores):
        # We evaluate the performance by multiplying the average reward of the
        # agents by their entropy. The entropy is weighted less as training goes
        # on, to encourage more exploitation - this is done by the "power" parameter

        # power ramps from 0.5 to -0 over the course of the simulation
        power = 0.5 * (
            1 - (self.eps - self.hyper_params["window_size"] / 2) / self.n_episodes
        )

        return [s[1] * s[2] ** power for s in scores]

    def _update_param_changes(self, param_change):
        if param_change == 0:
            self.search_count = 0
            # Make sure the next search goes in the opposite direction at base magnitude
            self.param_changes[self.optimise_over] = (
                -np.sign(self.param_changes[self.optimise_over])
                * self.hyper_params["param_changes"][self.optimise_over]
            )
            # Optimise the next hyperparameter if there is no change
            print("No Change - Switching Hyperparameters")
            self.optimise_over = next(self.optimisable_params)

        elif self.search_count >= self.hyper_params["max_goes"]:
            self.search_count = 0
            # Optimise the next hyperparameter if you have been following the same direction for long
            print("Maximum line distance reached - Switching Hyperparameters")
            self.optimise_over = next(self.optimisable_params)

        else:
            # Else go in the same direction but twice as far
            self.param_changes[self.optimise_over] *= 2
            self.search_count += 1

    def run(self, best_workers, best_params):
        """Run the simulation.

        The coordinator starts two sets of simulations at each step with different
        hyperparameters. It does this to search through hyperparameter space
        for the best parameters to improve agent performance.

        best_workers is a list which will be appended to by the coordinator to
            keep track of the best worker index on each iteration
        best_params is a list of the parameter, and the value of that parameter
            which performed best in each simulation
        """
        # Wait for the workers to be ready
        self._initial_wait_for_workers()

        self.pipes[0].send(["First"])
        searcher_params = self.get_searcher_params()
        self.pipes[1].send(["Searcher", searcher_params])

        for coordinator_round in count():
            self.eps = coordinator_round * self.hyper_params["window_size"]

            scores = self._await_workers()
            performances = self._scores_to_performances(scores)

            # Get the best worker (with the highest performance)
            best_index = np.argmax(performances)
            best_worker = scores[best_index][0]

            best_workers.append(best_worker)

            # Stop when you have reached n_episodes
            if self.eps >= self.n_episodes:
                [p.send(["Done"]) for p in self.pipes]
                break
            else:
                # Ask the best worker what their parameter value was
                print("Sending Best!")
                self.pipes[best_worker].send(["Best", self.optimise_over])
                print("Waiting for response")
                best_param = self.pipes[best_worker].recv()
                print("Response Recieved - Param is " + str(best_param))

                param_change = best_param - self.current_params[self.optimise_over]

                # Set the current param value to the best parameter value
                self.current_params[self.optimise_over] = copy.copy(best_param)

                # Set up which params will be tested next
                self._update_param_changes(param_change)

                searcher_params = self.get_searcher_params()
                print("Searcher params are " + str(searcher_params))

                # Send the new parameters to the searcher
                self.pipes[1 - best_worker].send(["Searcher", searcher_params])

                best_params.append([self.optimise_over, best_param])


def run_coordinator(coordinator_args, best_workers, best_params):
    coordinator = Coordinator(coordinator_args)
    coordinator.run(best_workers, best_params)
