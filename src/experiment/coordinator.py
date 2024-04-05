"""The coordinator, which supervises learning for a number of agents.
"""

import copy
from time import sleep

import numpy as np


def coordinator(coordinator_args, best_workers, best_params):
    """
    This is the 'coach'. It does all of the hyperparameter value optimising.
    It runs simulations in parallel with different hyperparameter values, and
    measures the performance of each based on a maximum entropy performance
    hueristic. It cycles through each of the available hyperparameter values,
    testing them one by one.
    """
    queue = coordinator_args["queue"]
    pipes = coordinator_args["pipes"]
    n_workers = coordinator_args["n_workers"]
    hyper_params = coordinator_args["hyper_params"]
    optimisable_params = list(hyper_params["param_changes"].keys())
    optimise_index = 0
    optimise_over = optimisable_params[optimise_index]
    current_params = hyper_params["param_inits"].copy()
    eps = 0

    param_changes = hyper_params["param_changes"].copy()
    param = hyper_params["param_inits"][optimise_over]

    # Wait for all processes to be ready, then start!
    readies = []
    while len(readies) < n_workers:
        readies.append(queue.get())
        boi = len(readies)
        print(f"Received response {boi}")

    sleep(2)
    print("All processes ready - begin!")
    pipes[0].send(["First"])
    new_params = current_params.copy()
    new_params[optimise_over] = (
        current_params[optimise_over] + param_changes[optimise_over]
    )
    pipes[1].send(["Searcher", new_params])

    search_count = 0

    while 1:
        eps += coordinator_args["hyper_params"]["window_size"]
        power = 0.5 * (
            1
            - (eps - coordinator_args["hyper_params"]["window_size"] / 2)
            / coordinator_args["n_episodes"]
        )
        scores = []
        while len(scores) < n_workers:
            scores.append(queue.get())
            boi = len(scores)
            print(f"Received response {boi}")

        performances = [s[1] * s[2] ** power for s in scores]

        best_index = np.argmax(performances)
        best_worker = scores[best_index][0]
        best_workers.append(best_worker)
        if eps >= coordinator_args["n_episodes"]:
            [p.send(["Done"]) for p in pipes]
            break
        else:
            print("Sending Best!")
            pipes[best_worker].send(["Best", optimise_over])
            print("Waiting for response")
            param = pipes[best_worker].recv()
            print("Response Recieved - Param is " + str(param))
            # old_param_changes = param_changes.copy()

            param_change = param - current_params[optimise_over]

            # Save the parameter in case you switch
            current_params[optimise_over] = copy.copy(param)

            if param_change == 0:
                search_count = 0
                # Make sure the next search goes in the opposite direction at base magnitude
                param_changes[optimise_over] = (
                    -np.sign(param_changes[optimise_over])
                    * hyper_params["param_changes"][optimise_over]
                )
                # Optimise the next hyperparameter if there is no change
                print("No Change - Switching Hyperparameters")
                optimise_index += 1
                if optimise_index >= len(optimisable_params):
                    optimise_index = 0
                optimise_over = optimisable_params[optimise_index]

            elif search_count >= hyper_params["max_goes"]:
                search_count = 0
                # Optimise the next hyperparameter if you have been following the same direction for long
                print("Maximum line distance reached - Switching Hyperparameters")
                optimise_index += 1
                if optimise_index >= len(optimisable_params):
                    optimise_index = 0
                optimise_over = optimisable_params[optimise_index]

            else:
                # Else go in the same direction but twice as far
                param_changes[optimise_over] *= 2
                search_count += 1

            # print('Response Recieved - Param: ' + optimise_over + ' is '+ str(param))
            new_params = current_params.copy()
            new_params[optimise_over] = (
                current_params[optimise_over] + param_changes[optimise_over]
            )
            print("New params are " + str(new_params))

            # Send the new parameters to the searcher
            pipes[1 - best_worker].send(["Searcher", new_params])

            best_params.append([optimise_over, param])
