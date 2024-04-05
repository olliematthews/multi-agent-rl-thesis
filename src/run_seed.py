# -*- coding: utf-8 -*-
"""
Contains the functions required to learn. run_seed is called externally, and
will utilise the other functions.
"""
import numpy as np
from environment import Environment
import pickle
import random
from multiprocessing import Pool, Process, Pipe, Queue, Manager
from time import sleep
import copy


def run_seed(seed_params):
    """
    Run the actor critic algorithm with a U-function and with the coach frame-
    work. The coordinator is effectively the coach, the simulator is a worker.
    """
    n_workers = 2

    simulator_pool = Pool(processes=n_workers)
    pool_params = []
    pipes = []
    m = Manager()
    queue = m.Queue()
    lock = m.Lock()
    best_workers = m.list()
    best_params = m.list()
    param_history = m.list()
    pipe_conn_1, pipe_conn_2 = Pipe()
    worker_pipe = [pipe_conn_1, pipe_conn_2]
    for worker, worker_pipe_conn in enumerate(worker_pipe):
        dic = seed_params.copy()
        parent_conn, child_conn = Pipe()
        pipes.append(parent_conn)

        dic.update(
            {
                "worker_number": worker,
                "pipe": child_conn,
                "worker_pipe": worker_pipe_conn,
                "queue": queue,
                "lock": lock,
                "param_history": param_history,
            }
        )
        pool_params.append(dic)

    coordinator_args = {
        "pipes": pipes,
        "queue": queue,
        "n_workers": n_workers,
        "n_episodes": seed_params["Sim"]["n_episodes"],
        "hyper_params": seed_params["Hyp"],
        "state_size": 13,
    }

    coordinator_process = Process(
        target=coordinator, args=(coordinator_args, best_workers, best_params)
    )
    coordinator_process.start()
    output = simulator_pool.map(simulator, pool_params)
    coordinator_process.join()
    return [output, list(best_workers), list(best_params), list(param_history)]


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
    state_size = coordinator_args["state_size"]

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


def simulator(seed_params):
    """
    This is the simulator. It runs a certain number of episodes of learning,
    and feeds back the performance to the coach. At the start of a simulation,
    if it is a 'best' worker, it learns off the experiences of the other worker
    and then saves the model. The 'searcher' then loads this model, and the two
    restart in parallel.
    """
    from models import Actor, Critic, StateNormaliser

    # Create gym and seed numpy
    env_params = seed_params["Env"]
    sim_params = seed_params["Sim"]
    hyper_params = seed_params["Hyp"].copy()

    worker_number = seed_params["worker_number"]
    pipe = seed_params["pipe"]
    worker_pipe = seed_params["worker_pipe"]
    queue = seed_params["queue"]
    lock = seed_params["lock"]
    param_history = seed_params["param_history"]
    nA = 3
    seed = seed_params["seed"]
    # actions = [0] * sim_params['n_cyclists']
    #    entropy_kick = 0.4
    env = Environment(
        env_params,
        n_cyclists=sim_params["n_cyclists"],
        poses=[0] * sim_params["n_cyclists"],
    )
    # action_space = env.action_space
    np.random.seed(seed)
    random.seed(seed)

    # nA = len(env.action_space)
    # Init weights
    if sim_params["weights"] == "none":
        model = {"normaliser": StateNormaliser(env, hyper_params)}

        acs = []
        for i in range(sim_params["n_cyclists"]):
            acs.append(
                {
                    "actor": Actor(env, hyper_params, np.random.randint(1000)),
                    "critic": Critic(env, hyper_params, np.random.randint(1000)),
                }
            )
        model.update({"acs": acs})
    else:
        model = sim_params["weights"]
    # Keep stats for final print of graph
    episode_rewards = []
    episode_entropies = []
    episode_velocities = []
    episode_distances = []

    episode = 0
    queue.put("Ready")

    probs_arrays = [[], []]
    action_arrays = [[], []]
    entropy_arrays = [[], []]
    state_arrays = [[], []]
    reward_arrays = [[], []]
    distance_arrays = [[], []]
    loc_end_array = []
    while 1:

        response = pipe.recv()
        print("Response Recieved! Response was " + str(response))
        if response[0] == "Done":
            break

        elif response[0] == "Best":
            # Find out which param we are interested in, and send it over
            optimise_over = response[1]
            param = getattr(model["acs"][0]["critic"], optimise_over)
            print(f"Sending params - {param}")
            # Send the params, indicating you have saved your model.
            pipe.send(param)

            print("Waiting for data")
            # Wait for the tuples from the other worker
            arrays = worker_pipe.recv()

            for i in range(len(arrays)):
                arrays[i] = np.array(arrays[i])

            # print('Before learn:')
            # print(model['acs'][0]['critic'].model.layers[2].get_weights())
            # print(model['acs'][0]['actor'].policy[2])

            # Learn from the tuples
            print("Data Received - Normalising States")
            # for boi in arrays[3]:
            #     print('hi')
            #     print(type(boi))
            #     model['normaliser'].normalise_batch(boi)
            arrays[3] = [model["normaliser"].normalise_batch(x) for x in arrays[3]]

            print("Learning from states")
            for index, ac in enumerate(model["acs"]):
                ac["critic"].learn_off_policy(index, arrays, ac["actor"])

            # print('After learn:')
            # print(model['acs'][0]['critic'].model.layers[2].get_weights())
            # print(model['acs'][0]['actor'].policy[2])

            # Save your model
            print("Saving Model")
            lock.acquire()
            for i, ac in enumerate(model["acs"]):
                ac["actor"].save_model(seed, i)
                ac["critic"].save_model(seed, i)
            model["normaliser"].save_model(seed)
            lock.release()
            # Tell the other worker that the model is saved
            worker_pipe.send("Model Saved!")

        elif response[0] == "Searcher":
            # Send your tuples to the other worker straight away
            worker_pipe.send(
                [
                    probs_arrays,
                    action_arrays,
                    entropy_arrays,
                    state_arrays,
                    reward_arrays,
                    distance_arrays,
                    loc_end_array,
                ]
            )

            params = response[1]

            # Wait for other worker to save
            worker_pipe.recv()

            # Load the best model
            print("Loading Model")
            lock.acquire()
            for i, ac in enumerate(model["acs"]):
                ac["actor"].load_model(seed, i)
                ac["critic"].load_model(seed, i)
                [setattr(ac["critic"], key, value) for key, value in params.items()]
            model["normaliser"].load_model(seed)
            lock.release()

            print("Model Loaded - Off we go")

        elif response[0] == "First":
            # Recieve first garbage array
            worker_pipe.recv()
            # Save your initial model
            print("Saving Model")
            lock.acquire()
            for i, ac in enumerate(model["acs"]):
                ac["actor"].save_model(seed, i)
                ac["critic"].save_model(seed, i)
            model["normaliser"].save_model(seed)
            lock.release()

            worker_pipe.send("Model Saved!")

            print("First model starting")

        else:
            assert 1

        # Initiliase arrays for storing experiences

        probs_arrays = []
        action_arrays = []
        entropy_arrays = []
        state_arrays = []
        reward_arrays = []
        distance_arrays = []
        for i in range(sim_params["n_cyclists"]):
            probs_arrays.append([])
            action_arrays.append([])
            entropy_arrays.append([])
            state_arrays.append([])
            reward_arrays.append([])
            distance_arrays.append([])
        loc_end_array = []
        global_step = 0

        np.random.seed(seed)
        random.seed(seed)

        param_history.append(
            [
                worker_number,
                model["acs"][0]["critic"].lambda_reward,
                model["acs"][0]["critic"].distance_penalty,
            ]
        )

        # Beginning of simulation ---------------------------------------------
        for local_episode in range(hyper_params["window_size"]):
            # Keep track of game score to print
            pose_inits = (
                [
                    np.round(np.random.randn() * sim_params["random_std"], 1)
                    for i in range(sim_params["n_cyclists"])
                ]
                if sim_params["random_init"]
                else [0] * sim_params["n_cyclists"]
            )
            states = env.reset(poses=pose_inits)

            # Store the unormalised states
            [state_arrays[s["number"]].append(s["state"]) for s in states]

            states = [model["normaliser"].normalise_state(state) for state in states]
            step_entropies = [[] for i in range(sim_params["n_cyclists"])]
            scores = [0] * sim_params["n_cyclists"]
            group_score = 0
            group_reward = 0
            group_distance = 0
            step = 0

            actions = []
            probss = []
            for state in states:
                cyclist_number = state["number"]
                # Sample from policy and take action in environment
                probs, action, entropy = model["acs"][cyclist_number][
                    "actor"
                ].choose_action(state["state"])
                probs_arrays[cyclist_number].append(probs)
                action_arrays[cyclist_number].append(action)
                entropy_arrays[cyclist_number].append(entropy)
                step_entropies[cyclist_number].append(entropy)
                actions.append(action)
                probss.append(probs)

                model["acs"][cyclist_number]["critic"].store_transition_1(
                    state["state"], entropy
                )
            # We pass both actions, and the index of the action which
            [
                model["acs"][state["number"]]["critic"].store_actions(
                    actions.copy(), probss.copy(), i
                )
                for i, state in enumerate(states)
            ]
            global_step += 1

            while True:
                # Step
                states, rewards, done, info = env.step(actions)
                [state_arrays[s["number"]].append(s["state"]) for s in states]
                actions = []
                probss = []
                # Store rewards, normalise states
                group_reward += (
                    sum(rewards) * sim_params["lambda_group"] / sim_params["n_cyclists"]
                )
                group_distance += info[0] * sim_params["lambda_group"]
                group_score = sum(rewards) / sim_params["n_cyclists"]
                states = [
                    model["normaliser"].normalise_state(state) for state in states
                ]

                # Store results from last step, choose next action, store next actions, probs etc.
                for reward, score, state in zip(rewards, rewards, states):
                    cyclist_number = state["number"]
                    scores[cyclist_number] += (
                        1 - sim_params["lambda_group"]
                    ) * score + group_score * sim_params["lambda_group"]
                    probs, action, entropy = model["acs"][cyclist_number][
                        "actor"
                    ].choose_action(state["state"])
                    probs_arrays[cyclist_number].append(probs)
                    action_arrays[cyclist_number].append(action)
                    entropy_arrays[cyclist_number].append(entropy)
                    step_entropies[cyclist_number].append(entropy)
                    actions.append(action)
                    probss.append(probs)

                # We pass both actions, and the index of the action which
                for index, (state, reward) in enumerate(zip(states, rewards)):
                    cyclist_number = state["number"]
                    input_reward = (1 - sim_params["lambda_group"]) * reward
                    input_distance = (1 - sim_params["lambda_group"]) * info[0]
                    reward_arrays[cyclist_number].append(input_reward)
                    distance_arrays[cyclist_number].append(input_distance)
                    model["acs"][cyclist_number]["critic"].store_transition_loop(
                        state["state"],
                        actions.copy(),
                        index,
                        step_entropies[cyclist_number][-1],
                        probss.copy(),
                        input_reward,
                        input_distance,
                        model["acs"][cyclist_number]["actor"],
                        episode,
                    )

                # Deal with episode ends
                if done:
                    dividors, subtractors = model["normaliser"].update_normalisation()
                    [
                        mod["critic"].update_dividor_subtractor(
                            dividors, subtractors, mod["actor"]
                        )
                        for mod in model["acs"]
                    ]
                    # Store the dead states
                    for i, state in enumerate(states):
                        model["acs"][state["number"]]["critic"].store_dead(
                            state["state"],
                            actions.copy(),
                            probss.copy(),
                            i,
                            model["acs"][state["number"]]["actor"],
                            local_episode,
                        )
                    #
                    [
                        m["critic"].store_final_reward(group_reward, group_distance)
                        for m in model["acs"]
                    ]
                    for i in range(sim_params["n_cyclists"]):
                        reward_arrays[i][-1] += group_reward
                        distance_arrays[i][-1] += group_distance

                        # Need to add in some data to represent transition back to initial state. Note this will be ignored by setting the advantages and value labelss to 0
                        reward_arrays[i].append(0)
                        distance_arrays[i].append(0)
                    loc_end_array.append(global_step)
                    global_step += 1

                    episode_velocities.append(info[1])
                    episode_distances.append(info[2])
                    [
                        score + sim_params["lambda_group"] * group_score
                        for score in scores
                    ]
                    break
                else:
                    step += 1
                    global_step += 1

            episode_entropies.append([np.mean(entrops) for entrops in step_entropies])
            # Append for logging and print
            episode_rewards.append(scores)
            if sim_params["print_rewards"]:
                params = [
                    model["acs"][0]["critic"].lambda_reward,
                    model["acs"][0]["critic"].distance_penalty,
                ]
                print(
                    f"Seed: {seed}, , Params: {params}, EP: {episode}, Score: {np.round(scores)}",
                    flush=True,
                )
                print()
            #        sys.stdout.flush()
            episode += 1
        # End of simulation----------------------------------------------------
        queue.put(
            [
                worker_number,
                np.mean(episode_rewards[-hyper_params["window_size"] :]),
                np.mean(episode_entropies[-hyper_params["window_size"] :]),
            ]
        )

    # End of all simulations---------------------------------------------------
    Seed_entropies = np.array(episode_entropies)
    Seed_rewards = np.array(episode_rewards)
    Seed_velocities = np.array(episode_velocities)
    Seed_distances = np.array(episode_distances)
    return Seed_rewards, Seed_entropies, Seed_velocities, Seed_distances
