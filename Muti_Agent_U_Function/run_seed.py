import numpy as np
from environment import Environment
import pickle
import random

    
def run_seed(seed_params):
    '''
    Runs a learning simulation for the seed in question.

    Parameters
    ----------
    seed_params : dict

    Returns
    -------
    Seed_rewards : np.array
        Reward history for the seed.
    Seed_entropies : np.array
        Entropy history for the seed.
    Loss_history : list
        History of the losses of the critic's value function.

    '''
    from networks import Actor, Critic, State_normaliser

    # Create gym and seed numpy
    env_params = seed_params['Env']
    sim_params = seed_params['Sim']
    hyper_params = seed_params['Hyp'].copy()
    
    # Change relative points to episode points for explore profile
    for point in hyper_params['lambda_reward_profile']:
        point[0] = int(point[0] * sim_params['n_episodes'])
        
    
    seed = seed_params['seed']
    env = Environment(env_params, n_cyclists = sim_params['n_cyclists'])
    np.random.seed(seed)
    random.seed(seed)

    # Init weights
    if sim_params['weights'] == 'none':
        model = {'normaliser' : State_normaliser(env, hyper_params)}

        acs = []
        for i in range(sim_params['n_cyclists']):
            acs.append({
                'actor' : Actor(env, hyper_params, np.random.randint(1000)),
                'critic' : Critic(env, hyper_params, np.random.randint(1000))
                })
        model.update({'acs' : acs})
    else:
        model = sim_params['weights']
    
    # Keep stats for final print of graph
    episode_rewards = []
    episode_entropies = []
    episode_velocities = []
    episode_distances = []
    # Main loop 
    for episode in range(sim_params['n_episodes']):
    	# Keep track of game score to print
        pose_inits = [np.round(np.random.randn() * sim_params['random_std'], 1) for i in range(sim_params['n_cyclists'])] if sim_params['random_init'] else [0] * sim_params['n_cyclists']
        states = env.reset(poses = pose_inits)
        
        states = [model['normaliser'].normalise_state(state) for state in states]
        step_entropies = [[] for i in range(sim_params['n_cyclists'])]
        scores = [0] * sim_params['n_cyclists']
        group_score = 0
        group_reward = 0
        step = 0

        actions = []
        probss = []
        for state in states:
            cyclist_number = state['number']
      		# Sample from policy and take action in environment
            probs, action, entropy = model['acs'][cyclist_number]['actor'].choose_action(state['state'])
            step_entropies[cyclist_number].append(entropy)
            actions.append(action)
            probss.append(probs)
              
            model['acs'][cyclist_number]['critic'].store_transition_1(state['state'], entropy)
        # We pass both actions, and the index of the action which
        [model['acs'][state['number']]['critic'].store_actions(actions.copy(), probss.copy(), i) for i, state in enumerate(states)]
        
        while True:
            # Step
            states, rewards, done, info = env.step(actions)
            actions = []
            probss = []
            # Store rewards, normalise states
            group_reward += sum(rewards) / sim_params['n_cyclists']
            group_score = sum(rewards) / sim_params['n_cyclists']
            states = [model['normaliser'].normalise_state(state) for state in states]             

            for reward, score, state in zip(rewards, rewards, states):
                cyclist_number = state['number']
                scores[cyclist_number] += (1 - sim_params['lambda_group']) * score + group_score * sim_params['lambda_group']
                probs, action, entropy = model['acs'][cyclist_number]['actor'].choose_action(state['state'])
                step_entropies[cyclist_number].append(entropy)
                actions.append(action)
                probss.append(probs)

            # We pass both actions, and the index of the action which
            for index, (state, reward) in enumerate(zip(states, rewards)):
                cyclist_number = state['number']
                input_reward = (1 - sim_params['lambda_group']) * reward
                model['acs'][cyclist_number]['critic'].store_transition_loop(state['state'], actions.copy(), index, step_entropies[cyclist_number][-1], probss.copy(), input_reward, model['acs'][cyclist_number]['actor'], episode)

            # Deal with episode ends
            if done:
                dividors, subtractors = model['normaliser'].update_normalisation()
                [mod['critic'].update_dividor_subtractor(dividors, subtractors, mod['actor']) for mod in model['acs']]
                # Store the dead states
                for i, state in enumerate(states):
                    model['acs'][state['number']]['critic'].store_dead(state['state'], actions.copy(), probss.copy(), i, model['acs'][state['number']]['actor'], episode)
#
                [m['critic'].store_final_reward(group_reward) for m in model['acs']]
                episode_velocities.append(info[0])
                episode_distances.append(info[1])
                [score + sim_params['lambda_group'] * group_score for score in scores]
                break
            else:
                step += 1  
                  
        episode_entropies.append([np.mean(entrops) for entrops in step_entropies])
    	# Append for logging and print
        episode_rewards.append(scores) 
        if sim_params['print_rewards']:
            print(f'Seed: {seed}, EP: {episode}, Score: {np.round(scores)}',flush = True)
            print()
        if episode % 1000 == 0:
            pickle.dump(episode_rewards, open('reward_dump' + str(seed) + '.p', 'wb'))

#        sys.stdout.flush()
    
    Seed_entropies = np.array(episode_entropies)
    Seed_rewards = np.array(episode_rewards)
    Seed_velocities = np.array(episode_velocities)
    Seed_distances = np.array(episode_distances)

    return Seed_rewards, Seed_entropies, seed_params, Seed_velocities, Seed_distances

