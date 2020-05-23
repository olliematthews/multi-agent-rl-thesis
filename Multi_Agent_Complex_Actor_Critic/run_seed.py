import numpy as np
from environment import Environment
from networks import Actor, Critic, State_normaliser
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
    env_params = seed_params['Env']
    sim_params = seed_params['Sim']
    hyper_params = seed_params['Hyp']
    
    seed = seed_params['seed']
    env = Environment(env_params, n_cyclists = sim_params['n_cyclists'])
    action_space = env.action_space
    np.random.seed(seed)
    state_normaliser = State_normaliser(env_params)
    nA = len(env.action_space)
    # Init weights
    model = []
    for i in range(sim_params['n_cyclists']):
        model.append({
            'actor' : Actor(env, hyper_params, seed),
            'critic' : Critic(env, hyper_params, seed)
            })
    
    # Keep stats for final print of graph
    episode_rewards = []
    episode_entropies = []
    
    # Main loop 
    for episode in range(sim_params['n_episodes']):
        cyclists_done = []
    	# Keep track of game score to print
        states = env.reset()
        
        states = [state_normaliser.normalise_state(state) for state in states]
        step_rewards = [[] for i in range(sim_params['n_cyclists'])]
        step_entropies = [[] for i in range(sim_params['n_cyclists'])]
        
        scores = [0] * sim_params['n_cyclists']
        step = 0

        while True:
            actions = []
            for state in states:
                cyclist_number = state['number']
        		# Sample from policy and take action in environment
                if cyclist_number in cyclists_done:
                    model[cyclist_number]['critic'].store_dead(state['state'], model[state['number']]['actor'])
                    actions.append(0)
                else:
                    try:
                        probs, action, entropy = model[cyclist_number]['actor'].choose_action(state['state'])
                    except ValueError:
                        break
                    step_entropies[cyclist_number].append(entropy)
                    actions.append(action)
                    
                    model[cyclist_number]['critic'].store_transition_1(state['state'], action, entropy, probs)
                 
            next_states, rewards, done, cyclists_done = env.step(actions)
            next_states = [state_normaliser.normalise_state(state) for state in next_states]             
            for next_state, reward in zip(next_states,rewards):
                cyclist_number = next_state['number']
                
                model[next_state['number']]['critic'].store_transition_2(reward, next_state['state'], model[cyclist_number]['actor'])

            states = next_states.copy()
            for state, reward in zip(states, rewards):
                scores[state['number']] += reward
                step_rewards[state['number']].append(reward)

            step += 1
            if done:
                # print(state['state'])
                [model[state['number']]['critic'].store_dead(state['state'], model[state['number']]['actor']) for state in states]
                break
                    
                
        episode_entropies.append([np.mean(entrops) for entrops in step_entropies])
    	# Append for logging and print
        episode_rewards.append(scores) 
        if sim_params['print_rewards']:
            print(f'Seed: {seed}, EP: {episode}, Score: {scores}',flush = True)
            print()
    
    Seed_entropies = np.array(episode_entropies)
    Seed_rewards = np.array(episode_rewards)
    return Seed_rewards, Seed_entropies, [c['critic'].losses for c in model]

