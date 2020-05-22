"""
Contains the 'agent' class that implements a policy based on a neural network,
and learns using REINFORCE.
Code based on the code described here:
    https://www.youtube.com/watch?v=IS0V8z8HXrM
"""

from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.initializers import glorot_normal, Constant
import keras.backend as K
import tensorflow as tf
import numpy as np

# Set a limit to how much GPU memory keras uses ------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



class Agent(object):
    def __init__(self, lr, env_params, alpha_mean = 0.1, alpha_std = 0.05, 
                 GAMMA=0.99, n_actions=3, layer1_size=5, layer2_size=5, 
                 input_dims=5, batch_size = 1, seed = 0):
        '''
        Initialiser

        Parameters
        ----------
        lr : float
            Learning rate.
        env_params : dict
            Environment parameters.
        alpha_mean : float, optional
            The momentum parameter for the baseline. The default is 0.1.
        alpha_std : float, optional
            The momentum parameter for the rolling standard deviation. The 
            default is 0.05.
        GAMMA : float, optional
            Discount rate. The default is 0.99.
        n_actions : int, optional
            The size of the action space. The default is 3.
        layer1_size : int, optional
            Size of the first hidden layer. The default is 5.
        layer2_size : int, optional
            Size of the second hidden layer. The default is 5.
        input_dims : int, optional
            Input layer size (state space size). The default is 5.
        batch_size : int, optional
            The batch size for policy updates. The default is 1.
        seed : int, optional
            Random seed. The default is 0.

        Returns
        -------
        None.

        '''
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.gamma = GAMMA
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.action_space = [i for i in range(n_actions)]
        self.rolling_average_reward = np.zeros(env_params['time_limit'] * batch_size)
        self.rolling_std_dev = 1
        self.episode = 0
        self.batch_size = batch_size
        self.seed = seed
        self.advantages = np.empty([0,])
        self.states = np.empty([0,self.input_dims])
        self.actions = np.empty([0,])
        np.random.seed(seed)
        self.policy, self.predict = self.build_policy_network()

    def build_policy_network(self):
        '''
        Build the two layer neural network.

        Returns
        -------
        policy : keras Model
            This is the model which should be used for learning.
        predict : keras Model
            This is the model which should be used for policy predictions.

        '''
        input = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        
        dense1 = Dense(self.fc1_dims, activation='relu', kernel_initializer = glorot_normal(seed=self.seed))(input)
        dense2 = Dense(self.fc2_dims, activation='relu', kernel_initializer = glorot_normal(seed=self.seed))(dense1)
        probs = Dense(self.n_actions, activation='softmax', kernel_initializer = Constant(value = np.random.normal(loc = 0.0, scale = np.sqrt(2/(self.n_actions + self.fc2_dims)))))(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)

        policy = Model(inputs=[input, advantages], outputs=[probs])

        policy.compile(optimizer=Adam(lr=self.lr), loss = custom_loss)

        predict = Model(inputs=[input], outputs=[probs])

        return policy, predict

    def print_weights(self):
        '''
        Function will print the weights for the first layer, for debugging 
        purposes.

        Returns
        -------
        None.

        '''
        print(self.policy.layers[1].get_weights())

    def choose_action(self, observation):
        '''
        Choose an action based on a state observation.

        Parameters
        ----------
        observation : np.array
            An observation of the state.

        Returns
        -------
        action : int
            Action.
        entropy : float
            Entropy.

        '''
        state = observation[np.newaxis, :]
        probs = self.predict.predict(state)[0]
        entropy = self.get_entropy(probs)
        action = np.random.choice(self.action_space, p=probs)

        return action, entropy
    
    def get_entropy(self, probs):
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
        return - np.sum(probs * np.log(probs + 1e-4)) 

    def store_transition(self, observation, action, reward):
        '''
        Store the transition.

        Parameters
        ----------
        observation : np.array
            State observation.
        action : int
            Action.
        reward : float
            The reward from the action taken.

        Returns
        -------
        None.

        '''
        
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        '''
        Update the policy based on the stored state, action, reward tuples.

        Returns
        -------
        None.

        '''
        
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)
        state_rewards = np.zeros_like(reward_memory).astype(float)
        for i in range(len(reward_memory)):
    		# Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
            state_rewards[i] = sum([ r * (self.gamma ** t) for t,r in enumerate(reward_memory[i:])])
        self.rolling_average_reward[:len(reward_memory)] = self.rolling_average_reward[:len(reward_memory)] * (1 - self.alpha_mean) + self.alpha_mean * state_rewards
        self.rolling_std_dev = np.sqrt(self.rolling_std_dev ** 2 * (1 - self.alpha_std) + np.mean(self.alpha_std * (state_rewards - self.rolling_average_reward[:len(reward_memory)]) ** 2))
        advantages = (state_rewards - self.rolling_average_reward[:len(reward_memory)]) / (self.rolling_std_dev + 1e-2)
        self.advantages = np.append(self.advantages,advantages)
        self.states = np.append(self.states, state_memory, axis = 0)
        self.actions = np.append(self.actions, action_memory, axis = 0)
        if self.episode % self.batch_size == self.batch_size - 1:
            actions = np.zeros([self.actions.shape[0], self.n_actions])
            actions[np.arange(self.actions.shape[0]), self.actions.astype(int)] = 1
            cost = self.policy.train_on_batch([self.states, self.advantages], actions)
            self.advantages = np.empty([0,])
            self.states = np.empty([0,self.input_dims])
            self.actions = np.empty([0,])

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.episode += 1
