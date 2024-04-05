# Codebase for Master's Project on Multi-Agent Reinforcement Learning

## Environment

The environment is meant to simulate cyclists in a race. The aim is to travel as far as possible. In a multi-agent case, this requires collaborative policies where the agents cycle close together.

## Notes

This is the final approach settled on. Some of the other approaches tested are on the `all-approaches` branch.


### Requirements
* Tensorflow < 2
* Keras (code written for version 2.3.1)
* Code written for Cuda 10.1



# Multi Agent, Actor Critic with U-Function and Hyperparameter Tuning

## Environment

This is the more simple multi-agent environment.
* The cyclist's state is defined by a pose, velocity, energy and time, as well as the relative poses, velocities and the total energy of the other cyclists.
* The episode ends **any one** of the cyclists run out of energy or the time runs out.
* The reward is the distance travelled in a step i.e. the cyclist's velocity.
* The cyclist's energy decreases proportionally to $v^3$, multiplied by $c_D$ which depends on the proximity to the rider in front.

## Algorithm

The algorithm is an implementation of the Actor Critic algorithm with a 'coach' framework for optimising learning hyperparameters. We also implement entropy regularisation. Instead of learning a single agent value function, we learn a multi-agent U-Function, which represents:

![equation](https://latex.codecogs.com/gif.latex?U(s,\bar{a})&space;=&space;\mathbb{E}\left[\sum_t&space;r_t&space;\middle|&space;s&space;=&space;s,&space;\bar{a}&space;=&space;\bar{a}\right]),

where ![equation](https://latex.codecogs.com/gif.latex?\bar{a}) represents the actions of all of the agents apart from one partiular agent. This deals with non-stationarity in the multi-agent environment.

### The Coach

The coach oversees learning and tries to optimise hyperparameters. In this case there are two hyperparameters:

* The exploitation parameter
* The distance penalty, which encourages cyclists to stay close together

The coach does this by running two simulations in parallel. Each one tests one hyperparameter value. At the end, the best hyperparameter informs the next hyperparameter values to test. We use a multiplicative line search, such that if we find that an increase in value of x is helpful, we try increasing by 2x on the next set of simulations. Else, we switch search directions. We cycle through hyperparameters one by one, never staying on one for longer than a set time.
