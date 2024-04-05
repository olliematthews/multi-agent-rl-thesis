# Multi-Agent Reinforcement Learning with a Coach

This is my dissertation project (done in 2020), for which the aim was to investigate techniques for cooperative multi-agent reinforcement learning in a simple problem setting.

The code on this branch is the final approach settled on. Some of the other approaches tested are on the `all-approaches` branch.

## Installation

The project can be setup with poetry, but running `poetry install`. You are advised to do this starting from a new virtual environment, with only `poetry` installed. 

The main entrypoint for the code is in `src/main.py`.



## Environment

The environment simulates cyclists in a 1d race. The aim is to travel as far as possible in a set number of time steps. Agents have finite energy, and can conserve energy by cycling as close to possible to the rider in front (simulating a slip-streaming effect).

* The cyclist's state is defined by a pose, velocity, energy and time, as well as the relative poses, velocities and the total energy of the other cyclists.
* The episode ends **any one** of the cyclists runs out of energy **or** the time runs out.
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
