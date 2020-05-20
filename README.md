# Codebase for Master's Project on Multi-Agent Reinforcement Learning

## Environment

The environment is meant to simulate cyclists in a race. The aim is to travel as far as possible. In a multi-agent case, this requires collaborative policies where the agents cycle close together.

## Notes
Each directory holds different stages of the code. The environment is either single, multi agent and different algorithms are implemented to each. The algorithms are:

* REINFORCE
* The Actor Critic algorithm
* The Actor Critic with a non-stationary U-function (cross between a V and Q function)
* The Actor Critic with a mechanism for automatically tuning hyperparameters according to a "coach"
