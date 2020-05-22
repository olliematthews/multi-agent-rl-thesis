# Single Agent, Actor Critic

## Environment

This is a single agent environment, so there is only one cyclist.
* The cyclist's state is defined by a pose, velocity, energy and time.
* The episode ends when the cyclist runs out of time or energy.
* The reward is the distance travelled in a step i.e. the cyclist's velocity.
* The cyclist's energy decreases proportionally to $v^3$.


## Algorithm

The algorithm is an implementation of the Actor Critic algorithm. We also use epsilon-greedy exploration.
