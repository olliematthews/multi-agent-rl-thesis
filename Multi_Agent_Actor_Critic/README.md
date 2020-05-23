# Single Agent, Reinforce

## Environment

This is the more simple multi-agent environment.
* The cyclist's state is defined by a pose, velocity, energy and time, as well as the relative poses, velocities and the total energy of the other cyclists.
* The episode ends **any one** of the cyclists run out of energy or the time runs out.
* The reward is the distance travelled in a step i.e. the cyclist's velocity.
* The cyclist's energy decreases proportionally to $v^3$, multiplied by $c_D$ which depends on the proximity to the rider in front.

## Algorithm

The algorithm is an implementation of the Actor Critic algorithm. We also implement entropy regularisation. 
