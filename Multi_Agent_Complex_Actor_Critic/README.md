# Multi Agent Complex, Actor Critic

## Environment

This is the original multi-agent environment tested. It is harder than the non 'complex' one.
* The cyclist's state is defined by a pose, velocity, energy and time.
* Each cyclist also recieves the relative position and velocity of the next rider in front. If there is none, they recieve 0.
* The episode ends all of the cyclists run out of energy or the time runs out.
* The reward is the distance travelled in a step i.e. the cyclist's velocity.
* The cyclist's energy decreases proportionally to $v^3$, multiplied by $c_D$ which depends on the proximity to the rider in front.

Note the environment is slightly different to the REINFORCE environment. This is due to the care needed in the Actor Critic when dealing with dead states.

## Algorithm

The algorithm is an implementation of the Actor Critic algorithm. We also implement entropy regularisation.
