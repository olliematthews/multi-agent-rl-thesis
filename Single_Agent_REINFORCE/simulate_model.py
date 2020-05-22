# -*- coding: utf-8 -*-
"""
Will generate an animation of a race.
"""

#from learning_function import policy
import pickle
from environment import Cyclist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython import get_ipython
import matplotlib

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

matplotlib.use("Agg")

get_ipython().run_line_magic('matplotlib', 'qt')

def updatefig(i):
    global cyclist, pose, background, cyclist_height, cyclist_width
    cyclist_image = np.zeros(background.shape)
    if i < len(pose):
        pose_cur = pose[i]
    else:
        pose_cur = pose[-1]
    cyclist_image[road_height - cyclist_height : road_height, pose_cur + 5 - int(cyclist_width/2): pose_cur + 5 + int(cyclist_width/2), :] = cyclist
    image = background - cyclist_image
    im.set_array(image)
    return im,

# Our policy that maps state to action parameterized by w
def policy(state,w):
	z = state.dot(w)
	exp = np.exp(z - np.max(z))
	return (exp/np.sum(exp))

def normalise_state(state, env_params):
    return state / np.array([env_params['race_length'], env_params['vel_max'], 100])

def run_policy(env_params, w):
    cyclist = Cyclist(env_params)
    state = np.squeeze(np.append(normalise_state(cyclist.reset(),env_params),1))
    poses = []
    np.random.seed(3)
    for i in range(200):
        probs = np.array(policy(state,w))
        u = np.random.uniform()
        aprob_cum = np.cumsum(probs)
        action = np.where(u <= aprob_cum)[0][0]
        poses.append(int(state[0] * env_params['race_length']))
        next_state,reward,done = cyclist.step(action - 1,env_params)
        state = np.append(normalise_state(next_state,env_params),1)

        if done:
            return poses
        
    
    
w_best,_, _, _, _, _, params = pickle.load(open('BigTest.p','rb'))


env_params = params['Env']

poses = run_policy(env_params, w_best)

fig = plt.figure()

pose = poses


background = np.ones([100,300,3]) * 1
track = np.zeros(background.shape)
road_height = int(0.8 * track.shape[0])
track[road_height:,:,:] = 1
background -= track

cyclist_height = 20
cyclist_width = 4
cyclist = np.ones([cyclist_height,cyclist_width,3]) * 1
cyclist[:,:,2] = 0

im = plt.imshow(background, animated=True)
plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True, save_count=200)
ani.save('im_3.mp4', writer=writer)

plt.show()
