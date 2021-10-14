import gym
from gym.wrappers.time_limit import TimeLimit

import numpy as np
import multiworld

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

class MyCustomPush(gym.core.Wrapper):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = - info['puck_distance']
        done = info['puck_distance'] < 0.03
        return obs, reward, done, info
    
def make_sawyer_push_env():
    multiworld.register_all_envs()
    env = gym.make('SawyerPushNIPSEasy-v0')
    presampled_goals_path = 'data/random_goals/pusher_goals_500.npy'
    presampled_goals = np.load(presampled_goals_path, allow_pickle=True)
    presampled_goals = presampled_goals.tolist()
    env = ImageEnv(env, imsize=48, presampled_goals=presampled_goals, 
            init_camera=sawyer_init_camera_zoomed_in, transpose=True)#, run_update_info=False)
    env = MyCustomPush(env)
    env = TimeLimit(env, max_episode_steps=50)
    return env
