import gym
from gym.wrappers.time_limit import TimeLimit

import numpy as np
import multiworld

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

class MyCustomPush(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = self.env._max_episode_steps

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = - info['puck_distance']
        success = bool(info['puck_distance'] < 0.03)
        done = done or success
        info['dist_to_goal'] = info['puck_distance'] if done else 0.
        info['success'] = success
        return obs, reward, done, info
    
def make_sawyer_push_env():
    multiworld.register_all_envs()
    env = gym.make('SawyerPushNIPSEasy-v0')
    presampled_goals_path = 'data/random_goals/pusher_goals_500.npy'
    presampled_goals = np.load(presampled_goals_path, allow_pickle=True)
    presampled_goals = presampled_goals.tolist()
    env = ImageEnv(env, imsize=48, presampled_goals=presampled_goals, 
            init_camera=sawyer_init_camera_zoomed_in, transpose=True)#, run_update_info=False)
    env = TimeLimit(env, max_episode_steps=50)
    env = MyCustomPush(env)
    return env
