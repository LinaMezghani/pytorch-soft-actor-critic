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
        self.imshape = (3, self.imsize, self.imsize)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = - info['puck_distance']
        success = bool(info['puck_distance'] < 0.03)
        done = done or success
        info['dist_to_goal'] = info['puck_distance'] if done else 0.
        info['success'] = success
        return self.reshape_obs(obs), reward, done, info

    def reset(self):
        obs = super().reset()
        return self.reshape_obs(obs)

    def reshape_obs(self, obs):
        for x in ['observation', 'desired_goal']:
            obs[x] = obs[x].astype(int).reshape(self.imshape)
            obs[x] = np.flip(obs[x], axis=2).copy().transpose(0, 2, 1)
        return obs
    
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
