import gym
from gym.wrappers.time_limit import TimeLimit

import numpy as np
import multiworld

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

class MyCustomPush(gym.core.Wrapper):
    def __init__(self, env, obs_type, success_thresh):
        super().__init__(env)
        self.obs_type = obs_type
        self.success_thresh = success_thresh
        self._max_episode_steps = self.env._max_episode_steps

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = - info['puck_distance']
        success = bool(info['puck_distance'] < self.success_thresh)
        done = done or success
        info['dist_to_goal'] = info['puck_distance'] if done else 0.
        info['success'] = success
        return self.process_obs(obs), reward, done, info

    def reset(self):
        obs = super().reset()
        return self.process_obs(obs)

    def get_num_inputs(self):
        if self.obs_type == 'rgb':
            return 8
        elif self.obs_type == 'vec':
            return 6

    def process_obs(self, obs):
        if self.obs_type == 'rgb':
            obs = np.stack((obs['observation'], obs['desired_goal']))
            return obs.astype('uint8')
        elif self.obs_type == 'vec':
            return np.concatenate((obs['state_observation'],
                obs['state_desired_goal'][2:]))
    
def make_sawyer_push_env(args):
    multiworld.register_all_envs()
    env = gym.make('SawyerPushNIPSEasy-v0')
    if args.obs == 'rgb':
        presampled_goals_path = 'data/random_goals/pusher_goals_500.npy'
        presampled_goals = np.load(presampled_goals_path, allow_pickle=True)
        presampled_goals = presampled_goals.tolist()
        env = ImageEnv(env, imsize=args.imsize,
                presampled_goals=presampled_goals,
                init_camera=sawyer_init_camera_zoomed_in, transpose=True,
                run_update_info=False)
    env = TimeLimit(env, max_episode_steps=50)
    env = MyCustomPush(env, obs_type=args.obs,
            success_thresh=args.success_thresh)
    return env
