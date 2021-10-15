import argparse
import datetime
import gym
import itertools
import torch
import submitit

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import multiprocessing as mp

from sac import SAC
from replay_memory import ReplayMemory
from envs import make_sawyer_push_env
from gym.spaces import Box

def process_obs(obs):
    return np.concatenate((obs['observation'], obs['desired_goal']))

#env = gym.make(args.env_name)
def main(args):
    #eval_envs = []
    #for _ in range(args.n_eval_procs):
    #    eval_envs.append(make_sawyer_push_env())

    obs_dim =48 * 48 * 3 * 2
    act_dim = 2
    #obs_dim = eval_envs[0].observation_space['observation'].shape[0] * 2
    #act_dim = eval_envs[0].action_space.shape[0]
    ctx = mp.get_context("fork")
    barriers, buffers, n_eval_done = create_mputils(args.n_eval_procs, obs_dim,
            act_dim, ctx)

    # Agent
    agent = SAC(obs_dim, Box(0, 1, shape=(2,)), args)
    
    #Tensorboard
    writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                                 args.policy, "autotune" if args.automatic_entropy_tuning else ""))
    
    # Training Loop
    total_numsteps = 0
    updates = 0
    
    for i_episode in itertools.count(1):
        avg_reward = 0.
        avg_puck_dist = 0.
        episodes = 10

        procs = []
        for i in range(args.n_eval_procs):
            p = ctx.Process(target=worker_eval, args=(i,
                buffers, barriers, n_eval_done, episodes,))
            p.start()
            procs.append(p)

        print("processes launched")

        n_eval_done.value = 0
        eval_stat = {x: 0.0 for x in ['success', 'dist_to_goal']}
        while n_eval_done.value < episodes:
            barriers["eval_obs"].wait()
            with torch.no_grad():
                actions = agent.select_actions(buffers['eval_obs'],
                        evaluate=True)
                buffers["eval_act"].copy_(actions)
            barriers["eval_act"].wait()
            barriers["eval_stp"].wait()
            for x in eval_stat:
                eval_stat[x] += buffers[f"eval_{x}"].sum().item()
        for p in procs:
            p.join()
        for x in eval_stat:
            eval_stat[x] /= n_eval_done.value
            writer.add_scalar(f'eval/{x}', eval_stat[x], i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}".format(episodes))
        print("----------------------------------------")

def worker_eval(i, buffers, barriers, num_eval_done, num_episodes):
    env = make_sawyer_push_env()
    obs = env.reset()
    while num_eval_done.value < num_episodes:
        buffers["eval_obs"][i] = torch.from_numpy(process_obs(obs))
        barriers["eval_obs"].wait()
        barriers["eval_act"].wait()
        obs, _, done, info = env.step(buffers["eval_act"][i])
        for key in ['success', 'dist_to_goal']:
            buffers[f"eval_{key}"][i] = info[key]
        if done:
            with num_eval_done.get_lock():
                num_eval_done.value += 1
            obs = env.reset()
        barriers["eval_stp"].wait()

def create_mputils(n, obs_dim, act_dim, ctx):
    Barrier = ctx.Barrier
    Value = ctx.Value

    barriers = {
        "eval_obs": Barrier(n + 1),
        "eval_act": Barrier(n + 1),
        "eval_stp": Barrier(n + 1),
    }

    num_eval_done = Value('i', 0)

    buffers = {}
    buffers["eval_obs"] = torch.zeros((n, obs_dim)) 
    buffers["eval_act"] = torch.zeros((n, act_dim))
    for x in ['success', 'dist_to_goal']:
        buffers[f"eval_{x}"] = torch.zeros(n)

    return barriers, buffers, num_eval_done

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--n-eval-procs', type=int, default=1, metavar='N',
                        help='num eval processes')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--local', action="store_true",
                        help='run locally (default: False)')
    args = parser.parse_args()

    if args.local:
        main(args)
        
    else:
        submitit_log = 'submitit_logs/'
        executor = submitit.AutoExecutor(folder=submitit_log)
        executor.update_parameters(timeout_min=4320, partition="devlab",
                gpus_per_node=1, name=exp_name)
        job = executor.submit(learn_model, args)
        print(job.job_id)
