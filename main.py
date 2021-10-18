import argparse
import datetime
import gym
import itertools
import torch
import submitit
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from sac import SAC
from replay_memory import ReplayMemory
from envs import make_sawyer_push_env

def get_exp_name(args):
    exp_name = '{}_SAC_{}_{}_alpha-{}_{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env_name,
            args.obs,
            "auto" if args.auto_entropy_tuning else args.alpha,
            args.suffix,
    )
    return exp_name

def main(args, exp_name):
    logs_dir = f'/checkpoint/linamezghani/sac_logs/{exp_name}'

    env = make_sawyer_push_env(args)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(4, env.action_space, args)

    #Tensorboard
    writer = SummaryWriter(logs_dir)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    total_loss = {x: 0. for x in ['c1', 'c2', 'policy', 'entropy']}
    avg_alpha = 0.
    total_episode_reward = 0
    total_puck_dist = 0
    prev_log_step = 0
    prev_save_step = 0
    num_episodes = 0

    for i_episode in itertools.count(1):
        loss = {}
        episode_steps = 0
        episode_reward = 0
        done = False
        state = env.reset()
        obs = env.process_obs(state)
        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(obs)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    (loss['c1'], loss['c2'], loss['policy'], loss['entropy'],
                            alpha) = agent.update_parameters(memory,
                                    args.batch_size, updates)
                    for k in loss:
                        total_loss[k] += loss[k]
                    avg_alpha += alpha
                    updates += 1

            next_state, reward, done, info = env.step(action) # Step
            next_obs = env.process_obs(next_state)
            episode_steps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(obs, action, reward, next_obs, mask) # Append transition to memory

            state = next_state

        total_numsteps += episode_steps
        total_episode_reward += episode_reward
        total_puck_dist += info['puck_distance']
        num_episodes += 1

        if total_numsteps > args.num_steps:
            break

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, puck_dist: {}".format(
            i_episode, total_numsteps, episode_steps, round(episode_reward, 2),
            round(info['puck_distance'], 2)
        ))

        if total_numsteps - prev_log_step > args.log_interval:
            writer.add_scalar('train/reward', total_episode_reward/num_episodes,
                    total_numsteps)
            writer.add_scalar('train/puck_dist', total_puck_dist/num_episodes,
                    total_numsteps)
            writer.add_scalar('train/replay_buffer_len', len(memory),
                    total_numsteps)
            if updates > 0:
                for k, v in total_loss.items():
                    writer.add_scalar(f'loss/{k}', v/updates, total_numsteps)
                writer.add_scalar(f"train/alpha", avg_alpha/updates,
                        total_numsteps)
            total_episode_reward = 0
            total_puck_dist = 0
            num_episodes = 0
            updates = 0
            total_loss = {x: 0. for x in total_loss}
            avg_alpha = 0.
            prev_log_step = total_numsteps

        if total_numsteps - prev_save_step > args.save_interval:
            save_file = memory.save_buffer(logs_dir, return_path=True)
            writer.add_scalar('train/replay_buffer_size',
                    os.path.getsize(save_file), total_numsteps)
            agent.save_checkpoint(logs_dir, total_numsteps)
            prev_save_step = total_numsteps

        if i_episode % args.eval_interval == 0 and args.eval is True:
            avg_reward = 0.
            avg_puck_dist = 0.
            for _  in range(args.num_eval_episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(env.process_obs(state), evaluate=True)

                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward

                    state = next_state
                avg_reward += episode_reward
                avg_puck_dist += info['puck_distance']
            avg_reward /= args.num_eval_episodes
            avg_puck_dist /= args.num_eval_episodes

            writer.add_scalar('test/avg_reward', avg_reward, total_numsteps)
            writer.add_scalar('test/avg_puck-dist', avg_puck_dist,
                    total_numsteps)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, Avg. Puck Dist: {}".format(
                args.num_eval_episodes, round(avg_reward, 2), round(avg_puck_dist, 2)))
            print("----------------------------------------")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="SawyerPushNIPSEasy-v0",
                        help='Mujoco Gym environment (default: SawyerPushNIPSEasy-v0)')
    parser.add_argument('--suffix', default="", help='exp suffix (default: "")')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--obs', default="rgb",
                        help='Observation Type: rgb | vec (default: rgb)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy every eval-interval episodes (default: True)')
    parser.add_argument('--imsize', type=int, default=48, metavar='N',
                        help='image size (default: 48)')
    parser.add_argument('--eval-interval', type=int, default=100, metavar='N',
                        help='perform eval every x episodes (default: 100)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='log every x steps (default: 100)')
    parser.add_argument('--save-interval', type=int, default=5000, metavar='N',
                        help='save every x steps (default: 100)')
    parser.add_argument('--num-eval-episodes', type=int, default=20, metavar='N',
                        help='(default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0.05, metavar='G',
                        help='Temperature parameter α determines the relative\
                                importance of the entropy term against the\
                                reward (default: 0.05)')
    parser.add_argument('--auto_entropy_tuning', type=bool, default=False,
            metavar='G', help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size (default: 1024)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=2, metavar='N',
                        help='model updates per simulator step (default: 2)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                        help='size of replay buffer (default: 100000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--local', action="store_true",
                        help='run locally (default: False)')
    parser.add_argument('--debug', action="store_true",
                        help='debug mode (default: False)')
    args = parser.parse_args()

    if args.debug:
        args.local = True
        args.log_interval = 50
        args.save_interval = 200
        args.eval_interval = 5
        args.num_eval_episodes = 10

    exp_name = get_exp_name(args)
    print(exp_name)

    if args.local:
        main(args, exp_name)

    else:
        submitit_log = 'submitit_logs/'
        executor = submitit.AutoExecutor(folder=submitit_log)
        executor.update_parameters(timeout_min=4320, partition="devlab",
                gpus_per_node=1, name=exp_name)
        job = executor.submit(main, args, exp_name)
        print(job.job_id)
