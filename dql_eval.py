import numpy as np
import gymnasium as gym
import random
import argparse
import os
import json
import torch

from dql import DoubleDQL


def main(args):
    # Load learning parameters
    with open(args.path + 'learning/params.dat') as pf:
        params = json.load(pf)

    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random generator seed
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    env = gym.make("CartPole-v1", render_mode=("human" if args.render else
                                               None))

    # The DoubleDQL class object
    dqn = DoubleDQL(
        train_env=env, eval_env=env, hl1_size=params['hl1_size'],
        hl2_size=params['hl2_size'], replay_mem_size=1, device=device)

    # Load the trained model from the specified path
    if args.latest:
        model_path = args.path + 'models/latest_policy.pth'
        print("Loading the latest policy model")
    elif args.trained:
        model_path = args.path + 'models/best_train_policy.pth'
        print("Loading the best trained policy model")
    else:  # Load the best evaluation set policy
        if args.best_episode is None:
            # Load the best policy model with the highest postfix episode number
            episode = max([int(f.split('_')[-1].split('.')[0]) for f in
                           os.listdir(args.path + 'models') if
                           'best_eval_policy' in f and 'train' not in f])
            model_path = args.path + f'models/best_eval_policy_{episode}.pth'
        else:
            episode = args.best_episode
            model_path = args.path + f'models/best_eval_policy_{episode}.pth'

        print(f"Loading model for episode: {episode}")

    dqn.load_main_dqn(model_path)

    # Evaluate the policy
    mean_reward, avg_steps = dqn.evaluate(
        num_episodes=args.num_episodes, epsilon=0)

    print(f"Mean reward: {np.round(mean_reward, 4)}, "
          f"Average steps: {np.round(avg_steps, 4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate a trained DDQL policy for the CartPole-v1 '
                    'environment')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to the trained model folder (default: None)')
    parser.add_argument('--latest', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Evaluate the latest policy')
    parser.add_argument('--trained', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Evaluate the best trained policy')
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--best_episode', type=int, default=None,
                        help='Evaluate the best policy at a specific episode')
    parser.add_argument('--render', default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Render the policy during evaluation')

    parser_args = parser.parse_args()

    main(parser_args)
