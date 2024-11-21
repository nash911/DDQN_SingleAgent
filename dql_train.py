import numpy as np
import gymnasium as gym
import random
import argparse
import os
import json

from datetime import datetime
from shutil import rmtree
from copy import deepcopy

import torch
import torch.nn as nn

from cartpole_leftright_env import CartPoleLeftRightEnv
from dql import DoubleDQL


def save_learning_params(max_train_episodes: int):
    """
    Saves the training parameters in a separate file for reference

    Parameters
    ----------
    max_train_episodes : int
        Maximum number of training episodes

    Return
    ------
    learning_params : dict
        Dictionary containing the environment and training parameters
    """

    # Create a dictionary to store the learning parameters
    learning_params = dict()

    # Environment
    # -----------
    learning_params['seed'] = SEED

    # Adam
    # ----
    learning_params['lr'] = LR
    learning_params['beta_1'] = BETA_1
    learning_params['beta_2'] = BETA_2

    # TDL
    # ---
    learning_params['gamma'] = GAMMA
    learning_params['epsilon'] = EPSILON
    learning_params['epsilon_decay'] = EPSILON_DECAY
    learning_params['epsilon_min'] = EPSILON_MIN
    learning_params['per_alpha'] = PER_ALPHA
    learning_params['per_beta'] = PER_BETA

    # NN
    # --
    learning_params['hl1_size'] = HL1_SIZE
    learning_params['hl2_size'] = HL2_SIZE
    learning_params['batch_size'] = BATCH_SIZE

    # DQL
    # ---
    learning_params['replay_mem_size'] = REPLAY_MEM_SIZE
    learning_params['initial_period'] = INITIAL_PERIOD
    learning_params['main_update_period'] = MAIN_UPDATE_PERIOD
    learning_params['target_update_period'] = TARGET_UPDATE_PERIOD
    learning_params['max_train_episodes'] = max_train_episodes

    # Logging
    # -------
    learning_params['evaluation_frequency'] = EVALUATION_FREQUENCY

    return learning_params


def main(args):
    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random generator seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Create a timestamp directory to save model, parameter and log files
    training_dir = \
        ('training/' +
         str(datetime.now().date()) + '_' + str(datetime.now().hour).zfill(2) + '-' +
         str(datetime.now().minute).zfill(2) + '/')

    # Delete it if a directory with the same name already exists
    if os.path.exists(training_dir):
        rmtree(training_dir)

    # Create empty directories for saving model, parameter and log files
    os.makedirs(training_dir)
    os.makedirs(training_dir + 'plots')
    os.makedirs(training_dir + 'learning')
    os.makedirs(training_dir + 'models')

    # Save the learning parameters for reference
    learning_params = save_learning_params(args.max_train_steps)

    # Dump learning params to file
    with open(training_dir + 'learning/params.dat', 'w') as param_file:
        json.dump(learning_params, param_file, indent=4)

    # env = CartPoleLeftRightEnv(gym.make("CartPole-v1"))
    env = gym.make("CartPole-v1")

    print(f"env.observation_space.size(): {env.observation_space.shape}")
    print(f"env.action_space.n: {env.action_space.n}")

    # Loss function for optimization - Mean Squared Error loss
    mse_loss = nn.MSELoss()

    # Create an additional CartPole-v1 environment for evaluation
    # eval_env = CartPoleLeftRightEnv(gym.make("CartPole-v1"))
    eval_env = gym.make("CartPole-v1")

    # The DoubleDQL class object
    dqn = DoubleDQL(
        train_env=env, eval_env=eval_env, loss_fn=mse_loss, lr=LR, gamma=GAMMA,
        epsilon=EPSILON, replay_mem_size=REPLAY_MEM_SIZE, per_alpha=PER_ALPHA,
        per_beta=PER_BETA, hl1_size=HL1_SIZE, hl2_size=HL2_SIZE, device=device)

    # Reset Target-DQN parameters to Main-DQN
    dqn.update_target_dqn()

    # Train the agent
    dqn.train(max_train_steps=args.max_train_steps,
              init_training_period=INITIAL_PERIOD,
              main_update_period=MAIN_UPDATE_PERIOD,
              target_update_period=TARGET_UPDATE_PERIOD,
              batch_size=BATCH_SIZE,
              epsilon_decay=EPSILON_DECAY,
              epsilon_min=EPSILON_MIN,
              evaluation_freq=EVALUATION_FREQUENCY,
              path=training_dir,
              show_plot=args.plot)


if __name__ == '__main__':
    # Environment
    # -----------
    SEED = 0

    # Adam
    # ------
    LR = 0.001
    BETA_1 = 0.9
    BETA_2 = 0.999

    # TDL
    # -----
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_DECAY = 0.99999
    EPSILON_MIN = 0.01
    PER_ALPHA = 0.6
    PER_BETA = 0.4

    # NN
    # ----
    HL1_SIZE = 48
    HL2_SIZE = 48
    BATCH_SIZE = 128

    # DQL
    # -----
    REPLAY_MEM_SIZE = 100_000
    INITIAL_PERIOD = 5000
    MAIN_UPDATE_PERIOD = 4
    TARGET_UPDATE_PERIOD = 100
    EPISODE_LENGTH = 500

    # Logging
    # ---------
    EVALUATION_FREQUENCY = 200
    RECORD = True
    RECORD_PATH = 'videos/images/'

    parser = argparse.ArgumentParser(description='DDQL for CartPole-v1')
    parser.add_argument('--max_train_steps', type=int, default=1_500_000,
                        help='maximum number of training steps (default: 1_500_000)')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='plot learning curve (default: False)')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Verbose mode (default: False)')
    parser_args = parser.parse_args()

    main(parser_args)
