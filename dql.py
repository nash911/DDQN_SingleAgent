import numpy as np
import time

from collections import OrderedDict
from typing import Sequence, Tuple
from gymnasium import Env

import torch
import torch.nn as nn
import torch.optim as optim

from utils import init_layer, plot_all


class ReplayMemory:
    """
        A class for storing and sampling transitions for Experience Replay.

        This class creates a memory buffer of a predetermined size for storing
        and sampling batch-sized transitions {sₜ, aₜ, rₜ, sₜ₊₁} during training.
    """

    def __init__(self, mem_size: int, state_shape: Tuple[int, ...],
                 action_shape: Tuple[int], alpha: float, beta: float, device):
        """
           The init() method creates and initializes memory buffers for storing
           transitions. It also initializes counters for indexing the array for
           roll-over transition insertion and sampling.

           Parameters
           ----------
           mem_size : int
               The maximum size of the replay memory buffer.
           state_shape : Tuple
               The shape of the state tensor.
           action_shape : Tuple
               The shape of the action tensor.
           alpha : float
               The alpha parameter for prioritized experience replay.
           beta : float
               The beta parameter for prioritized experience replay.
           device : torch.device
               The device to run the training on.
        """

        self.mem_size = mem_size
        self.state_shape = state_shape
        self.action_shape = action_shape
        self._alpha = alpha
        self._beta = beta
        self._max_td_delta = 1e+1  # High initial TD error value to ensure that
                                   # first transitions are sampled for training
        self._max_td_delta_updated = False
        self.device = device
        self.mem_count = 0
        self.current_index = 0

        self.states = OrderedDict()
        self.actions = OrderedDict()
        self.rewards = OrderedDict()
        self.terminals = OrderedDict()
        self.action_masks = OrderedDict()

        self.states = torch.zeros((self.mem_size, *self.state_shape),
                                  dtype=torch.float32).to(device=self.device)
        self.actions = torch.zeros(
            (mem_size,), dtype=torch.int64).to(device=self.device)
        self.rewards = torch.zeros(
            (mem_size,), dtype=torch.float32).to(device=self.device)
        self.terminals = torch.zeros(
            (mem_size,), dtype=torch.int).to(device=self.device)
        self.action_masks = torch.zeros(
            (mem_size, action_shape[0]), dtype=torch.int).to(device=self.device)
        self.td_delta = torch.ones(
            (mem_size,), dtype=torch.float32).to(device=self.device)

    def add(self, state: torch.tensor, action: torch.tensor, reward: float,
            terminal: bool, action_mask: torch.tensor) -> None:
        """
           Method for inserting a transition {sₜ, aₜ, rₜ, ɤₜ, sₜ₊₁} to the
           replay memory buffer.

           Parameters
           ----------
           state : tensor
               An array of observations from the environment.
           action : tensor
               The action taken by the agent.
           reward : float
               The observed reward.
           terminal : bool
                A boolean indicating if state sₜ is a terminal state or not.
           action_mask : tensor
               A boolean array indicating the valid actions.
        """

        self.states[self.current_index % self.mem_size] = state
        self.actions[self.current_index % self.mem_size] = action
        self.rewards[self.current_index % self.mem_size] = reward
        self.terminals[self.current_index % self.mem_size] = int(terminal)
        self.action_masks[self.current_index % self.mem_size] = action_mask
        self.td_delta[self.current_index % self.mem_size] = self._max_td_delta

        self.current_index = ((self.current_index + 1) % self.mem_size)
        self.mem_count = max(self.mem_count, self.current_index)

    def sample_batch(self, batch_size: int = 128) -> Sequence[torch.tensor]:
        """
           Method for random sampling transitions {s, a, r, s'} of batch_size
           from the replay memory buffer.

           Parameters
           ----------
           batch_size : int
               Number of transitions to be sampled

           Returns
           -------
           Sequence[torch.tensor]
               A list of torch tensors each containing the sampled states,
               actions, rewards, terminal booleans, and the respective
               next-states (s').
        """
        # Sample transitions from the replay memory buffer based on the
        # priority weights
        with torch.no_grad():
            priority_weights = (
                self.td_delta[:self.mem_count].detach().cpu().numpy())

            # Set the priority weight of the most recent transition to zero, to
            # prevent sampling it
            priority_weights[self.current_index - 1] = 0.0

        # Calculate the probability of sampling each transition based on the
        # priority weights
        p = priority_weights / priority_weights.sum()

        while True:
            sampled_idx = np.random.choice(
                self.mem_count, size=batch_size, replace=False, p=p)

            if (sampled_idx == self.current_index - 1).any():
                # Resample if any sampled transition is the most recently
                # recorded transition
                continue
            break

        # Calculate the importance weights for the sampled transitions
        importance_weights = torch.tensor((1.0 / (
            self.mem_count * priority_weights[sampled_idx])) ** self._beta,
            dtype=torch.float32).to(self.device)

        return (
            self.states[sampled_idx],
            self.actions[sampled_idx],
            self.rewards[sampled_idx],
            self.terminals[sampled_idx],
            self.states[(sampled_idx + 1) % self.mem_count],  # s'
            self.action_masks[(sampled_idx + 1) % self.mem_count],
            importance_weights,
            sampled_idx
        )

    def update_max_td_delta(self, max_td_delta: float) -> None:
        """
           Method for updating the maximum TD error in the replay memory buffer.

           Parameters
           ----------
           max_td_delta : float
               The maximum TD error in the replay memory buffer.
        """
        # if max_td_delta > self._max_td_delta:
        #     print(f"Previous Max TD Error: {self._max_td_delta} -- "
        #           f"New Max TD Error: {max_td_delta}")

        if self._max_td_delta_updated:
            self._max_td_delta = max(max_td_delta, self._max_td_delta)
        else:
            # Update the maximum TD error on the first call
            self._max_td_delta = max_td_delta
            self._max_td_delta_updated = True

    def __len__(self):
        return self.mem_count

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta


class FCAgent(nn.Module):
    """
        A class for the Fully Connected Deep Q-Network (DQN) Agent.

        Parameters
        ----------
        obs_shape : Tuple[int]
            The shape of the observation tensor.
        action_shape : Tuple[int]
            The number of actions in the action space.
        hl1_size : int
            The size of the first hidden layer.
        hl2_size : int
            The size of the second hidden layer.
        device : torch.device
            The device to run the training on.
    """
    def __init__(self, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...],
                 hl1_size: int, hl2_size: int, device):
        super().__init__()

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device

        self.deep_q_network = nn.Sequential(
            init_layer(nn.Linear(self.obs_shape[0], hl1_size)),
            nn.ReLU(),
            init_layer(nn.Linear(hl1_size, hl2_size)),
            nn.ReLU(),
            init_layer(nn.Linear(hl2_size, action_shape[0]))
        )

    def forward(self, x):
        """Forward Pass"""
        out = self.deep_q_network(x)
        return out


class DoubleDQL:
    """
        Double Deep Q-Learning Class.

        This class implements the Double Deep Q-Learning algorithm for training
        an agent in an environment with discrete action space.

        Parameters
        ----------
        train_env : Env
            The environment for training the policy.
        eval_env : Env
            The environment for evaluating the policy.
        loss_fn : torch.nn.loss
            Loss function object for training the main-dqn.
        gamma : float
            Discount parameter ɣ.
        epsilon : float
            Exploration parameter Ɛ.
        lr : float
            Learning rate for the optimizer.
        per_alpha : float
            Alpha parameter for prioritized experience replay.
        per_beta : float
            Beta parameter for prioritized experience replay.
        replay_mem_size : int
            The maximum size of the replay memory buffer.
        device : torch.device
            The device to run the training on.
    """

    def __init__(self, train_env: Env, eval_env: Env, loss_fn=None,
                 gamma: float = 0.99, epsilon: float = 1.0, lr: float = 0.001,
                 per_alpha: float = 0.6, per_beta: float = 0.4,
                 replay_mem_size: int = 200_000, hl1_size: int = 64,
                 hl2_size: int = 64, device: torch.device = torch.device("cpu")
                 ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.device = device

        self.state_shape = train_env.observation_space.shape
        self.action_shape = (train_env.action_space.n,)

        self.main_dqn = FCAgent(
            obs_shape=self.state_shape, action_shape=self.action_shape,
            hl1_size=hl1_size, hl2_size=hl2_size, device=self.device).to(
            self.device)
        self.target_dqn = FCAgent(
            obs_shape=self.state_shape, action_shape=self.action_shape,
            hl1_size=hl1_size, hl2_size=hl2_size, device=self.device).to(
            self.device)

        self.update_target_dqn()

        self.replay_memory = ReplayMemory(
            mem_size=replay_mem_size, state_shape=self.state_shape,
            action_shape=self.action_shape, alpha=per_alpha, beta=per_beta,
            device=device)

        self.optimizer = optim.Adam(self.main_dqn.parameters(), lr=lr)
        self.loss_fn = loss_fn

        self.gamma = gamma
        self.epsilon = epsilon

    def update_target_dqn(self) -> None:
        """
           Method for updating target-dqn parameters to that of the main-dqn.
        """

        self.target_dqn.load_state_dict(self.main_dqn.state_dict())

    def load_main_dqn(self, model_path: str) -> None:
        """
           Method to load main-dqn from saved model file.
        """

        self.main_dqn.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device),
                       weights_only=True))
        self.main_dqn.eval()

    def epsilon_sample(self, q_values: torch.tensor, action_mask: np.array,
                       epsilon: float) -> int:
        """
           Method for sampling from the discrete action space based on Ɛ-greedy
           exploration strategy.

           Parameters
           ----------
           q_values : torch.tensor
               An array of q-values.
           action_mask : np.array
               An array of booleans indicating the valid actions.
           epsilon : float
               The probability of sampling an action from a uniform
               distribution.

           Returns
           -------
           int
               The index of the sampled action.
        """
        if np.random.rand() < epsilon:  # Exploration
            with np.errstate(divide='ignore'):
                action = np.argmax(np.random.rand(self.action_shape[0]) +
                                   np.log(action_mask.cpu().numpy()))
                return int(action)
        else:  # Exploitation
            with np.errstate(divide='ignore'):
                q_values = q_values + torch.log(action_mask)
            return int(torch.argmax(q_values).cpu().numpy())

    def evaluate(self, num_episodes: int = 20, epsilon: float = 0) -> Tuple[
         float, float]:
        """
           Method for evaluating policy during training.

           This method evaluates the current policy for a fixed number of
           episodes (games) and returns the mean normalized episode reward

           Parameters
           ----------
           num_episodes : int
               Number of episodes to evaluate the policy for.
           epsilon : float
               Exploration parameter Ɛ.

           Returns
           -------
           float
               Mean episode reward
           float
               Mean episode length
        """

        # Initialize lists to store normalized episode rewards and
        # episode lengths
        average_episode_reward = list()
        episode_length = list()

        # Evaluate the current policy num_episodes times with different seeds
        # for each episode to get a better estimate of the policy performance
        for episode in range(num_episodes):
            episode_reward = 0
            steps = 0
            done = False

            # Reset the environment and get the initial state and player ID for
            # the first player
            state, _ = self.eval_env.reset(seed=episode)

            while not done:
                # Convert the observation to a torch tensor and add padding
                obs = torch.tensor(state, dtype=torch.float32).to(self.device)

                # Get the action mask for the current player
                action_mask = torch.ones(
                    self.action_shape, dtype=torch.int).to(self.device)

                with torch.no_grad():
                    # Get Q(sₜ,a|θ) ∀ a ∈ A from the main-dqn
                    q_values = self.main_dqn(obs)

                    # From Q(sₜ,a|θ) sample aₜ based on Ɛ-greedy
                    action = self.epsilon_sample(
                        q_values, action_mask, epsilon=epsilon)

                # Step through the environment with action aₜ, receiving
                # reward rₜ, and observing the new state sₜ₊₁
                state, reward, terminated, truncated, info = (
                    self.eval_env.step(int(action)))

                # Get the reward and terminal status of the current episode
                done = terminated or truncated

                episode_reward += reward

                # Increment the number of steps in the current episode
                steps += 1

            # Append the normalized episode reward and episode length to the
            # respective lists
            average_episode_reward.append(episode_reward/500)
            episode_length.append(steps)

        return (float(np.mean(average_episode_reward)),
                float(np.mean(episode_length)))


    def train(self, max_train_steps, init_training_period, main_update_period,
              target_update_period, batch_size=64, epsilon_decay: float = 0.999,
              epsilon_min: float = 0.01, evaluation_freq: int = 1000,
              verbose: bool = False, path: str = None, show_plot: bool = False):
        """
           Method for training the policy based on the DDQL algorithm.

           Parameters
           ----------
           max_train_steps : int
               Maximum number of training steps.
           init_training_period : int
               Number of time steps of recording transitions, before initiating
               policy training.
           main_update_period : int
               Number of time steps between consecutive main-dqn batch updates.
           target_update_period : int
               Number of time steps between consecutive target-dqn updates to
               main-dqn.
           batch_size : int
               Batch size for main-dqn training.
           epsilon_decay : float
               Decay rate for exploration parameter Ɛ.
           epsilon_min : float
               Minimum value for exploration parameter Ɛ.
           evaluation_freq : int
               Number of time steps between policy evaluations during training.
           verbose : bool
               Boolean indicating whether to print training progress.
           path : str
               Path to save the trained model.
           show_plot : bool
               Boolean indicating whether to show the learning curves plot.
        """

        start_time = time.time()

        # Initialize lists for storing learning curve data
        t_list = list()
        train_loss = list()
        train_reward = list()
        train_steps = list()
        train_episode_steps = list()
        train_episodes = list()

        eval_reward = list()
        eval_steps = list()
        eval_episodes = list()
        epsilon = list()

        train_step = 0
        episode_steps = 0
        train_episodes_count = 0
        best_train_reward = 0.0
        best_eval_reward = 0.0
        best_train_model_episode = 0
        saved_model_txt = None
        train_episode_start_idx = 10_000

        # Set the main-dqn to training mode
        self.main_dqn.train()
        self.target_dqn.train()

        # Reset the environment and get the initial state
        state, _ = self.train_env.reset(seed=train_episode_start_idx +
                                             train_episodes_count)

        # The state -> action -> reward, next-state loop for policy training
        while train_episodes_count < max_train_steps:
            # Convert the observation to a torch tensor
            obs = torch.tensor(state, dtype=torch.float32).to(self.device)

            # Set action mask as all ones
            # TODO: Replace with the actual action mask
            action_mask = torch.ones(self.action_shape, dtype=torch.int).to(
                self.device)

            with torch.no_grad():
                # Get Q(sₜ,a|θ) ∀ a ∈ A from the main-dqn
                q_values = self.main_dqn(obs)

                # From Q(sₜ,a|θ) sample aₜ based on Ɛ-greedy
                action = self.epsilon_sample(
                    q_values, action_mask, epsilon=self.epsilon)

            # Step through the environment with action aₜ, receiving reward rₜ,
            # and observing the new state sₜ₊₁
            state, reward, terminated, truncated, info = self.train_env.step(int(action))

            # Check if is the end of the episode
            done = terminated or truncated

            # Save the transition {sₜ, aₜ, rₜ, sₜ₊₁} to the Replay Memory
            self.replay_memory.add(obs, action, float(reward), done,
                                   action_mask)

            # Count the number of steps in the current episode
            episode_steps += 1

            # Increment the training step count
            train_step += 1

            # if train_step % 1000 == 0:
            #     end_time = time.time()
            #     print(f"Step: {train_step//1000} -- Time: "
            #           f"{np.round(end_time - start_time, 4)}s")
            #     start_time = time.time()

            if self.replay_memory.__len__() > init_training_period:
                # Decay exploration parameter Ɛ over time to a minimum of
                # EPSILON_MIN: Ɛₜ = (Ɛ-decay)ᵗ
                if self.epsilon > epsilon_min:
                    self.epsilon *= epsilon_decay

                # Main-DQN batch update
                if train_step % main_update_period == 0:
                    # From Replay Memory Buffer, uniformly sample a batch of
                    # transitions
                    (states, actions, rewards, terminals, state_primes,
                     action_mask_primes, importance_weights, sampled_idx) = (
                        self.replay_memory.sample_batch(batch_size=batch_size))

                    with torch.no_grad():
                        # Best next action estimate of the main-dqn, for the
                        # sampled batch:
                        # aⱼ = argmaxₐ Q(sₜ₊₁, a|θ), a ∈ A
                        best_action = torch.argmax(
                            self.main_dqn(state_primes) + torch.log(
                                action_mask_primes), axis=-1, keepdims=True)

                        target_all_q = self.target_dqn(state_primes)

                        # Target q value for the sampled batch:
                        # yⱼ = rⱼ, if sⱼ' is a terminal-state
                        # yⱼ = rⱼ + ɣ Q(sⱼ',aⱼ|θ⁻), otherwise.
                        target_q = \
                            rewards + (self.gamma * torch.gather(
                                input=target_all_q, dim=1,
                                index=best_action).reshape(-1) *
                                       (1 - terminals))

                    # Predicted q-value of the main-dqn, for the sampled batch
                    # Q(sⱼ,aⱼ|θ)
                    pred_q = self.main_dqn(states)
                    pred_q_a = torch.gather(
                        input=pred_q, dim=1, index=actions.reshape(-1, 1)
                    ).reshape(-1)

                    # Calculate loss:
                    # L(θ) = 𝔼[(Q(s,a|θ) - y)²]
                    loss = self.loss_fn(pred_q_a * importance_weights,
                                        target_q * importance_weights)

                    # Calculate the gradient of the loss w.r.t main-dqn
                    # parameters θ
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Update main-dqn parameters θ:
                    self.optimizer.step()

                    # Update the sampled transitions' TD errors in the replay
                    # memory buffer
                    # δⱼ = |Q(sⱼ,aⱼ|θ) - yⱼ|ᵅ
                    td_delta = ((torch.abs(target_q - pred_q_a) + 1e-5) **
                                self.replay_memory.alpha)
                    self.replay_memory.td_delta[sampled_idx] = td_delta

                    # Update the maximum TD error in the replay memory buffer
                    self.replay_memory.update_max_td_delta(
                        torch.max(td_delta).item())

                    # For plotting
                    t_list.append(train_step)
                    train_loss.append(loss.detach().cpu().numpy())
                    epsilon.append(self.epsilon)

                # Target-DQN update period
                if train_step % target_update_period == 0:
                    # Reset Target-DQN to Main-DQN
                    self.update_target_dqn()

            # Reset the environment and store normalized episode reward on
            # completion of the current episode
            if done:
                # Increment episode count
                train_episodes_count += 1

                # Policy evaluation period
                if train_episodes_count % np.abs(evaluation_freq) == 0:
                    # Evaluate the current policy
                    self.main_dqn.eval()
                    (mean_eval_reward, mean_eval_steps) = self.evaluate(
                        num_episodes=20, epsilon=0)

                    # Save evaluation metrics for plotting
                    eval_episodes.append(train_episodes_count)
                    eval_reward.append(mean_eval_reward)
                    eval_steps.append(mean_eval_steps)

                    if verbose:
                        print(
                            f"{train_episodes_count} -- Mean Eval Reward: " +
                            f"{mean_eval_reward} -- Mean Eval Steps: " +
                            f"{mean_eval_steps}")

                    # # Save a snapshot of the best policy (main-dqn) based on the
                    # # training results
                    # if train_rl_win >= best_train_reward:
                    #     torch.save(self.main_dqn.state_dict(), path + (
                    #         'models/best_train_policy.pth'))
                    #     best_train_reward = train_rl_win
                    #     best_train_model_episode = train_episodes_count

                    # Save a snapshot of the best policy (main-dqn) based on the
                    # evaluation results
                    if mean_eval_reward >= best_eval_reward:
                        torch.save(self.main_dqn.state_dict(), path + (
                            'models/best_eval_policy_' +
                            f'{train_episodes_count}.pth'))
                        best_eval_reward = mean_eval_reward
                        saved_model_txt = (
                            "Best Model Saved @ Episode " +
                            f"{train_episodes_count} with Eval reward: " +
                            f"{np.round(mean_eval_reward, 4)}")
                        print("\n" + saved_model_txt + "\n")

                    # Plot loss, rewards, and epsilon
                    plot_all(
                        t_list, train_loss, train_reward, train_episodes,
                        eval_reward, eval_episodes, epsilon, show=show_plot,
                        path=path, text=saved_model_txt)

                    # Reset the main-dqn to training mode
                    self.main_dqn.train()

                # Reset the environment and get the initial state and player
                # ID for the first player
                state, _ = self.train_env.reset(seed=train_episode_start_idx +
                                                     train_episodes_count)

                train_episode_steps.append(episode_steps)
                episode_steps = 0

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time - start_time))
        input("Completed training.\nPress Enter to exit")
