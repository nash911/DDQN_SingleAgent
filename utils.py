import torch
import matplotlib.pyplot as plt

from typing import Sequence


def init_layer(layer, bias_const=0.0):
    """
    Initialize a Linear or Convolutional layer.

    Parameters
    ----------
    layer : torch.nn.Module
        A Linear or Convolutional layer.
    bias_const : float
        A constant to initialize the bias with.

    Returns
    -------
    torch.nn.Module
        The initialized layer.
    """

    torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def plot_all(train_t: Sequence[int], train_loss: Sequence[float],
             train_reward: Sequence[float], train_episode_t: Sequence[int],
             eval_reward: Sequence[float], eval_episode_t: Sequence[int],
             epsilons: Sequence[float], text: str, path: str = None,
             show: bool = False) -> None:
    """
       Method for plotting learning curves during policy training.

       Training loss, normalized episodic training and evaluation rewards, and
       percentage of transitions with goal = 'Right' are plotted.

       Parameters
       ----------
       train_t : Sequence[int]
           A list of time-steps pertaining to training-loss.
       train_loss : Sequence[float]
           A list of training-losses.
       train_reward : Sequence[float]
           A list of training rewards.
       train_episode_t : Sequence[int]
           A list of time-steps pertaining to the training rewards.
       eval_reward : Sequence[float]
           A list of evaluation rewards.
       eval_episode_t: Sequence[int]
           A list of time-steps pertaining to the evaluation rewards.
       epsilons: Sequence[float]
           A list of epsilon values.
       text : str
           Text to be displayed on the plot.
       path : str
           Path to save the plot.
       show : bool
           Flag to display the plot.
    """

    fig, axs = plt.subplots(3, figsize=(10, 14), sharey=False, sharex=False)

    # Training Loss plot
    axs[0].plot(train_t, train_loss, color='red', label='Train')
    axs[0].set(title='Training Loss')
    axs[0].set(ylabel='Loss')
    axs[0].set(xlabel='Time-step')
    axs[0].legend(loc='upper left')

    # Normalized episodic reward of the policy during training and evaluation
    axs[1].plot(train_episode_t, train_reward, color='red', label='Train')
    axs[1].plot(eval_episode_t, eval_reward, color='blue', label='Evaluation')
    axs[1].set(title='Normalized Episode Reward')
    axs[1].set(ylabel='Normalized Reward')
    axs[1].set(xlabel='Episode')
    axs[1].legend(loc='upper left')

    # Epsilon plot
    axs[2].plot(train_t, epsilons, color='green', label='Epsilon')
    axs[2].set(title='Epsilon')
    axs[2].set(ylabel='Epsilon')
    axs[2].set(xlabel='Time-step')
    axs[2].legend(loc='upper right')

    # Add text to the plot
    if text is not None:
        x_min, x_max = axs[0].get_xlim()
        y_min, y_max = axs[0].get_ylim()
        axs[0].text(x_min + 0.5, y_max * 1.1, text, fontsize=12)

    if path is not None:
        plt.savefig(path + "plots/learning_curves.png")

    if show:
        plt.show(block=False)
        plt.pause(0.01)

    # Close the plot
    plt.close()
