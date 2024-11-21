# DDQN_SingleAgent

Double Deep Q-Network PyTorch implementation

## Installation

[Poetry](https://python-poetry.org/) is used for dependency management. Install Poetry based on your OS:

<details>
  <summary>Windows:</summary>
  
  ```bash
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
  ```
  Note: If you have installed Python through the Microsoft Store, replace py with python in the command above.
</details>


<details>
  <summary>Linux/MacOS:</summary>
  
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
</details>


To install all the dependencies, please enter the following from the project's root directory:

```bash
poetry install

```

Then enter the virtual environment:

```bash
poetry shell

```

To train the sample Gym environment

```bash
python dql_train.py --max_train_steps [NO. OF MAX TRAINING STEPS]

```

To evaluate a trained policy

```bash
python dql_eval.py --path [PATH/TO/TRAINED/MODEL] --render

```
