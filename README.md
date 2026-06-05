# Advanced Deep Reinforcement Learning: DQN to SAC

A progressive series of Jupyter notebooks implementing deep RL algorithms from scratch using PyTorch and PyTorch Lightning — starting with vanilla DQN and building up to Soft Actor-Critic and Hindsight Experience Replay.

## Notebooks

| # | Notebook | Algorithm | Environment |
|---|----------|-----------|-------------|
| 1 | [DQN with PyTorch Lightning](2_dqn_pytorch_lightning.ipynb) | Deep Q-Network (DQN) | LunarLander-v2 |
| 2 | [Hyperparameter Tuning](3_hyperparameter_tuning.ipynb) | DQN + Optuna | LunarLander-v2 |
| 3 | [Normalized Advantage Function](4_normalized_advantage_function.ipynb) | NAF (continuous DQN) | LunarLanderContinuous-v2 |
| 4 | [DDPG](5_deep_deterministic_policy_gradient.ipynb) | Deep Deterministic Policy Gradient | Ant (Brax) |
| 5 | [TD3](6_twin_delayed_ddpg.ipynb) | Twin Delayed DDPG | Ant (Brax) |
| 6 | [SAC](7_soft_actor_critic.ipynb) | Soft Actor-Critic | FetchReachDense-v1 |
| 7 | [HER](8_hindsight_experience_replay.ipynb) | Hindsight Experience Replay + SAC | FetchReach-v1 |
| — | [Rainbow DQN](Rainbow_DQN.ipynb) | Rainbow (C51 + Noisy + Dueling + PER + N-step + Double) | Qbert (Atari) |

## Algorithms covered

**Value-based (discrete actions)**
- **DQN** — experience replay, target network, epsilon-greedy exploration
- **Rainbow DQN** — combines six improvements: distributional RL (C51), noisy networks, dueling architecture, prioritized experience replay, n-step returns, and double DQN
- **Hyperparameter tuning** — automated search over DQN hyperparameters using Optuna

**Value-based (continuous actions)**
- **NAF** — Normalized Advantage Function; extends Q-learning to continuous action spaces without a separate policy network

**Actor-Critic (continuous actions)**
- **DDPG** — deterministic policy gradient with an actor-critic architecture; trained on vectorized Brax physics environments
- **TD3** — addresses DDPG overestimation with twin critics, delayed policy updates, and target policy smoothing
- **SAC** — maximum-entropy RL with a stochastic policy; balances exploration and exploitation via an entropy bonus
- **HER** — Hindsight Experience Replay; enables SAC to learn from sparse-reward goal-conditioned tasks by relabelling failed trajectories

## Tech stack

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/) 1.6
- [OpenAI Gym](https://www.gymlibrary.dev/) 0.23 (Box2D, Atari, Robotics)
- [Brax](https://github.com/google/brax) 0.10 — GPU-accelerated physics for DDPG/TD3
- [Optuna](https://optuna.org/) — hyperparameter optimisation
- TensorBoard — training metrics

## Setup

```bash
# Core dependencies (notebooks install their own extras via pip)
pip install torch pytorch-lightning==1.6.0 gym optuna

# Atari (Rainbow DQN)
pip install gym[atari,accept-rom-license]==0.23.1

# Box2D (DQN / NAF / hyperparameter tuning)
pip install swig gym[box2d]==0.23.1

# Robotics envs (SAC / HER)
pip install gym[robotics]

# GPU-accelerated physics (DDPG / TD3)
pip install brax==0.10.5
```

Each notebook also uses `pyvirtualdisplay` for headless rendering and `moviepy` for recording episode videos. On a headless machine:

```bash
apt-get install -y xvfb
pip install pyvirtualdisplay
```

## Running

Open any notebook in Jupyter and run all cells top-to-bottom. Training progress is logged to TensorBoard:

```bash
tensorboard --logdir lightning_logs/
```

Episode videos are saved to a `videos/` folder inside the working directory and can be replayed with the `display_video()` helper at the bottom of each notebook.
