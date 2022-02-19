import argparse
import functools
import logging
import os

import gym
import numpy as np
import pfrl
import torch
from pfrl import explorers, replay_buffers
from pfrl.nn import BoundByTanh, ConcatObsAndAction
from pfrl.policies import DeterministicHead
from torch import nn

import wandb
from utils import EvalWandbHook


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--env_id", type=str, default="Swimmer-v3", help="Gym Env ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount Rate")

    args = parser.parse_args()

    wandb.init(project="pfrl", tags=[args.env_id], group=f"{args.env_id}_DDPG")
    wandb.config.update(args)

    logging.basicConfig(level=logging.INFO)

    pfrl.utils.set_random_seed(args.seed)

    def make_env(test):
        env = gym.make(args.env_id)
        # Use different random seeds for train and test envs
        env_seed = 2**32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    sample_env = make_env(test=False)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size
    q_func = nn.Sequential(
        ConcatObsAndAction(),
        nn.Linear(obs_size + action_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, 1),
    )
    policy = nn.Sequential(
        nn.Linear(obs_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, action_size),
        BoundByTanh(low=action_space.low, high=action_space.high),
        DeterministicHead(),
    )

    opt_a = torch.optim.Adam(policy.parameters())
    opt_c = torch.optim.Adam(q_func.parameters())

    rbuf = replay_buffers.ReplayBuffer(10**6)

    explorer = explorers.AdditiveGaussian(scale=0.1, low=action_space.low, high=action_space.high)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    agent = pfrl.agents.DDPG(
        policy,
        q_func,
        opt_a,
        opt_c,
        rbuf,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=10000,
        target_update_method="soft",
        target_update_interval=1,
        update_interval=1,
        soft_update_tau=5e-3,
        n_times_update=1,
        gpu=0 if torch.cuda.is_available() else -1,
        minibatch_size=100,
        burnin_action_func=burnin_action_func,
    )

    outdir = os.path.join(wandb.run.dir, "model")

    pfrl.experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        eval_env=make_env(test=True),
        outdir=outdir,
        steps=1e6,
        eval_n_steps=None,
        eval_n_episodes=10,
        eval_interval=5000,
        train_max_episode_len=timestep_limit,
        evaluation_hooks=[EvalWandbHook()],
    )


if __name__ == "__main__":
    main()
