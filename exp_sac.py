import argparse
import functools
import logging
import os

import gym
import numpy as np
import pfrl
import torch
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda
from torch import distributions, nn

import wandb
from utils import EvalWandbHook


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--env_id", type=str, default="Swimmer-v3", help="Gym Env ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount Rate")

    args = parser.parse_args()

    wandb.init(project="pfrl", tags=[args.env_id], group=f"{args.env_id}_SAC")
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

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )

    policy = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_size * 2),
        Lambda(squashed_diagonal_gaussian_head),
    )
    torch.nn.init.xavier_uniform_(policy[0].weight)
    torch.nn.init.xavier_uniform_(policy[2].weight)
    torch.nn.init.xavier_uniform_(policy[4].weight, gain=1.0)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        torch.nn.init.xavier_uniform_(q_func[1].weight)
        torch.nn.init.xavier_uniform_(q_func[3].weight)
        torch.nn.init.xavier_uniform_(q_func[5].weight)
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(10**6)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=args.gamma,
        replay_start_size=10000,
        gpu=0 if torch.cuda.is_available() else -1,
        minibatch_size=256,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer_lr=3e-4,
    )

    outdir = os.path.join(wandb.run.dir, "model")

    pfrl.experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        eval_env=make_env(test=True),
        outdir=outdir,
        steps=10e6,
        eval_n_steps=None,
        eval_n_episodes=10,
        eval_interval=5000,
        train_max_episode_len=timestep_limit,
        evaluation_hooks=[EvalWandbHook()],
    )


if __name__ == "__main__":
    main()
