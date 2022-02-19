import argparse
import logging
import os

import gym
import pfrl
import torch
from torch import nn

import wandb
from utils import EvalWandbHook


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--env_id", type=str, default="Swimmer-v3", help="Gym Env ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount Rate")

    args = parser.parse_args()

    wandb.init(project="pfrl", tags=[args.env_id], group=f"{args.env_id}_PPO")
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

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(obs_space.low.size, clip_threshold=5)

    obs_size = obs_space.low.size
    action_size = action_space.low.size
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_size),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    # While the original paper initialized weights by normal distribution,
    # we use orthogonal initialization as the latest openai/baselines does.
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1)

    # Combine a policy and a value function into a single model
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    agent = pfrl.agents.PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=0 if torch.cuda.is_available() else -1,
        update_interval=2048,
        minibatch_size=64,
        epochs=10,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=args.gamma,
        lambd=0.95,
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
