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
    parser.add_argument("--group", default=None, help="WandB Group ID")

    args = parser.parse_args()

    wandb.init(project="pfrl", tags=[args.env_id], group=args.group)
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

    obs_normalizer = pfrl.nn.EmpiricalNormalization(obs_space.low.size, clip_threshold=5)
    obs_size = obs_space.low.size
    action_size = action_space.low.size

    policy = nn.Sequential(
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

    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1e-2)

    vf_opt = torch.optim.Adam(vf.parameters())

    agent = pfrl.agents.TRPO(
        policy=policy,
        vf=vf,
        vf_optimizer=vf_opt,
        obs_normalizer=obs_normalizer,
        gpu=0 if torch.cuda.is_available() else -1,
        update_interval=5000,
        max_kl=0.01,
        conjugate_gradient_max_iter=20,
        conjugate_gradient_damping=1e-1,
        gamma=args.gamma,
        lambd=0.97,
        vf_epochs=5,
        entropy_coef=0,
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
