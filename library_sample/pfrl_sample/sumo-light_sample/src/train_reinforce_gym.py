"""An example of training a REINFORCE agent against OpenAI Gym envs.

This script is an example of training a REINFORCE agent against OpenAI Gym
envs. Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_reinforce_gym.py

To solve InvertedPendulum-v1, run:
    python train_reinforce_gym.py --env InvertedPendulum-v1
"""
import argparse
import os
import copy

import gym
import gym_sumo
import gym.spaces
import torch
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.policies import GaussianHeadWithFixedCovariance, SoftmaxCategoricalHead

from experiments_sumo import eval_sumo


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="sumo-light-v0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gpu", type=int, default=device)
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10 ** 5)
    parser.add_argument("--eval-interval", type=int, default=10 ** 4)
    parser.add_argument("--eval-n-runs", type=int, default=100)
    parser.add_argument("--reward-scale-factor", type=float, default=1e-2)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    eval_sumo.save_sumo_version(args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    log_file_name = os.path.join(args.outdir, "log.log")
    logging.basicConfig(level=args.log_level, filename=log_file_name)

    def make_env(test):
        kwargs = copy.deepcopy(kwargs_eval if test else kwargs_learn)
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        kwargs["seed"] = env_seed
        if args.monitor:
            kwargs["mode"] = "gui"
        env = gym.make(args.env, **kwargs)
        # Use different random seeds for train and test envs
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        return env

    train_env = make_env(test=False)
    timestep_limit = train_env.spec.max_episode_steps
    obs_space = train_env.observation_space
    action_space = train_env.action_space

    obs_size = obs_space.low.size
    hidden_size = 200
    # Switch policy types accordingly to action space types
    if isinstance(action_space, gym.spaces.Box):
        model = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, action_space.low.size),
            GaussianHeadWithFixedCovariance(0.3),
        )
    else:
        model = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, action_space.n),
            SoftmaxCategoricalHead(),
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    agent = pfrl.agents.REINFORCE(
        model,
        opt,
        gpu=args.gpu,
        beta=args.beta,
        batchsize=args.batchsize,
        max_grad_norm=1.0,
    )
    if args.load:
        agent.load(args.load)

    eval_env = make_env(test=True)

    if args.demo:
        # eval_stats = experiments.eval_performance(
        #     env=eval_env,
        #     agent=agent,
        #     n_steps=None,
        #     n_episodes=args.eval_n_runs,
        #     max_episode_len=timestep_limit,
        # )
        eval_stats = eval_sumo.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=train_env,
            eval_env=eval_env,
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            train_max_episode_len=timestep_limit,
            use_tensorboard=True,
        )


if __name__ == "__main__":
    kwargs_learn = {"mode": "cui", "carnum": 1, "label": "learn", "step_length": 1}
    kwargs_eval = {"mode": "cui", "carnum": 1, "label": "eval", "step_length": 1}
    device = 0 if torch.cuda.is_available() else -1
    main()
