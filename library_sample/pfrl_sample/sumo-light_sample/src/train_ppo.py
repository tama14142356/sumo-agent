"""A training script of PPO on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1709.06560 as much
as possible.
"""
import argparse
import functools
import os

import gym
import gym.spaces
import gym_sumo
import numpy as np
import copy
import torch
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO

import train_agent_batch_sumo
import eval_sumo


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=device, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="sumo-light-v0",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=100,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="Interval in timesteps between model updates.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to update model for per PPO iteration.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument(
        "--video-freq", type=int, default=1, help="record video freaquency"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=32, help="number of hidden neural nerwork"
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lambd", type=float, default=0.97)
    # optimizer hipery param
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--rbuf-capacity", type=float, default=5 * 10 ** 5)
    parser.add_argument("--amsgrad", type=bool, default=False)
    args = parser.parse_args()

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    eval_sumo.save_sumo_version(args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    log_file_name = os.path.join(args.outdir, "log.log")
    logging.basicConfig(level=args.log_level, filename=log_file_name)

    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        kwargs_tmp = kwargs_eval if test else kwargs_learn
        kwargs = copy.deepcopy(kwargs_tmp)
        kwargs["seed"] = env_seed
        kwargs["label"] = kwargs_tmp["label"] + str(process_seed)
        if args.monitor:
            kwargs["mode"] = "gui"
        env = gym.make(args.env, **kwargs)
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, (lambda e: e % args.video_freq)
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

    # Only for getting timesteps, and obs-action spaces
    sample_env = gym.make(args.env, **kwargs_learn)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)
    sample_env.close()

    # assert isinstance(action_space, gym.spaces.Box)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5
    )

    obs_size = obs_space.low.size
    hidden_size = args.hidden_size
    if isinstance(action_space, gym.spaces.Box):
        action_size = action_space.low.size
        policy = torch.nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )
    else:
        policy = torch.nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            # nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, action_space.n),
            pfrl.policies.SoftmaxCategoricalHead(),
        )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
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

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )

    agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=args.gamma,
        lambd=args.lambd,
    )

    if args.load or args.load_pretrained:
        if args.load_pretrained:
            raise Exception("Pretrained models are currently unsupported.")
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(utils.download_model("PPO", args.env, model_type="final")[0])

    if args.demo:
        env = make_batch_env(True)
        # eval_stats = experiments.eval_performance(
        #     env=env,
        #     agent=agent,
        #     n_steps=None,
        #     n_episodes=args.eval_n_runs,
        #     max_episode_len=timestep_limit,
        # )
        eval_stats = eval_sumo.eval_performance(
            env=env,
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
        env.close()
    else:
        # experiments.train_agent_batch_with_evaluation(
        #     agent=agent,
        #     env=make_batch_env(False),
        #     eval_env=make_batch_env(True),
        #     outdir=args.outdir,
        #     steps=args.steps,
        #     eval_n_steps=None,
        #     eval_n_episodes=args.eval_n_runs,
        #     eval_interval=args.eval_interval,
        #     log_interval=args.log_interval,
        #     max_episode_len=timestep_limit,
        #     save_best_so_far_agent=False,
        # )
        train_agent_batch_sumo.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=True,
        )


if __name__ == "__main__":
    kwargs_learn = {"mode": "cui", "carnum": 1, "label": "learn", "step_length": 1}
    kwargs_eval = {"mode": "cui", "carnum": 1, "label": "eval", "step_length": 1}
    device = 0 if torch.cuda.is_available() else -1
    main()
