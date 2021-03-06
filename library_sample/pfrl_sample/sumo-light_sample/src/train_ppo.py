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

import experiments_sumo
from experiments_sumo import eval_sumo


def main():
    import logging

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        default="sumo-light-v0",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument("--kwargs-learn", type=dict, default=kwargs_learn)
    parser.add_argument(
        "--eval-env",
        type=str,
        default="sumo-fix-v0",
        help="OpenAI Gym MuJoCo env to evaluate agent.",
    )
    parser.add_argument("--kwargs-eval", type=dict, default=kwargs_eval)
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
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    # monitor hyper param
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--video-freq", type=int, default=1, help="record video freaquency"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")

    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )

    # nn hyper param
    parser.add_argument(
        "--hidden-size", type=int, default=32, help="number of hidden neural nerwork"
    )

    # optimizer hipery param
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--amsgrad", action="store_true", default=False)

    # observation normalize hyper param
    parser.add_argument("--obs-normalize-batch-axis", type=int, default=0)
    parser.add_argument("--obs-normalize-eps", type=float, default=1e-2)
    parser.add_argument("--obs-normalize-until", type=int, default=None)
    parser.add_argument("--obs-normalize-clip-threshold", type=int, default=5)

    # PPO hyper param (except model, optimizer, obs_normalizer)
    parser.add_argument(
        "--gpu", type=int, default=device, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lambd", type=float, default=0.97)
    parser.add_argument("--value-func-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="Interval in timesteps between model updates.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to update model for per PPO iteration.",
    )
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--clip-eps-vf", type=float, default=None)
    parser.add_argument("--standardize-advantages", action="store_false", default=True)
    parser.add_argument("--recurrent", action="store_true", default=False)
    parser.add_argument("--max-recurrent-sequence-len", type=int, default=None)
    parser.add_argument("--act-deterministically", action="store_true", default=False)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--value-stats-window", type=int, default=1000)
    parser.add_argument("--entropy-stats-window", type=int, default=1000)
    parser.add_argument("--value-loss-stats-window", type=int, default=100)
    parser.add_argument("--policy-loss-stats-window", type=int, default=100)

    # train or eval hyper param
    parser.add_argument(
        "--steps",
        type=int,
        default=10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument("--eval-n-steps", type=int, default=None)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=1000,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--eval-max-episode-len", type=int, default=None)
    parser.add_argument("--return-window-size", type=int, default=100)
    parser.add_argument("--successful-score", type=float, default=None)
    parser.add_argument("--save-best-so-far-agent", action="store_false", default=True)
    parser.add_argument("--use-tensorboard", action="store_false", default=True)

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
    eval_sumo.save_args_compact(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    log_file_name = os.path.join(args.outdir, "log.log")
    logging.basicConfig(level=args.log_level, filename=log_file_name)

    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        kwargs_tmp = args.kwargs_eval if test else args.kwargs_learn
        env_name = args.eval_env if test else args.env
        kwargs = copy.deepcopy(kwargs_tmp)
        kwargs["seed"] = env_seed
        kwargs["label"] = kwargs_tmp["label"] + str(process_seed)
        if args.monitor:
            kwargs["mode"] = "gui"
        env = gym.make(env_name, **kwargs)
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, video_callable=(lambda e: e % args.video_freq == 0)
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
        obs_space.low.size,
        batch_axis=args.obs_normalize_batch_axis,
        eps=args.obs_normalize_eps,
        until=args.obs_normalize_until,
        clip_threshold=args.obs_normalize_clip_threshold,
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
        gamma=args.gamma,
        lambd=args.lambd,
        value_func_coef=args.value_func_coef,
        entropy_coef=args.entropy_coef,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps=args.clip_eps,
        clip_eps_vf=args.clip_eps_vf,
        standardize_advantages=args.standardize_advantages,
        recurrent=args.recurrent,
        max_recurrent_sequence_len=args.max_recurrent_sequence_len,
        act_deterministically=args.act_deterministically,
        max_grad_norm=args.max_grad_norm,
        value_stats_window=args.value_stats_window,
        entropy_stats_window=args.entropy_stats_window,
        value_loss_stats_window=args.value_loss_stats_window,
        policy_loss_stats_window=args.policy_loss_stats_window,
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
        #     n_steps=args.eval_n_steps,
        #     n_episodes=args.eval_n_runs,
        #     max_episode_len=timestep_limit,
        # )
        eval_stats = eval_sumo.eval_performance(
            env=env,
            agent=agent,
            n_steps=args.eval_n_steps,
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
        #     steps=args.steps,
        #     eval_n_steps=args.eval_n_steps,
        #     eval_interval=args.eval_interval,
        #     outdir=args.outdir,
        #     eval_n_episodes=args.eval_n_runs,
        #     checkpoint_freq=args.checkpoint_freq,
        #     max_episode_len=timestep_limit,
        #     step_offset=args.step_offset,
        #     eval_max_episode_len=args.eval_max_episode_len,
        #     return_window_size=args.return_window_size,
        #     eval_env=make_batch_env(True),
        #     log_interval=args.log_interval,
        #     successful_score=args.successful_score,
        #     save_best_so_far_agent=args.save_best_so_far_agent,
        # )
        experiments_sumo.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            eval_n_episodes=args.eval_n_runs,
            checkpoint_freq=args.checkpoint_freq,
            max_episode_len=timestep_limit,
            step_offset=args.step_offset,
            eval_max_episode_len=args.eval_max_episode_len,
            return_window_size=args.return_window_size,
            eval_env=make_batch_env(True),
            log_interval=args.log_interval,
            successful_score=args.successful_score,
            save_best_so_far_agent=args.save_best_so_far_agent,
            use_tensorboard=args.use_tensorboard,
        )


if __name__ == "__main__":
    kwargs_learn = {
        "mode": "cui",
        "carnum": 1,
        "label": "learn",
        "road_freq": 1000,
        "road_ratio": 0.2,
        "step_length": 1,
        "max_length": 100.0,
        "is_length": True,
        "is_random_route": False,
    }
    kwargs_eval = {
        "mode": "cui",
        "carnum": 1,
        "label": "eval",
        "step_length": 1,
        "route_length": [1000.0, 2000.0],
    }
    device = 0 if torch.cuda.is_available() else -1
    main()
