"""An example of training Double DQN against OpenAI Gym Envs.

This script is an example of training a Double DQN agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported. For continuous action
spaces, A NAF (Normalized Advantage Function) is used to approximate Q-values.

To solve CartPole-v0, run:
    python train_double_dqn_gym.py --env CartPole-v0

To solve Pendulum-v0, run:
    python train_double_dqn_gym.py --env Pendulum-v0
"""

import argparse
import os
import sys

import gym
import gym_sumo
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
import copy

import pfrl
from pfrl import experiments, explorers
from pfrl import nn as pnn
from pfrl import q_functions, replay_buffers, utils
from pfrl.agents.double_dqn import DoubleDQN

from experiments_sumo import eval_sumo

FUNCTION_MAP = {"relu": F.relu}


def main():
    import logging

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--env", type=str, default="sumo-light-v0")
    parser.add_argument("--eval-env", type=str, default="sumo-fix-v0")
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--render-train", action="store_true")
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")

    # monitor hiper param
    parser.add_argument(
        "--video-freq", type=int, default=1, help="record video freaquency"
    )
    parser.add_argument("--monitor", action="store_true", default=False)

    # scale reward env hyper param
    parser.add_argument("--reward-scale-factor", type=float, default=1e-3)

    # q func hyper param
    parser.add_argument("--n-hidden-channels", type=int, default=100)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    # action space is box
    parser.add_argument("--scale-mu", action="store_false", default=True)
    # action space is discrete
    parser.add_argument("--last-wscale", type=float, default=1.0)
    parser.add_argument(
        "--nonlinearity", type=str, choices=list(FUNCTION_MAP.keys()), default="relu"
    )

    # optimizer hiper param
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--amsgrad", action="store_true", default=False)

    # replay buffers hyper param
    parser.add_argument("--rbuf-capacity", type=float, default=5 * 10 ** 5)
    # replay buffers hyper param if prioritized-replay is True
    parser.add_argument("--prioritized-replay", action="store_true")
    parser.add_argument("--rbuf-alpha", type=float, default=0.6)
    parser.add_argument("--rbuf-beta0", type=float, default=0.4)
    parser.add_argument("--rbuf-eps", type=float, default=0.01)
    parser.add_argument("--rbuf-normalize-by-max", action="store_false", default=True)
    parser.add_argument("--rbuf-error-min", type=int, default=0)
    parser.add_argument("--rbuf-error-max", type=int, default=1)
    parser.add_argument("--rbuf-num-steps", type=int, default=1)

    # explorer(select action func) hyper param
    parser.add_argument("--start-epsilon", type=float, default=1.0)
    parser.add_argument("--end-epsilon", type=float, default=0.1)
    parser.add_argument("--final-exploration-steps", type=int, default=10 ** 4)
    parser.add_argument("--noisy-net-sigma", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=0.2)

    # DoubleDqn hyper param (except optimizer, replar buffer, explorer, q func)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gpu", type=int, default=device)
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10 ** 2)
    parser.add_argument("--clip-delta", action="store_false", default=True)
    parser.add_argument("--target-update-method", type=str, default="hard")
    parser.add_argument("--soft-update-tau", type=float, default=1e-2)
    parser.add_argument("--n-times-update", type=int, default=1)
    parser.add_argument("--batch-accumulator", type=str, default="mean")
    parser.add_argument("--episodic-update-len", type=int, default=None)
    parser.add_argument("--recurrent", action="store_true", default=False)
    parser.add_argument("--max-grad-norm", type=float, default=None)

    # train or evaluate hyper param
    parser.add_argument("--steps", type=int, default=10 ** 6)
    parser.add_argument("--eval-n-steps", type=int, default=None)
    parser.add_argument("--eval-n-runs", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=kwargs_learn["road_freq"])
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--eval-max-episode-len", type=int, default=None)
    parser.add_argument("--successful-score", type=float, default=None)
    parser.add_argument("--save-best-so-far-agent", action="store_false", default=True)
    parser.add_argument("--use-tensorboard", action="store_false", default=True)
    # actor learner
    parser.add_argument(
        "--actor-learner",
        action="store_true",
        help="Enable asynchronous sampling with asynchronous actor(s)",
    )  # NOQA
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help=(
            "The number of environments for sampling (only effective with"
            " --actor-learner enabled)"
        ),
    )  # NOQA
    parser.add_argument("--eval-success-threshold", type=float, default=0.0)

    args = parser.parse_args()

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    eval_sumo.save_sumo_version(args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    log_file_name = os.path.join(args.outdir, "log.log")
    logging.basicConfig(level=logging.INFO, filename=log_file_name)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    def make_env(idx=0, test=False):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        utils.set_random_seed(env_seed)
        kwargs_tmp = kwargs_eval if test else kwargs_learn
        env_name = args.eval_env if test else args.env
        kwargs = copy.deepcopy(kwargs_tmp)
        kwargs["seed"] = env_seed
        kwargs["label"] = kwargs_tmp["label"] + str(process_seed)
        if args.monitor:
            kwargs["mode"] = "gui"
        env = gym.make(env_name, **kwargs)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, video_callable=(lambda e: e % args.video_freq == 0)
            )
        if isinstance(env.action_space, spaces.Box):
            utils.env_modifiers.make_action_filtered(env, clip_action_filter)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if (args.render_eval and test) or (args.render_train and not test):
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space

    if isinstance(action_space, spaces.Box):
        action_size = action_space.low.size
        # Use NAF to apply DQN to continuous action spaces
        q_func = q_functions.FCQuadraticStateQFunction(
            obs_size,
            action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            action_space=action_space,
            scale_mu=args.scale_mu,
        )
        # Use the Ornstein-Uhlenbeck process for exploration
        ou_sigma = (action_space.high - action_space.low) * args.sigma
        explorer = explorers.AdditiveOU(sigma=ou_sigma)
    else:
        n_actions = action_space.n
        q_func = q_functions.FCStateQFunctionWithDiscreteAction(
            obs_size,
            n_actions,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            nonlinearity=FUNCTION_MAP[args.nonlinearity],
            last_wscale=args.last_wscale,
        )
        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon,
            args.end_epsilon,
            args.final_exploration_steps,
            action_space.sample,
        )

    if args.noisy_net_sigma is not None:
        pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()

    opt = optim.Adam(
        q_func.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )

    rbuf_capacity = args.rbuf_capacity
    if args.minibatch_size is None:
        args.minibatch_size = 32
    if args.prioritized_replay:
        betasteps = (args.steps - args.replay_start_size) // args.update_interval
        rbuf = replay_buffers.PrioritizedReplayBuffer(
            rbuf_capacity,
            alpha=args.rbuf_alpha,
            beta0=args.rbuf_beta0,
            betasteps=betasteps,
            eps=args.rbuf_eps,
            normalize_by_max=args.rbuf_normalize_by_max,
            error_min=args.rbuf_error_min,
            error_max=args.rbuf_error_max,
            num_steps=args.rbuf_num_steps,
        )
    else:
        rbuf = replay_buffers.ReplayBuffer(rbuf_capacity, num_steps=args.rbuf_num_steps)

    agent = DoubleDQN(
        q_func,
        opt,
        rbuf,
        args.gamma,
        explorer,
        gpu=args.gpu,
        replay_start_size=args.replay_start_size,
        minibatch_size=args.minibatch_size,
        update_interval=args.update_interval,
        target_update_interval=args.target_update_interval,
        clip_delta=args.clip_delta,
        target_update_method=args.target_update_method,
        soft_update_tau=args.soft_update_tau,
        n_times_update=args.n_times_update,
        batch_accumulator=args.batch_accumulator,
        episodic_update_len=args.episodic_update_len,
        recurrent=args.recurrent,
        max_grad_norm=args.max_grad_norm,
    )

    if args.load:
        agent.load(args.load)

    eval_env = make_env(test=True)

    if args.demo:
        # eval_stats = experiments.eval_performance(
        #     env=eval_env,
        #     agent=agent,
        #     n_steps=args.eval_n_steps,
        #     n_episodes=args.eval_n_runs,
        #     max_episode_len=timestep_limit,
        # )
        eval_stats = eval_sumo.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=args.eval_n_steps,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        env.close()
        eval_env.close()
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )

    elif not args.actor_learner:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            checkpoint_freq=args.checkpoint_freq,
            train_max_episode_len=timestep_limit,
            step_offset=args.step_offset,
            eval_max_episode_len=args.eval_max_episode_len,
            eval_env=eval_env,
            successful_score=args.successful_score,
            step_hooks=(),
            save_best_so_far_agent=args.save_best_so_far_agent,
            use_tensorboard=args.use_tensorboard,
        )
        env.close()
        eval_env.close()
    else:
        # using impala mode when given num of envs

        # When we use multiple envs, it is critical to ensure each env
        # can occupy a CPU core to get the best performance.
        # Therefore, we need to prevent potential CPU over-provision caused by
        # multi-threading in Openmp and Numpy.
        # Disable the multi-threading on Openmp and Numpy.
        os.environ["OMP_NUM_THREADS"] = "1"  # NOQA

        (
            make_actor,
            learner,
            poller,
            exception_event,
        ) = agent.setup_actor_learner_training(args.num_envs)

        poller.start()
        learner.start()

        experiments.train_agent_async(
            args.outdir,
            args.num_envs,
            make_env,
            profile=args.profile,
            steps=args.steps,
            eval_interval=args.eval_interval,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=args.eval_n_runs,
            eval_success_threshold=args.eval_success_threshold,
            max_episode_len=args.eval_max_episode_len,
            step_offset=args.step_offset,
            successful_score=args.successful_score,
            make_agent=make_actor,
            save_best_so_far_agent=args.save_best_so_far_agent,
            use_tensorboard=args.use_tensorboard,
            stop_event=learner.stop_event,
            exception_event=exception_event,
        )

        poller.stop()
        learner.stop()
        poller.join()
        learner.join()


if __name__ == "__main__":
    kwargs_learn = {
        "mode": "cui",
        "carnum": 1,
        "label": "learn",
        "step_length": 1,
        "road_freq": 1000,
    }
    kwargs_eval = {"mode": "cui", "carnum": 1, "label": "eval", "step_length": 1}
    device = 0 if torch.cuda.is_available() else -1
    main()
