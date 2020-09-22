"""An example of training a REINFORCE agent against OpenAI Gym envs.

This script is an example of training a REINFORCE agent against OpenAI Gym
envs. Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_reinforce_gym.py

To solve InvertedPendulum-v1, run:
    python train_reinforce_gym.py --env InvertedPendulum-v1
"""
import argparse

import gym
import gym.spaces
import torch
from torch import nn
from torch_geometric.nn import GCNConv

import pfrl
from pfrl import experiments
from pfrl import utils
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.policies import GaussianHeadWithFixedCovariance

# default
# args_ini = {
#     'step_length': 0.01,
#     'isgraph': True,
#     'area': 'nishiwaseda',
#     'carnum': 100,
#     'mode': 'gui' (or 'cui'),
#     'simlation_step': 100
# }
args_ini = {
    'mode': 'cui',
    'carnum': 10
}

gpudefault = 0 if torch.cuda.is_available() else -1


class Net(nn.Module):
    def __init__(self, env):
        super(Net, self).__init__()
        self.env = env
        dataset = env.unwrapped.data
        num_action = env.action_space[0][0].n + env.action_space[0][1].low.size
        hidden_size = 200
        self.conv1 = GCNConv(dataset.num_node_features, hidden_size)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = GCNConv(hidden_size, num_action)
        self.soft = SoftmaxCategoricalHead()
        self.gaus = GaussianHeadWithFixedCovariance(0.3)

    def forward(self, data):
        device = 'cpu' if gpudefault < 0 else 'cuda'
        graph = self.env.unwrapped.getData(data).to(device)
        # graph = data
        x, edge_index = graph.x, graph.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        length = len(x)
        x1 = x[length - 1]
        x2 = x[length - 2: length - 1]
        f1 = self.soft(x1)
        f2 = self.gaus(x2)
        f = torch.cat((f1, f2), 0)
        return f


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="gym_sumo:sumo-v0",
        help="Gym Env ID."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed [0, 2 ** 32)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=gpudefault,
        help="GPU device ID. Set to -1 to use CPUs only."
    )
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
    parser.add_argument(
        "--batchsize",
        type=int,
        default=10,
        help="Size of minibatch (in timesteps).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10 ** 5,
        help="Total time steps for training."
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10 ** 4,
        help="Interval (in timesteps) between evaluation phases.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=100,
        help="Number of episodes ran in an evaluation phase.",
    )
    parser.add_argument("--reward-scale-factor", type=float, default=1e-2)
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run demo episodes, not training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Directory path to load a saved agent data from"
            " if it is a non-empty string."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=logging.INFO,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(test):
        env = gym.make(args.env, **args_ini)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
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
    model = Net(train_env)

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
        eval_stats = experiments.eval_performance(
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
    main()
