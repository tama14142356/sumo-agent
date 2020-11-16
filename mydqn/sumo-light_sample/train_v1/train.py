import math
import random

import gym
import gym_sumo
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import DQN
from replay_memory import ReplayMemory, Transition
from save_write_result import SaveWriteResult


def act_random(step):
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * step / EPS_DECAY)
    return random.random() < eps


def main(kwargs):
    # env spaces
    env = gym.make("sumo-light-v0", **kwargs)
    obs_space = env.observation_space
    act_space = env.action_space

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # models
    policy_net = DQN(obs_space.shape[0], HIDDEN, act_space.n).to(device)
    target_net = DQN(obs_space.shape[0], HIDDEN, act_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    replay_memory = ReplayMemory(CAPACITY)

    # step (over episodes)
    step = 0

    # learn
    for episode in range(EPISODE):
        print(f"Episode {episode}")

        # prepare
        obs = torch.as_tensor(env.reset(), dtype=torch.float)
        done = False

        # for writing tensorboard
        total_reward = 0.0
        reward_list = []
        loss_list = []

        # episode
        while not done:
            # determine action
            if act_random(step):
                action = act_space.sample()
            else:
                with torch.no_grad():
                    Q = policy_net(obs.to(device)[None, ...])
                action = Q.argmax().item()

            # act in env
            next_obs, reward, done, _ = env.step(action)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float)
            step += 1

            replay_memory.push(obs, action, next_obs, reward, done)

            obs = next_obs
            reward_list.append(reward)
            total_reward += reward

            if len(replay_memory) < BATCH_SIZE:
                continue

            # get batch from replay memory
            transitions = replay_memory.sample(BATCH_SIZE)  # List[Transition]
            # Transition(List[obs], List[action], ...)
            batch = Transition(*zip(*transitions))
            obs_batch = torch.stack(batch.obs).to(device)
            action_batch = torch.tensor(batch.action).to(device)[:, None]
            next_obs_batch = torch.stack(batch.next_obs).to(device)
            reward_batch = torch.tensor(batch.reward).to(device)[:, None]
            done_batch = torch.tensor(batch.done).to(device)[:, None]

            # calc loss
            Q = policy_net(obs_batch).gather(1, action_batch)
            with torch.no_grad():
                target_Q = (
                    target_net(next_obs_batch).max(1, keepdim=True)[0] * ~done_batch
                )
            loss = F.smooth_l1_loss(Q, reward_batch + GAMMA * target_Q)

            # update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_list.append(loss_value)

        # update target
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # writing tensorboard
        save_write_result.writing_list(
            tag="agent/loss", target_list=loss_list, end_step=step
        )
        save_write_result.writing_list(
            tag="agent/reward", target_list=reward_list, end_step=step
        )

        save_write_result.writer.add_scalar(
            tag="agent/reward_min", scalar_value=min(reward_list), global_step=step
        )
        save_write_result.writer.add_scalar(
            tag="agent/reward_max", scalar_value=max(reward_list), global_step=step
        )
        save_write_result.writer.add_scalar(
            tag="agent/total_reward", scalar_value=total_reward, global_step=step
        )
    # save model parameters
    target_filename, policy_filename = "target_model.pt", "policy_model.pt"
    save_write_result.save_model(target_net, target_filename, device)
    save_write_result.save_model(policy_net, policy_filename, device)
    if str(device) != "cpu":
        cpu_device = torch.device("cpu")
        save_write_result.save_model(policy_net, policy_filename, cpu_device)
        save_write_result.save_model(target_net, target_filename, cpu_device)

    env.close()


if __name__ == "__main__":
    # hyperparameter
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    HIDDEN = 96
    LR = 1e-4
    CAPACITY = 10000
    EPISODE = 10
    save_write_result = SaveWriteResult(EPISODE)

    kwargs = {"mode": "cui", "carnum": 1}

    main(kwargs)
    save_write_result.close()
