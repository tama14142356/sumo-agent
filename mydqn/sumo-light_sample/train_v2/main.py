import math
import random

import gym
import gym_sumo
import torch
import torch.optim as optim
import torch.nn.functional as F

from memory import ReplayMemory, Transition
from model import DQN
from save_write_result import SaveWriteResult


def make_env(kwargs):
    return gym.make("sumo-light-v0", **kwargs)


def obs_to_tensor(obs):
    return torch.tensor(obs, dtype=torch.float)[None, ...]


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return -1.0

    # get batch from memory
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # reshape batch
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    ).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # now Q
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # target Q
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # calc loss, update
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def select_action(state, policy_net):
    global steps_done

    # calc epsilon
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )

    steps_done += 1

    # epsilon-greedy
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


def main():
    env = make_env(kwargs_eval)

    # net
    policy_net = DQN(obs_space.shape[0], HIDDEN, n_actions).to(device)
    target_net = DQN(obs_space.shape[0], HIDDEN, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    # learn
    for i_episode in range(EPISODES):
        episode_return = 0
        reward_list, loss_list = [], []
        state = obs_to_tensor(env.reset())
        done = False

        # episode
        while not done:
            # calc action, act
            action = select_action(state, policy_net).cpu()
            next_state, reward, done, _ = env.step(action.item())
            episode_return += reward
            reward_list.append(reward)

            # push to memory
            if done:
                next_state = None
            else:
                next_state = obs_to_tensor(next_state)
            reward = torch.tensor([reward])
            memory.push(state, action, next_state, reward)

            state = next_state

            loss_value = optimize_model(memory, policy_net, target_net, optimizer)
            if loss_value > 0:
                loss_list.append(loss_value)

        save_write_result.writing_list(
            tag="agent/loss", target_list=loss_list, end_step=steps_done
        )
        save_write_result.writing_list(
            tag="agent/reward", target_list=reward_list, end_step=steps_done
        )

        save_write_result.writer.add_scalar(
            tag="agent/total_reward",
            scalar_value=episode_return,
            global_step=steps_done,
        )
        save_write_result.writer.add_scalar(
            tag="agent/reward_min",
            scalar_value=min(reward_list),
            global_step=steps_done,
        )
        save_write_result.writer.add_scalar(
            tag="agent/reward_max",
            scalar_value=max(reward_list),
            global_step=steps_done,
        )

        # update target
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # log
        print(f"Episode {i_episode}: return={episode_return}")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs_default = {"mode": "cui", "carnum": 1, "label": "learn"}
    kwargs_eval = {"mode": "cui", "carnum": 1, "label": "eval"}
    env = make_env(kwargs_default)

    obs_space = env.observation_space
    act_space = env.action_space
    n_actions = act_space.n
    env.close()
    del env

    steps_done = 0

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    EPISODES = 1000
    HIDDEN = 128

    save_write_result = SaveWriteResult(EPISODES)
    main()
    save_write_result.close()
