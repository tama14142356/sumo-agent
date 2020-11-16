import gym
import torch

from model import DQN
from save_write_result import SaveWriteResult


def make_env():
    return gym.make("CartPole-v0")


def obs_to_tensor(obs):
    return torch.tensor(obs, dtype=torch.float)[None, ...]


def select_action(state, policy_net):
    global steps_done

    steps_done += 1

    with torch.no_grad():
        return policy_net(state.to(device)).max(1)[1].view(1, 1)


def main():
    env = make_env()
    env = gym.wrappers.Monitor(env, "./video", force=True)

    # net
    policy_net = DQN(obs_space.shape[0], HIDDEN, n_actions).to(device)
    target_net = DQN(obs_space.shape[0], HIDDEN, n_actions).to(device)
    policy_net_file_name = "policy_model.pt"
    target_net_file_name = "target_model.pt"
    print(policy_net.state_dict(), " init")
    print(target_net.state_dict(), " init")
    if str(device) == "cpu":
        policy_net_file_name = "policy_model_cpu.pt"
        target_net_file_name = "target_model_cpu.pt"
    save_write_result.load_model(policy_net, policy_net_file_name, 1)
    save_write_result.load_model(target_net, target_net_file_name, 1)
    print(policy_net.state_dict(), " load")
    print(target_net.state_dict(), " load")

    # demo
    for i_episode in range(EPISODES):
        episode_return = 0
        state = obs_to_tensor(env.reset())
        done = False
        local_step = 0
        reward_min, reward_max = 0, 0

        # episode
        while not done:
            # calc action, act
            action = select_action(state, policy_net).cpu()
            next_state, reward, done, _ = env.step(action.item())
            episode_return += reward
            reward_max = reward if local_step == 0 else max(reward_max, reward)
            reward_min = reward if local_step == 0 else min(reward_min, reward)

            # push to memory
            if done:
                next_state = None
            else:
                next_state = obs_to_tensor(next_state)
            reward = torch.tensor([reward])

            state = next_state

            save_write_result.writer.add_scalar(
                tag="eval/reward", scalar_value=reward, global_step=steps_done
            )
            local_step += 1

        save_write_result.writer.add_scalar(
            tag="eval/total_reward",
            scalar_value=episode_return,
            global_step=steps_done,
        )
        save_write_result.writer.add_scalar(
            tag="eval/reward_min", scalar_value=reward_min, global_step=steps_done
        )
        save_write_result.writer.add_scalar(
            tag="eval/reward_max", scalar_value=reward_max, global_step=steps_done
        )

        # update target
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # log
        print(f"Episode {i_episode}: steps={local_step}: return={episode_return}")

    env.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env()
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
    EPISODES = 100
    HIDDEN = 128
    LR = 1e-3
    save_write_result = SaveWriteResult(EPISODES, True)

    main()
    save_write_result.close()
