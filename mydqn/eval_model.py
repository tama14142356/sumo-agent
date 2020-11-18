import torch
from model import DQN

import gym
import gym_sumo
import os


kwargs_policy = {
    "mode": "gui",
    "carnum": 1,
    "label": "policy",
}
kwargs_target = {
    "mode": "gui",
    "carnum": 1,
    "label": "target",
}

result_path = os.path.join(
    os.path.dirname(__file__), "../results/sumo-light/mydqn_result/episode_10000"
)
demo_path = os.path.join(result_path, "demo_step_length_1")
demo_policy_path = os.path.join(demo_path, "policy")
demo_target_path = os.path.join(demo_path, "target")
env_policy = gym.make("sumo-light-v0", **kwargs_policy)
env_policy = gym.wrappers.Monitor(env_policy, demo_policy_path, force=True)
env_target = gym.make("sumo-light-v0", **kwargs_target)
env_target = gym.wrappers.Monitor(env_target, demo_target_path, force=True)
obs_space = env_policy.observation_space
act_space = env_policy.action_space
HIDDEN = 96

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# models
policy_net = DQN(obs_space.shape[0], HIDDEN, act_space.n).to(device)
target_net = DQN(obs_space.shape[0], HIDDEN, act_space.n).to(device)

policy_path = os.path.join(result_path, "policy_model_cpu.pt")
policy_net.load_state_dict(torch.load(policy_path))
target_path = os.path.join(result_path, "target_model_cpu.pt")
target_net.load_state_dict(torch.load(target_path))
policy_net.eval()
target_net.eval()
# print(policy_net.state_dict(), target_net.state_dict())

obs = env_policy.reset()

done = False
reset = False
max_step = 200
step = 0
while True:
    obs = torch.as_tensor(obs, dtype=torch.float)
    with torch.no_grad():
        Q = policy_net(obs.to(device)[None, ...])
    action = Q.argmax().item()
    obs, reward, done, _ = env_policy.step(action)
    print(step, reward, done, action)
    step += 1
    reset = step == max_step
    if done or reset:
        break

print("policy_step: ", step)
env_policy.close()

obs = env_target.reset()
done = False
reset = False
step = 0
while True:
    obs = torch.as_tensor(obs, dtype=torch.float)
    with torch.no_grad():
        Q = target_net(obs.to(device)[None, ...])
    action = Q.argmax().item()
    obs, reward, done, _ = env_target.step(action)
    print(step, reward, done, action)
    step += 1
    reset = step == max_step
    if done or reset:
        break

print("target_step: ", step)
env_target.close()
