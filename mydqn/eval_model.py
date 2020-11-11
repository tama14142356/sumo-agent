import torch
from sumo_light_v0.model import DQN

import gym
import gym_sumo
import os


kwargs_policy = {
    # "mode": "cui",
    "carnum": 1
}
kwargs_target = {
    "mode": "cui",
    "carnum": 1,
    "label": "default2",
}

env_policy = gym.make("sumo-light-v0", **kwargs_policy)
env_target = gym.make("sumo-light-v0", **kwargs_target)
obs_space = env_policy.observation_space
act_space = env_policy.action_space
HIDDEN = 96

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# models
policy_net = DQN(obs_space.shape[0], HIDDEN, act_space.n).to(device)
target_net = DQN(obs_space.shape[0], HIDDEN, act_space.n).to(device)

result_path = os.path.join(
    os.path.dirname(__file__), "../results/sumo-light1/mydqn/episode_10000"
)
policy_path = os.path.join(result_path, "policy_model_cpu.pt")
policy_net.load_state_dict(torch.load(policy_path))
target_path = os.path.join(result_path, "target_model_cpu.pt")
target_net.load_state_dict(torch.load(target_path))
policy_net.eval()
target_net.eval()
print(policy_net.state_dict(), target_net.state_dict())

obs = env_policy.reset()
obs = env_target.reset()

done = False
step = 0
while not done:
    obs = torch.as_tensor(obs, dtype=torch.float)
    with torch.no_grad():
        Q = policy_net(obs.to(device)[None, ...])
    action = Q.argmax().item()
    obs, reward, done, _ = env_policy.step(action)
    print(step, reward, done, action)
    step += 1

print("policy_step: ", step)
env_policy.close()

done = False
step = 0
while not done:
    obs = torch.as_tensor(obs, dtype=torch.float)
    with torch.no_grad():
        Q = target_net(obs.to(device)[None, ...])
    action = Q.argmax().item()
    obs, reward, done, _ = env_target.step(action)
    print(step, reward, done, action)
    step += 1

print("target_step: ", step)
env_target.close()
