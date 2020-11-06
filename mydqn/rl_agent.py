import gym
import gym_sumo
from gym import wrappers

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import copy
import time


MONITOR = False
mode = "cui"
if MONITOR:
    mode = "gui"

kwargs = {
    "mode": mode,
    "carnum": 1
}


class NeuaralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hidden_size = 200
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

        self.batch_size = 32
        self.batch_input_shape = (self.batch_size, input_size)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        # x = self.relu(x)
        return x


class DQNAgent:
    def __init__(self, model, gamma=0.85, epsilon_decay=0.001, epsilon_min=0.01,
                 learning_rate=0.005, memory_size=200, start_reduce_epsilon=200):
        self.model = model
        self.batch_size = model.batch_size
        self.batch_input_shape = model.batch_input_shape
        self.memory = list()
        self.memory_size = memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.start_reduce_epsilon = start_reduce_epsilon
        self.step = 0

        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.model.apply(self.init_parameters)

    def act(self, state):
        if self.step > self.start_reduce_epsilon and self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        action = env.action_space.sample()
        if np.random.random() > self.epsilon:
            obs_size = self.batch_input_shape[1]
            obs = torch.tensor(state.reshape((1, obs_size)))
            prediction = self.model(obs)
            _, indices_prediction = torch.max(prediction, 1)
            action = indices_prediction.numpy()[0]
        self.step += 1
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.memory_size:
            return -1.0

        batch_size = self.batch_size
        # samples = np.random.choice(self.memory, self.batch_size)
        samples = np.random.permutation(self.memory)
        sample_idx = range(1)
        total_loss = 0.0
        # sample_idx = range(len(samples))
        for i in sample_idx[::batch_size]:
            batch = samples[i:i + batch_size]
            cur_states = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(
                self.batch_input_shape)
            actions = np.array(batch[:, 1].tolist(), dtype=np.int32)
            rewards = np.array(batch[:, 2].tolist(), dtype=np.float32)
            next_states = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(
                self.batch_input_shape)
            dones = np.array(batch[:, 4].tolist(), dtype=np.bool)
            q = self.model(torch.tensor(cur_states))
            target = copy.deepcopy(q.numpy())
            predictions = self.model(torch.tensor(next_states))
            Q_max, _ = torch.max(predictions, 1)
            Q_max_array = Q_max.numpy()
            for j in range(batch_size):
                discreate_action = int(actions[j])
                target[j][discreate_action] = rewards[j] + \
                    Q_max_array[j] * self.gamma * (not dones[j])
            loss = self.train_step(cur_states, torch.tensor(target))
            total_loss += loss

        return total_loss

    def train_step(self, train_x, train_y):
        self.model.train()
        prediction = self.model(train_x)
        self.optimizer.zero_grad()
        loss = self.criterion(prediction, train_y)
        loss.backword()

        self.optimizer.step()

        return loss.item()

    def save_model(self, fn):
        self.model.save_model(fn)

    def init_parameters(self, layers):
        if type(layers) == nn.Linear:
            nn.init.xavier_uniform_(layers.weight)
            layers.bias.data.fill_(0.0)


if __name__ == "__main__":
    trials = 2000
    trial_len = 100
    train_freq = 10
    TENSOR_BOARD_LOG_DIR = './result-sumo-light1'
    writer = SummaryWriter(log_dir=TENSOR_BOARD_LOG_DIR)

    env = gym.make("sumo-light-v0", **kwargs)
    if MONITOR:
        env = wrappers.Monitor(env, "./video", force=True)

    obs_size = env.observation_space.low.size
    num_action = env.action_space.n

    Q = NeuaralNet(obs_size, num_action)
    # Q_ast = copy.deepcopy(Q)
    dqn_agent = DQNAgent(Q, env)

    total_losses = []
    total_step = 0
    total_rewards = []
    buffer_size = dqn_agent.memory_size
    cur_state = env.reset()

    for i in range(buffer_size):
        action = dqn_agent.act(cur_state)
        next_state, reward, done, info = env.step(action)
        dqn_agent.remember(cur_state, action, reward, next_state, done)
        cur_state = next_state if not done else env.reset()

    start = time.time()

    for trial in range(trials):
        cur_state = env.reset()
        step_num = 0
        total_loss = 0.0
        total_reward = 0.0
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            next_state, reward, done, info = env.step(action)
            dqn_agent.remember(cur_state, action, reward, next_state, done)
            loss = dqn_agent.replay()
            if loss > 0:
                total_loss += loss
            step_num = step
            total_step += 1
            total_reward += reward

            cur_state = next_state

            if done:
                break

        total_losses.append(total_loss)
        total_rewards.append(total_reward)
        reward_mean = total_reward / step_num

        writer.add_scalar(
            tag='loss', scalar_value=total_loss, global_step=trial)
        writer.add_scalar(
            tag='total_reward', scalar_value=total_reward, global_step=trial)
        writer.add_scalar(
            tag='reward_mean', scalar_value=reward_mean, global_step=trial)

        print(f"Finished trial {trial}")
