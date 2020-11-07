import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.optim import Adam

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

env = gym.make('CartPole-v0')
env.observation_space.shape

OBSERVATION_SIZE = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n

network = nn.Sequential(
    nn.Linear(in_features=OBSERVATION_SIZE, out_features=HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(in_features=HIDDEN_SIZE, out_features=NUM_ACTIONS),
)


class Episode():
    """
    Episodeの情報を保持するためのクラス。

    Attributes
    ----------
    reward : int
        獲得した報酬の値。
    episode_step_list : list of EpisodeStep
        Episode内の各アクション単位のオブジェクトを格納したリスト。
    """

    def __init__(self, reward, episode_step_list):
        """
        Parameteres
        -----------
        reward : int
            獲得した報酬の値。
        episode_step_list : list of EpisodeStep
            Episode内の各アクション単位のオブジェクトを格納したリスト。
        """
        self.reward = reward
        self.episode_step_list = episode_step_list


class EpisodeStep():
    """
    Episode中のAction単体分の情報の保持するためのクラス。

    Attributes
    ----------
    observation : ndarray
        (4,)のサイズの、観測値の配列。
    action : int
        選択されたActionの番号。
    """

    def __init__(self, observation, action):
        """
        Parameters
        ----------
        observation : ndarray
            (4,)のサイズの、観測値の配列。
        action : int
            選択されたActionの番号。
        """
        self.observation = observation
        self.action = action


def iter_batch():
    """
    バッチ1回分の処理を行う。

    Returns
    -------
    episode_list : list of Episode
        1回のバッチで実行されたEpisodeを格納したリスト。
        バッチサイズの件数分、Episodeが追加される。
    """
    # 一度のバッチでの各Episodeの情報を格納するリスト。
    episode_list = []

    episode_reward = 0.0
    episode_step_list = []

    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([obs])
        act_probabilities_v = sm(network(input=obs_v))
        tmp = act_probabilities_v.data.numpy()
        act_probabilities = act_probabilities_v.data.numpy()[0]
        action = np.random.choice(a=len(act_probabilities), p=act_probabilities)

        next_obs, reward, is_done, _ = env.step(action=action)
        episode_reward += reward

        # 新しいObservationではなく、今回のRewardを獲得した時点のObservation
        # をリストに追加します。
        episode_step = EpisodeStep(observation=obs, action=action)
        episode_step_list.append(episode_step)

        # is_doneがTrueになった、ということはEpisode単体の終了を意味します。
        if is_done:
            episode = Episode(
                reward=episode_reward, episode_step_list=episode_step_list)
            episode_list.append(episode)

            # 次のEpisodeのために、各値をリセットします。
            episode_reward = 0.0
            episode_step_list = []
            next_obs = env.reset()

            if len(episode_list) == BATCH_SIZE:
                return episode_list

        obs = next_obs


def get_episode_filtered_results(episode_list):
    """
    バッチ単位の処理で生成されたEpisodeのリスト内容を、指定されている
    パーセンタイルのしきい値を参照してフィルタリングし、結果の（優秀な）
    エピソードのObservationやActionのテンソル、Rewardの平均値などを
    取得する。

    Parameters
    ----------
    episode_list : list of Episode
        対象のバッチ単位でのEpisodeを格納したリスト。

    Returns
    -------
    train_obs_v : FloatTensor
        しきい値によるフィルタリング後の残ったEpisodeの、Observationの
        (4,)のサイズのデータを各エピソードのAction数分だけ格納した
        テンソル(M, 4)のサイズで設定される。（Mは残ったEpisodeの
        Action数に依存する）。学習用に参照される。
    train_act_v : LongTensor
        しきい値によるフィルタリング後の残ったEpisodeの、各Actionの
        値を格納したテンソル(M,)のサイズで設定される。（Mは残った
        EpisodeのAction数に依存し、train_obs_vのサイズと一致した
        値が設定される）。学習用に参照される。
    reward_bound : int
        フィルタリング処理で参照された、報酬のしきい値の値。
    reward_mean : float
        指定されたバッチでのEpisode全体の、Rewardの平均値。
    """
    reward_list = []
    for episode in episode_list:
        reward_list.append(episode.reward)
    reward_bound = np.percentile(a=reward_list, q=PERCENTILE)
    reward_mean = float(np.mean(reward_list))

    train_obs_list = []
    train_act_list = []
    for episode in episode_list:
        # 各Episodeに対して、パーセンタイルで算出したしきい値未満のものを
        # 対象外とする。
        if episode.reward < reward_bound:
            continue

        for episode_step in episode.episode_step_list:
            train_obs_list.append(episode_step.observation)
            train_act_list.append(episode_step.action)

    train_obs_v = torch.FloatTensor(train_obs_list)
    train_act_v = torch.LongTensor(train_act_list)

    return train_obs_v, train_act_v, reward_bound, reward_mean


loss_func = nn.CrossEntropyLoss()
optimizer = Adam(params=network.parameters(), lr=0.01)
TENSOR_BOARD_LOG_DIR = './tensorboard_log'
writer = SummaryWriter(log_dir=TENSOR_BOARD_LOG_DIR)


iter_no = 0
while True:

    episode_list = iter_batch()
    train_obs_v, train_act_v, reward_bound, reward_mean = \
        get_episode_filtered_results(episode_list=episode_list)
    optimizer.zero_grad()
    network_output_tensor = network(train_obs_v)
    loss_v = loss_func(network_output_tensor, train_act_v)
    loss_v.backward()
    optimizer.step()

    loss = loss_v.item()
    log_str = 'iter_no : %d' % iter_no
    log_str += ', loss : %.3f' % loss
    log_str += ', reward_bound : %.1f' % reward_bound
    log_str += ', reward_mean : %.1f' % reward_mean
    print(log_str)

    writer.add_scalar(
        tag='loss', scalar_value=loss, global_step=iter_no)
    writer.add_scalar(
        tag='reward_bound', scalar_value=reward_bound,
        global_step=iter_no)
    writer.add_scalar(
        tag='reward_mean', scalar_value=reward_mean,
        global_step=iter_no)

    if reward_mean > 199:
        print('Rewardの平均値が目標値を超えたため、学習を停止します。')
        break

    iter_no += 1

writer.close()