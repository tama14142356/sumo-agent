import logging
import statistics
import numpy as np


def _run_episodes_sumo(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns."""
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    scores = []
    terminate = False
    timestep = 0

    reset = True
    while not terminate:
        if reset:
            obs = env.reset()
            done = False
            test_r = 0
            episode_len = 0
            info = {}
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        cur_speed = env.get_speed()
        cur_sm_time = env.traci_connect.simulation.getTime()
        cur_step = env.get_cur_step()
        test_r += reward
        episode_len += 1
        timestep += 1
        reset = done or episode_len == max_episode_len or info.get("needs_reset", False)
        agent.observe(obs, reward, done, reset)
        logger.info(
            "episode step:%s action:%s reward:%s speed:%s sim_time:%s cur_step:%s",
            episode_len,
            action,
            reward,
            cur_speed,
            cur_sm_time,
            cur_step,
        )
        if reset:
            logger.info(
                "evaluation episode %s length:%s R:%s",
                len(scores),
                episode_len,
                test_r,
            )
            # As mixing float and numpy float causes errors in statistics
            # functions, here every score is cast to float.
            scores.append(float(test_r))
        if n_steps is None:
            terminate = len(scores) >= n_episodes
        else:
            terminate = timestep >= n_steps
    # If all steps were used for a single unfinished episode
    if len(scores) == 0:
        scores.append(float(test_r))
        logger.info(
            "evaluation episode %s length:%s R:%s", len(scores), episode_len, test_r
        )
    return scores


def run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    with agent.eval_mode():
        return _run_episodes_sumo(
            env=env,
            agent=agent,
            n_steps=n_steps,
            n_episodes=n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )


def eval_performance(
    env, agent, n_steps, n_episodes, max_episode_len=None, logger=None
):
    """Run multiple evaluation episodes and return statistics.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation episodes.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        Dict of statistics.
    """

    assert (n_steps is None) != (n_episodes is None)

    scores = run_evaluation_episodes(
        env,
        agent,
        n_steps,
        n_episodes,
        max_episode_len=max_episode_len,
        logger=logger,
    )
    stats = dict(
        episodes=len(scores),
        mean=statistics.mean(scores),
        median=statistics.median(scores),
        stdev=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        max=np.max(scores),
        min=np.min(scores),
    )
    return stats
