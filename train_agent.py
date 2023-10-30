from collections import deque
from typing import List

import numpy as np
import torch

from base_agent import BaseAgent
from utils import logger


def train(
    env,
    agent: BaseAgent,
    num_episodes: int = 2000,
    max_timesteps: int = 1000,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
    threshold: float = 13.0,
) -> List[float]:
    """ code to train an agent
    :param env: environment that agent will train in
    :param agent: (BaseAgent) agent to be trained
    :param num_episodes: (int) maximum number of training episodes
    :param max_timesteps: (int) maximum number of time steps per episode
    :param eps_start: (float) starting value of epsilon assuming epsilon-greedy approach
    :param eps_end: (float) absolute minimum value of epsilon
    :param eps_decay: (float) multiplicative factor (per episode) for decreasing epsilon
    :param threshold: (float) required threshold score required for training to be complete
    :return:
    """
    scores: List[float] = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    # loop through each episode
    for i_episode in range(1, num_episodes + 1):
        logger.debug('Episode %d', i_episode)
        state_brain = env.reset()
        state = state_brain['BananaBrain'].__dict__['vector_observations']
        score = 0

        # loop through all time steps within episode
        for t in range(max_timesteps):
            action = agent.act(state, eps)
            logger.debug('Episode %d, Time %d, Chosen Action %d', i_episode, t, action)
            state_brain = env.step(vector_action=int(action))
            next_state = state_brain['BananaBrain'].__dict__['vector_observations'][0]
            reward = state_brain['BananaBrain'].__dict__['rewards'][0]
            done = state_brain['BananaBrain'].__dict__['local_done'][0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            # exit the episode if done condition reached
            if done:
                break

        # save the scores
        scores_window.append(score)
        scores.append(score)

        # reduce the epsilon
        eps = max(eps_end, eps_decay * eps)

        score_str = '\rEpisode {}\tAverage Score: {:.2f}'

        logger.info(score_str.format(i_episode, np.mean(scores_window)))
        if (i_episode % 100) == 0:
            logger.info(score_str.format(i_episode, np.mean(scores_window)))

        # If the avg score of latest window is above threshold, then stop training and save model
        if np.mean(scores_window) >= threshold:
            solved_str = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
            logger.info(solved_str.format(i_episode - 100, np.mean(scores_window)))
            torch.save(agent.actor_model_current.state_dict(), 'models\\checkpoint.pth')
            break

    return scores
