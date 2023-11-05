
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
    threshold: float = 30.0,
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
    brain = 'ReacherBrain'

    # loop through each episode
    for i_episode in range(1, num_episodes + 1):
        state_brain = env.reset()
        state = state_brain[brain].__dict__['vector_observations']
        score = 0
        did_finish = False

        # loop through all time steps within episode
        for t in range(max_timesteps):
            action = agent.act(state, add_noise=True)
            state_brain = env.step(vector_action=action)
            next_state = state_brain[brain].__dict__['vector_observations'][0]
            reward = state_brain[brain].__dict__['rewards'][0]
            done = state_brain[brain].__dict__['local_done'][0]
            logger.debug('%d - %d - %s - %s' % (i_episode, t, str(action), str(reward)))
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            # exit the episode if done condition reached
            if done:
                did_finish = True
                exit(0)
                break

        # save the scores
        scores_window.append(score)
        scores.append(score)

        score_str = 'Episode {}\tAverage Score: {:.2f}'
        out_s = score_str.format(i_episode, np.mean(scores_window))

        logger.info(out_s)
        print(out_s)

        # If the avg score of latest window is above threshold, then stop training and save model
        if np.mean(scores_window) >= threshold:
            solved_str = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
            logger.info(solved_str.format(i_episode - 100, np.mean(scores_window)))
            torch.save(agent.actor_model_current.state_dict(), 'models\\checkpoint_actor.pth')
            torch.save(agent.critic_model_current.state_dict(), 'models\\checkpoint_critic.pth')
            break

    return scores
