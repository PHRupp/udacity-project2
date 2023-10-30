import traceback as tb
from os.path import join

import numpy as np
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt

from dqn_agent import DQNAgent
from train_agent import train
from utils import logger

nav_dir = "C:\\Users\\pathr\\PycharmProjects\\Value-based-methods\\p1_navigation\\"
app_path = "Banana_Windows_x86_64\\Banana.exe"
env = UnityEnvironment(file_name=join(nav_dir, app_path))

try:
    scores = train(
        env=env,
        agent=DQNAgent(
            state_size=37,
            action_size=4,
            seed=123456789,
            lr=5e-4,  #5e-4
            buffer_size=int(1e5),
            train_batch_size=64,
            discount_factor=0.99,  #99
            TAU=1e-3,  # update of best parameters
            update_iteration=4,
        ),
        num_episodes=2000,
        max_timesteps=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        threshold=14.0,
    )

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

except Exception:
    logger.critical(tb.format_exc())

logger.info('Exiting...')
env.close()
exit(0)
