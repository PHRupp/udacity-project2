import traceback as tb
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from ddpg_agent import DDPGAgent
from train_agent import train
from utils import logger

projects_dir = "C:\\Users\\pathr\\PycharmProjects\\"
app_path = "Reacher_Windows_x86_64\\Reacer.exe"
env = UnityEnvironment(file_name=join(projects_dir, app_path))

try:
    scores = train(
        env=env,
        agent=DDPGAgent(
            state_size=33,
            action_size=4,
            seed=123456789,
            lr_actor=1e-4,
            lr_critic=1e-3,
            buffer_size=int(1e5),
            train_batch_size=128,
            discount_factor=0.99,
            TAU=1e-3,  # update of best parameters
            update_iteration=4,
        ),
        num_episodes=10,
        max_timesteps=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        threshold=30.0,
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
