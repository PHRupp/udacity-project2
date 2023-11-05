
import traceback as tb
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from ddpg_agent import DDPGAgent
from train_agent import train
from utils import logger


projects_dir = "C:\\Users\\pathr\\PycharmProjects\\"
app_path = "Reacher_Windows_x86_64\\Reacher.exe"
app_path = join(projects_dir, app_path)
env = UnityEnvironment(file_name=app_path)

try:
    scores = train(
        env=env,
        agent=DDPGAgent(
            state_size=33,
            action_size=4,
            seed=546879,
            lr_actor=1e-3,
            lr_critic=1e-3,
            buffer_size=int(1e5),
            train_batch_size=256,
            discount_factor=0.95,
            TAU=1e-3,  # update of best parameters
            update_iteration=20,
            weight_decay=0.0001,
            num_updates_per_interval=10,
            noise_decay=0.999,
        ),
        num_episodes=100,
        max_timesteps=200,
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
