import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from qnetwork import QNetwork
from replay_buffer import ReplayBuffer
from base_agent import BaseAgent


torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(torch_device)


class DQNAgent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        lr=5e-4,
        buffer_size=int(1e5),
        train_batch_size=64,
        discount_factor=0.99,
        TAU=1e-3, # update of best parameters
        update_iteration=4,
    ):
        """Initialize the DQN agent

        :param state_size: number of dimensions within state space
        :param action_size: number of dimensions within action space
        :param seed: random seed
        :param lr: learning rate
        :param buffer_size: total size of the replay buffer
        :param train_batch_size: size of the batch taken from buffer
        :param discount_factor: reward discount factor
        :param TAU: best model parameter updates
        :param update_iteration: update the model at every Nth iteration
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        self.lr = lr
        self.buffer_size = buffer_size
        self.train_batch_size = train_batch_size
        self.discount_factor = discount_factor
        self.TAU = TAU
        self.update_iteration = update_iteration

        # Q-Network
        self.qnet_model_current = QNetwork(state_size, action_size, seed).to(device)
        self.qnet_model_best = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnet_model_current.parameters(), lr=lr)

        # Replay memory
        self.replay_buffer = ReplayBuffer(action_size, buffer_size, train_batch_size, seed, device)

        # Initialize update iteration
        self.update_num = 0
    
    def step(self, state, action, reward, next_state, done):
        """ Step the agent which may update the underlying model using SARSA data

        :param state: Iterable[float] of state_size dimensions containing state space at time T
        :param action: int Chosen action index at time T
        :param reward: Reward received from taking action A with state S at time T
        :param next_state: Iterable[float] of state_size dimensions containing state space at time T+1
        :param done: boolean indicating episode done condition: True = done, False = not done
        """
        # store experiences in replay buffer
        self.replay_buffer.add_experience(
            state,
            action,
            reward,
            next_state,
            done
        )

        # learn at update_iteration
        self.update_num += 1
        if self.update_num == self.update_iteration:
            self.update_num = 0

            # update model with random replays from buffer
            if self.replay_buffer.has_enough_data():
                self.model_update(self.replay_buffer.sample())

    def act(self, state, epsilon=0):
        """Returns the action selected

        :param state: Iterable[float] of state_size dimensions containing state space at time T
        :param epsilon: (float) epsilon value for randomly selecting action
        """

        # convert the numpy state into a torch expected format
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # set model to evaluation mode
        self.qnet_model_current.eval()

        # ensure all gradients are detached from graphs
        with torch.no_grad():

            # calculate the actions from the model and current state
            actions = self.qnet_model_current(state)

        # set model to train mode
        self.qnet_model_current.train()

        # Epsilon-greedy action selection
        action = None
        if random.random() > epsilon:
            action = np.argmax(actions.cpu().data.numpy())
        else:
            action = random.choice(range(self.action_size))

        return action

    def model_update(self, experiences):
        """Update value parameters using given batch of experience tuples.

        :param experiences: SARSA named tuples
        """
        # Split into components
        states, actions, rewards, next_states, dones = experiences

        # max predicted Q-vals from best model
        Q_best_next = self.qnet_model_best(next_states).detach().max(1)[0].unsqueeze(1)

        # Q-best-vals for current states
        Q_best = rewards + (self.discount_factor * Q_best_next * (1 - dones))

        # expected Q-vals from current model
        Q_expected = self.qnet_model_current(states).gather(1, actions)

        # get loss using mean squared error
        loss = F.mse_loss(Q_expected, Q_best)

        # reduce loss using optimizer along gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the best network
        for best_param, current_param in zip(
            self.qnet_model_best.parameters(),
            self.qnet_model_current.parameters(),
        ):
            current_portion_param = self.TAU * current_param.data
            best_portion_param = (1.0 - self.TAU) * best_param.data
            best_param.data.copy_(current_portion_param + best_portion_param)
