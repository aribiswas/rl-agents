import torch
import torch.optim as optim
import numpy as np
from utils.replay import ReplayMemory
from utils.torch_utils import to_torch_tensor, soft_update


class DQNAgent:

    def __init__(self,
                 model,
                 actions,
                 replay_len=int(1e6),
                 discount_factor=0.99,
                 epsilon=0.9,
                 epsilon_decay=1e-6,
                 epsilon_min=0.1,
                 batch_size=64,
                 learn_rate=0.01,
                 target_update_freq=5,
                 target_smooth_factor=0.01,
                 logger=None):
        """
        Initialize a Deep Q-Network agent.

        Parameters
        ----------
        model :
            Q network model (PyTorch or TensorFlow)
        replay_len : number, optional
            Capacity of replay memory. The default is int(1e6).
        discount_factor : number optional
            Discount factor. The default is 0.99.
        epsilon : number, optional
            Exploration parameter. The default is 0.05.
        epsilon_decay : number, optional
            Decay rate for epsilon. The default is 1e6.
        epsilon_min : number, optional
            Minimum value of epsilon. The default is 0.1.
        batch_size : number, optional
            Batch size for training. The default is 128.
        learn_rate : number, optional
            Learn rate for Q-Network. The default is 0.01.
        target_update_freq : number, optional
            Update frequency for target Q-Network. The default is 0.01.
        target_smooth_factor : number, optional
            Smoothing factor for target Q-Network update. The default is 0.01.

        """

        # initialize agent parameters
        self.actions = actions
        self.num_act = len(self.actions)
        self.step_count = 0
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.target_update_freq = target_update_freq
        self.target_smooth_factor = target_smooth_factor

        # create local and target Q networks
        self.model = model
        self.target_model = model

        # initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)

        # initialize experience buffer
        self.replay = ReplayMemory(state_dim=self.model.input_size,
                                   action_dim=(1,),
                                   max_len=replay_len)

        # logger
        self.logger = logger
        self.learn_count = 0

    def action(self, state):
        """
        Get action from the policy, given the state.
        Parameters
        ----------
        state : numpy array
            State of the environment.
        Returns
        -------
        action : numpy array
            Action.
        """
        # obtain network output y = [Q(s,a1), ... Q(s,an)]
        with torch.no_grad():
            y = self.model(state)

        # compute action
        ep_choice = np.random.rand(1)
        if ep_choice > self.epsilon:
            # epsilon greedy action (exploitation)
            action_choice = torch.argmax(y).numpy()
        else:
            # random action (exploration)
            action_choice = np.random.choice(np.arange(self.model.num_outputs))

        # decay and log epsilon
        ep_next = self.epsilon * (1 - self.epsilon_decay)
        self.epsilon = max(self.epsilon_min, ep_next)
        if self.logger is not None:
            self.logger.add_scalar('Exploration/Epsilon', self.epsilon, self.step_count)

        return self.actions[action_choice]

    def step(self, state, action, reward, next_state, done):
        """
        Step the agent by storing experiences and learning from data.
        Parameters
        ----------
        state : numpy array
            State of the environment.
        action : numpy array
            Actions.
        reward : numpy array
            Rewards.
        next_state : numpy array
            Next states.
        done : numpy array
            Termination flag.
        Returns
        -------
        None.
        """
        # increase step count
        self.step_count += 1

        # convert action to action choice
        action = self.actions.index(action)

        # add experience to buffer
        self.replay.append(state, action, reward, next_state, done)

        # learn from experiences
        if self.replay.len() > self.batch_size:
            self.learn()

    def learn(self):
        """
        Train the agent.
        :return: None
        """
        # create batch experiences for learning
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # convert to torch tensor
        states, actions, rewards, next_states, dones = to_torch_tensor(states, actions, rewards, next_states, dones)

        with torch.no_grad():

            # *** Double DQN ***

            # amax = argmax Q(s+1)
            a_max = torch.argmax(self.model(next_states), dim=1)
            a_max = a_max.reshape((self.batch_size, 1))

            # Q'(s+1|amax)  -> q value for argmax of actions
            target_out = self.target_model(next_states)
            targetQ = torch.stack([target_out[i][a_max[i]] for i in range(self.batch_size)])

            # y = r + gamma * Q'(s+1|amax)
            y = rewards + self.discount_factor * targetQ * (1-dones)

        # Q(s|a) -> q value for action from local policy
        model_out = self.model(states)
        Q = torch.stack([model_out[i][actions[i].numpy()] for i in range(self.batch_size)])

        # calculate mse loss
        td_error = y-Q
        loss = torch.mean(td_error**2)

        # update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # gradient clipping
        self.optimizer.step()

        # soft update target network
        if self.step_count % self.target_update_freq == 0:
            soft_update(self.target_model, self.model, self.target_smooth_factor)

        # log data
        self.learn_count += 1
        if self.logger is not None:
            self.logger.add_scalar('Critic/Loss', loss, self.learn_count)
