import torch
import torch.optim as optim
import numpy as np
from utils.replay import PrioritizedReplayMemory
from utils.exploration import OUNoise
from utils.torch_utils import to_tensor, soft_update, replay_entry


class DDPGAgent:

    def __init__(self,
                 actor_model,
                 critic_model,
                 replay_len=int(1e6),
                 discount_factor=0.99,
                 theta=0.15,
                 sigma=0.2,
                 sigma_decay=1e-6,
                 sigma_min=0.01,
                 mu=0,
                 batch_size=128,
                 actor_learn_rate=0.001,
                 critic_learn_rate=0.001,
                 target_update_freq=1,
                 target_smooth_factor=0.001,
                 logger=None):

        # initialize agent parameters
        self.step_count = 0
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.actor_learn_rate = actor_learn_rate
        self.critic_learn_rate = critic_learn_rate
        self.target_update_freq = target_update_freq
        self.target_smooth_factor = target_smooth_factor

        # exploration
        num_actions = max(actor_model.action_dim)
        self.ounoise = OUNoise(num_actions, theta, sigma, sigma_decay, sigma_min, mu)

        # create local and target networks
        self.actor_model = actor_model
        self.target_actor_model = actor_model
        self.critic_model = critic_model
        self.target_critic_model = critic_model

        # initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=actor_learn_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=critic_learn_rate)

        # initialize experience buffer
        self.replay = PrioritizedReplayMemory(maxlen=replay_len)

        # logger
        self.logger = logger
        self.learn_count = 0

    def action(self, state, clip=(-1, 1), add_noise=False):
        # obtain network output y = [Q(s,a1), ... Q(s,an)]
        with torch.no_grad():
            y = self.actor_model(state).numpy()

        # compute noisy action
        if add_noise is True:
            noise = self.ounoise.step()
        else:
            noise = np.zeros(self.actor_model.action_dim)
        action = y + noise
        action = np.clip(action, clip[0], clip[1])

        # log exploration
        if self.logger is not None:
            for ct in range(len(noise)):
                self.logger.add_scalar(f'Exploration/OU Noise({ct})', noise[ct], self.step_count)

        return action

    def step(self, state, action, reward, next_state, done):
        # increase step count
        self.step_count += 1

        # add experience to replay memory
        entry = replay_entry(state, action, reward, next_state, done)
        self.replay.append(entry)

        # learn from experiences
        if self.replay.len() >= self.batch_size:
            self.learn()

    def learn(self):
        """
        Train the agent.
        :return: None
        """
        # create batch experiences for learning
        minibatch, indices, weights = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, dones = to_tensor(minibatch)

        with torch.no_grad():
            # next action
            next_actions = self.target_actor_model(next_states)

            # Q'(s+1,a+1) -> q value for next state and action
            targetQ = self.target_critic_model(next_states, next_actions)

            # y = r + gamma * Q'(s+1,a+1)
            y = rewards + self.discount_factor * targetQ * (1-dones)

        # Q(s,a) -> q value for action from local policy
        Q = self.critic_model(states, actions)

        # calculate critic loss
        td_error = (y-Q).squeeze(1)
        weighted_td_error = td_error * torch.as_tensor(weights)
        critic_loss = torch.mean(weighted_td_error ** 2)

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 1)  # gradient clipping
        self.critic_optimizer.step()

        # freeze critic before actor loss computation
        for p in self.critic_model.parameters():
            p.requires_grad = False

        # calculate actor loss
        actor_loss = -self.critic_model(states, self.actor_model(states)).mean()

        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), 1)  # gradient clipping
        self.actor_optimizer.step()

        # Unfreeze critic
        for p in self.critic_model.parameters():
            p.requires_grad = True

        # soft update target network
        if self.step_count % self.target_update_freq == 0:
            soft_update(self.target_actor_model, self.actor_model, self.target_smooth_factor)
            soft_update(self.target_critic_model, self.critic_model, self.target_smooth_factor)

        # update priorities in replay
        self.replay.update(indices, td_error.detach().tolist())

        # log data
        self.learn_count += 1
        if self.logger is not None:
            self.logger.add_scalar('Loss/Actor', actor_loss.detach().numpy(), self.learn_count)
            self.logger.add_scalar('Loss/Critic', critic_loss.detach().numpy(), self.learn_count)
