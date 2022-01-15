import torch
import torch.optim as optim
import numpy as np
from utils.replay import ReplayMemory
from utils.torch_utils import to_tensor, replay_entry


class PPOAgent:

    def __init__(self,
                 actor_model,
                 critic_model,
                 actions,
                 horizon=500,
                 clip_factor=0.2,
                 discount_factor=0.99,
                 gae_factor=0.95,
                 batch_size=32,
                 num_epochs=10,
                 actor_learn_rate=0.001,
                 critic_learn_rate=0.001,
                 logger=None):

        # initialize agent parameters
        self.step_count = 0
        self.discount_factor = discount_factor
        self.gae_factor = gae_factor
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.actor_learn_rate = actor_learn_rate
        self.critic_learn_rate = critic_learn_rate
        self.horizon = horizon
        self.clip_factor = clip_factor
        self.actions = actions
        self.step_count = 0
        self.horizon_count = 0
        self.advantage_buffer = []

        # models
        self.actor_model = actor_model
        self.critic_model = critic_model

        # initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=actor_learn_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=critic_learn_rate)

        # initialize experience buffer
        self.replay = ReplayMemory(maxlen=horizon)

        # logger
        self.logger = logger
        self.actor_learn_count = 0
        self.critic_learn_count = 0

    def action(self, state, greedy=True):
        with torch.no_grad():
            action_probs = self.actor_model(state)
        if greedy:
            choice = torch.argmax(action_probs)
        else:
            dist = torch.distributions.categorical.Categorical(probs=action_probs)
            choice = dist.sample()
        return self.actions[choice]

    def step(self, state, action, reward, next_state, done):
        # increase step count
        self.step_count += 1
        self.horizon_count += 1

        # add experience to replay memory
        entry = replay_entry(state, action, reward, next_state, done)
        self.replay.append(entry)

        # learn from experiences
        if self.horizon_count >= self.horizon:
            self.learn()
            self.replay.reset()
            self.horizon_count = 0

    def learn(self):
        """
        Train the agent.
        :return: None
        """
        # extract trajectory upto horizon steps
        experiences = self.replay.all()
        states, actions, rewards, next_states, dones = to_tensor(experiences)
        advantages, rewards_to_go = self.process_trajectory(states, actions, rewards, next_states, dones)

        # store old policy log probs
        pi_old = self.actor_model(states).detach()

        # update policy in multiple optim steps
        for _ in range(self.num_epochs):
            self.actor_learn_count += 1
            self.actor_optimizer.zero_grad()
            pi = self.actor_model(states)
            ratio = torch.exp(torch.log(pi) - torch.log(pi_old))  # log probs are more stable than raw probabilities
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, min=1-self.clip_factor, max=1+self.clip_factor) * advantages
            actor_loss = torch.mean(torch.minimum(surr1, surr2))
            actor_loss.backward()
            self.actor_optimizer.step()
            # log data
            if self.logger is not None:
                self.logger.add_scalar('Loss/Actor', actor_loss.detach().numpy(), self.actor_learn_count)

        # update critic in multiple optim steps
        for _ in range(self.num_epochs):
            self.critic_learn_count += 1
            self.critic_optimizer.zero_grad()
            currentVs = self.critic_model(states)
            critic_loss = torch.mean((currentVs - rewards_to_go) ** 2)
            critic_loss.backward()
            self.critic_optimizer.step()
            # log data
            if self.logger is not None:
                self.logger.add_scalar('Loss/Critic', critic_loss.detach().numpy(), self.critic_learn_count)

    def process_trajectory(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            # compute deltas, advantages, rewards to go
            #
            # delta d_t = r_t + g * V(s+1) - V(s)
            # advantage A_t = d_t + (g*l)*d_t+1 + (g*l)^2*d_t+2 + ... + (g*l)^(T-t+1)*d_T-1
            # reward to go R_t = r_t + r_t+1 + ... + r_T
            #
            # r_t = reward at time t
            # T = horizon limit
            # g = discount factor
            # l = gae factor
            nextVs = self.critic_model(next_states)
            currentVs = self.critic_model(states)
            deltas = rewards + self.discount_factor * nextVs * (1 - dones) - currentVs
            advantages = torch.zeros(self.horizon, 1)
            rewards_to_go = torch.zeros(self.horizon, 1)
            for idx in range(self.horizon):
                exponent = 0
                isdone = 0
                for t in range(idx, self.horizon):
                    if dones[t][0] > 0:
                        isdone = 1
                    rewards_to_go[idx] += rewards[t] * (1 - isdone)
                    ad = (self.discount_factor * self.gae_factor) ** exponent * deltas[t]
                    advantages[idx] += ad
                    exponent += 1
                self.advantage_buffer.append(advantages[idx].numpy())

            # normalize the advantages
            adv_mean = np.mean(self.advantage_buffer)
            adv_std = np.std(self.advantage_buffer) + 1.e-6
            advantages = (advantages - adv_mean) / adv_std

        return advantages, rewards_to_go

    def create_batch_indices(self):
        # divide data into batches, e.g. if batch_size=4 and horizon=10 then
        # batch_indices = [ [0,1,2,3], [4,5,6,7], [8,9,10] ]
        batch_indices = []
        num_batches = self.horizon // self.batch_size
        for batch_ct in range(num_batches):
            beg = batch_ct * self.batch_size
            end = (batch_ct + 1) * self.batch_size
            batch_indices.append(np.arange(beg, end))
        # add the remaining indices
        if num_batches * self.batch_size < self.horizon:
            batch_indices.append(np.arange(num_batches * self.batch_size, self.horizon))
        return batch_indices
