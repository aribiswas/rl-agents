import torch
import torch.optim as optim
import numpy as np
from utils.replay import ReplayMemory
from utils.torch_utils import replay_entry, to_tensor, soft_update


class TD3Agent:

    def __init__(self,
                 actor_model,
                 critic_model1,
                 critic_model2,
                 discount_factor=0.99,
                 batch_size=128,
                 replay_length=int(1e6),
                 actor_lr=0.001,
                 critic_lr=0.001,
                 noise_mean=0,
                 noise_std=0.1,
                 noise_decay=0.001,
                 noise_min=0.01,
                 policy_update_frequency=5,
                 target_policy_smoothing_noise_mean=0,
                 target_policy_smoothing_noise_std=0.1,
                 target_policy_smoothing_noise_limit=(-1, 1),
                 target_action_limit=(-1, 1),
                 target_smooth_factor=0.001,
                 logger=None):

        self.actor_model = actor_model
        self.critic_model_1 = critic_model1
        self.critic_model_2 = critic_model2
        self.target_actor_model = actor_model
        self.target_critic_model_1 = self.critic_model_1
        self.target_critic_model_2 = self.critic_model_2
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.noise_min = noise_min
        self.policy_update_frequency = policy_update_frequency
        self.target_policy_smoothing_noise_mean = target_policy_smoothing_noise_mean
        self.target_policy_smoothing_noise_std = target_policy_smoothing_noise_std
        self.target_policy_smoothing_noise_limit = target_policy_smoothing_noise_limit
        self.target_action_limit = target_action_limit
        self.target_smooth_factor = target_smooth_factor
        self.logger = logger

        self.replay = ReplayMemory(maxlen=replay_length)

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer_1 = optim.Adam(self.critic_model_1.parameters(), lr=critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_model_2.parameters(), lr=critic_lr)

        self.step_count = 0
        self.critic_learn_count = 0
        self.actor_learn_count = 0

    def action(self, state, clip=(-1, 1), add_noise=False):
        with torch.no_grad():
            y = self.actor_model(state).numpy()

        if add_noise:
            noise = np.random.normal(loc=self.noise_mean,
                                     scale=self.noise_std,
                                     size=self.actor_model.action_dim)
            self.noise_std = np.max([self.noise_min, self.noise_std * (1-self.noise_decay)])
        else:
            noise = np.zeros(self.actor_model.action_dim)

        action = np.clip(y+noise, clip[0], clip[1])

        # log exploration
        if self.logger is not None:
            for ct in range(len(noise)):
                self.logger.add_scalar(f'Exploration/Noise S.Dev.({ct})', noise[ct], self.step_count)

        return action

    def step(self, state, action, reward, next_state, done):
        self.step_count += 1

        # append experience to replay memory
        entry = replay_entry(state, action, reward, next_state, done)
        self.replay.append(entry)

        # learn
        if self.step_count >= self.batch_size:
            self.learn()

    def learn(self):
        # sample a minibatch from the replay
        minibatch = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, dones = to_tensor(minibatch)

        # compute the targets
        with torch.no_grad():
            # next actions = clip(target_policy(next_states) + clipped_target_noise, a_min, a_max)
            next_actions = self.target_actor_model(next_states)
            target_policy_noise = torch.normal(mean=self.target_policy_smoothing_noise_mean,
                                               std=self.target_policy_smoothing_noise_std,
                                               size=(self.batch_size,) + self.actor_model.action_dim)
            target_policy_noise = torch.clamp(target_policy_noise,
                                              min=self.target_policy_smoothing_noise_limit[0],
                                              max=self.target_policy_smoothing_noise_limit[1])
            next_actions += target_policy_noise
            next_actions = torch.clamp(next_actions,
                                       min=self.target_action_limit[0],
                                       max=self.target_action_limit[1])

            # take the min of two target Qs
            target_Qs_1 = self.target_critic_model_1(next_states, next_actions)
            target_Qs_2 = self.target_critic_model_2(next_states, next_actions)
            target_Qs = torch.minimum(target_Qs_1, target_Qs_2)

            # target y = r + gamma * (1-d) * target_Q
            y = rewards + self.discount_factor * (1 - dones) * target_Qs

        # update critic 1
        current_Qs_1 = self.critic_model_1(states, actions)
        critic_loss_1 = torch.mean((current_Qs_1 - y).squeeze(1) ** 2)
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        # update critic 2
        current_Qs_2 = self.critic_model_2(states, actions)
        critic_loss_2 = torch.mean((current_Qs_2 - y).squeeze(1) ** 2)
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # log data
        self.critic_learn_count += 1
        if self.logger is not None:
            self.logger.add_scalar('Loss/Critic_1', critic_loss_1.detach().numpy(), self.critic_learn_count)
            self.logger.add_scalar('Loss/Critic_2', critic_loss_2.detach().numpy(), self.critic_learn_count)

        # delayed policy update
        if self.step_count % self.policy_update_frequency == 0:
            # freeze critic
            for p in self.critic_model_1.parameters():
                p.requires_grad = False

            # update actor
            act = self.actor_model(states)
            actor_loss = -torch.mean(self.critic_model_1(states, act))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # unfreeze critic
            for p in self.critic_model_1.parameters():
                p.requires_grad = True

            # soft update targets
            soft_update(self.target_actor_model, self.actor_model, self.target_smooth_factor)
            soft_update(self.target_critic_model_1, self.critic_model_1, self.target_smooth_factor)
            soft_update(self.target_critic_model_2, self.critic_model_2, self.target_smooth_factor)

            # log data
            self.actor_learn_count += 1
            if self.logger is not None:
                self.logger.add_scalar('Loss/Actor', actor_loss.detach().numpy(), self.actor_learn_count)

