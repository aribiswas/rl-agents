# Train LunarLanderContinuous-v2 with TD3 agent.

import gym
import torch
import argparse
import collections
from td3.torch_td3 import TD3Agent
from models.torch_models import QValueFunction, DeterministicActor
from torch.utils.tensorboard import SummaryWriter

# Set up
torch.manual_seed(0)
logdir = 'logs/lunar_lander_td3'


def train_lunar_lander(max_episodes=1000, max_steps=500):
    logger = SummaryWriter(logdir)
    env = gym.make('LunarLanderContinuous-v2')
    actor_model = DeterministicActor(state_dim=env.observation_space.shape,
                                     action_dim=env.action_space.shape)
    critic_model_1 = QValueFunction(state_dim=env.observation_space.shape,
                                    action_dim=env.action_space.shape)
    critic_model_2 = QValueFunction(state_dim=env.observation_space.shape,
                                    action_dim=env.action_space.shape)
    agent = TD3Agent(actor_model,
                     critic_model_1,
                     critic_model_2,
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
                     logger=logger)

    # reward window for averaging
    reward_window = collections.deque([], maxlen=100)

    # training loop
    for ep_ct in range(max_episodes):
        state = env.reset()

        sum_reward = 0
        step_ct = 0
        while step_ct <= max_steps:
            env.render()
            action = agent.action(state, clip=(-1, 1), add_noise=True)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            sum_reward += reward
            step_ct += 1
            if done:
                break

        # log and print episode info
        reward_window.append(sum_reward)
        avg_reward = sum(reward_window) / len(reward_window)
        logger.add_scalar('Episode/Reward', sum_reward, ep_ct)
        print(f'Episode={ep_ct}, Episode Reward={sum_reward}, Average Reward={avg_reward}, Steps={step_ct}')

        # exit if training is complete
        if avg_reward >= 200.0:
            break

    env.close()
    logger.close()


def main(args):
    train_lunar_lander(args.max_episodes, args.max_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gym Lunar Lander TD3.')
    parser.add_argument('--max_episodes', metavar='max_episodes', type=int, nargs='?', default=1000,
                        help='Max episodes.')
    parser.add_argument('--max_steps', metavar='max_steps', type=int, nargs='?', default=500,
                        help='Max steps.')
    args = parser.parse_args()
    main(args)
