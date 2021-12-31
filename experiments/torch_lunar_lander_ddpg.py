# Train LunarLanderContinuous-v2 with DDPG agent.

import gym
import torch
import argparse
import collections
from ddpg.torch_ddpg import DDPGAgent
from models.torch_models import QValueFunction, DeterministicActor
from torch.utils.tensorboard import SummaryWriter

# Set up
torch.manual_seed(0)
logdir = 'logs/lunar_lander_ddpg'


def train_lunar_lander(max_episodes=1000, max_steps: object = 500):
    logger = SummaryWriter(logdir)
    env = gym.make('LunarLanderContinuous-v2')
    actor_model = DeterministicActor(state_dim=env.observation_space.shape,
                                     action_dim=env.action_space.shape)
    critic_model = QValueFunction(state_dim=env.observation_space.shape,
                                  action_dim=env.action_space.shape)
    agent = DDPGAgent(actor_model,
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
    parser = argparse.ArgumentParser(description='Gym Lunar Lander DDPG.')
    parser.add_argument('--max_episodes', metavar='max_episodes', type=int, nargs='?', default=1000,
                        help='Max episodes.')
    parser.add_argument('--max_steps', metavar='max_steps', type=int, nargs='?', default=500,
                        help='Max steps.')
    args = parser.parse_args()
    main(args)
