# Train CartPole-v0 with DQN agent.
#
# CartPole-v0 is considered "solved" when the agent obtains an
# average reward of at least 195.0 over 100 consecutive episodes.

import gym
import torch
import argparse
import collections
from dqn.torch_dqn import DQNAgent
from models.torch_models import QNetwork
from torch.utils.tensorboard import SummaryWriter

# Set up
torch.manual_seed(10)
logdir = 'logs/cartpole_dqn'


def train_cartpole(max_episodes=1000, max_steps=500):
    logger = SummaryWriter(logdir)
    env = gym.make('CartPole-v0')
    model = QNetwork(input_size=env.observation_space.shape,
                     num_outputs=env.action_space.n)
    agent = DQNAgent(model,
                     actions=[0, 1],
                     replay_len=int(1e6),
                     discount_factor=0.99,
                     epsilon=0.9,
                     epsilon_decay=1e-4,
                     epsilon_min=0.1,
                     batch_size=64,
                     learn_rate=0.001,
                     target_update_freq=5,
                     target_smooth_factor=0.01,
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
            action = agent.action(state)
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
        if avg_reward >= 195.0:
            break

    env.close()
    logger.close()


def main(args):
    train_cartpole(args.max_episodes, args.max_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gym Cartpole DQN.')
    parser.add_argument('--max_episodes', metavar='max_episodes', type=int, nargs='?', default=1000,
                        help='Max episodes.')
    parser.add_argument('--max_steps', metavar='max_steps', type=int, nargs='?', default=500,
                        help='Max steps.')
    args = parser.parse_args()
    main(args)
