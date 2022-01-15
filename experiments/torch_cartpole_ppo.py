# Train CartPole-v0 with PPO agent.
#
# CartPole-v0 is considered "solved" when the agent obtains an
# average reward of at least 195.0 over 100 consecutive episodes.

import gym
import torch
import argparse
import collections
from ppo.torch_ppo import PPOAgent
from models.torch_models import ValueFunction, StochasticActor
from torch.utils.tensorboard import SummaryWriter

# Set up
torch.manual_seed(0)
logdir = 'logs/cartpole_ppo'


def train_cartpole(max_episodes=1000, max_steps=500):
    logger = SummaryWriter(logdir)
    env = gym.make('CartPole-v0')
    actor_model = StochasticActor(state_dim=env.observation_space.shape,
                                  action_dim=(env.action_space.n,))
    critic_model = ValueFunction(state_dim=env.observation_space.shape)
    agent = PPOAgent(actor_model,
                     critic_model,
                     actions=[0, 1],
                     horizon=128,
                     clip_factor=0.2,
                     discount_factor=0.99,
                     gae_factor=0.95,
                     batch_size=64,
                     num_epochs=50,
                     actor_learn_rate=0.001,
                     critic_learn_rate=0.001,
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
            action = agent.action(state, greedy=False)
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
    parser = argparse.ArgumentParser(description='Gym Cartpole PPO.')
    parser.add_argument('--max_episodes', metavar='max_episodes', type=int, nargs='?', default=1000,
                        help='Max episodes.')
    parser.add_argument('--max_steps', metavar='max_steps', type=int, nargs='?', default=500,
                        help='Max steps.')
    args = parser.parse_args()
    main(args)
