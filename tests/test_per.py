import unittest
import numpy
from utils.replay import PrioritizedReplayMemory
from collections import namedtuple


class PrioritizedReplayMemoryTestCase(unittest.TestCase):
    def test_construction(self):
        replay = PrioritizedReplayMemory(maxlen=10)
        self.assertEqual(replay.memory, [[] for _ in range(10)])
        self.assertEqual(replay.tree, [0 for _ in range(2*10-1)])
        self.assertEqual(replay.alpha, 0.6)
        self.assertEqual(replay.beta, 0.5)
        self.assertEqual(replay.beta_factor, 0.001)
        self.assertEqual(replay.eps, 0.0001)
        self.assertEqual(replay.maxlen, 10)
        self.assertEqual(replay.mem_index, 0)
        self.assertEqual(replay.max_priority, 1)

    def test_append(self):
        replay = PrioritizedReplayMemory(maxlen=10)
        experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])
        state = numpy.random.rand(4, 1)
        action = numpy.random.rand(2, 1)
        reward = numpy.random.rand(1, 1)
        next_state = numpy.random.rand(4, 1)
        done = numpy.array(0)
        memory = [[] for _ in range(10)]

        for ct in range(5):
            entry = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
            memory[ct] = entry
            replay.append(entry)
        self.assertEqual(replay.memory, memory)

        for ct in range(5, 10):
            entry = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
            memory[ct] = entry
            replay.append(entry)
        replay.append(entry)
        memory[0] = entry
        self.assertEqual(replay.memory, memory)

    def test_sample(self):
        replay = PrioritizedReplayMemory(maxlen=10)
        experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])
        for ct in range(10):
            state = numpy.random.rand(4, 1)
            action = numpy.random.rand(2, 1)
            reward = numpy.random.rand(1, 1)
            next_state = numpy.random.rand(4, 1)
            done = numpy.array(0)
            entry = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
            replay.append(entry)
        minibatch, indices, weights = replay.sample(batch_size=5)
        self.assertEqual(len(minibatch), 5)
        self.assertEqual(len(indices), 5)
        self.assertEqual(len(weights), 5)


if __name__ == '__main__':
    unittest.main()
