import unittest
import numpy
from utils.replay import ReplayMemory
from collections import namedtuple


class PrioritizedReplayMemoryTestCase(unittest.TestCase):
    def test_construction(self):
        replay = ReplayMemory(maxlen=10)
        self.assertEqual(replay.memory, [[] for _ in range(10)])
        self.assertEqual(replay.maxlen, 10)
        self.assertEqual(replay.mem_idx, 0)

    def test_append(self):
        replay = ReplayMemory(maxlen=10)
        experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])
        state = numpy.random.rand(4, 1)
        action = numpy.random.rand(2, 1)
        reward = numpy.random.rand(1, 1)
        next_state = numpy.random.rand(4, 1)
        done = numpy.array(0)
        memory = [[] for _ in range(10)]
        entry = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)

        for ct1 in range(5):
            memory[ct1] = entry
            replay.append(entry)
        self.assertEqual(replay.memory, memory)

        for ct2 in range(5, 10):
            memory[ct2] = entry
            replay.append(entry)

        replay.append(entry)
        memory[0] = entry
        self.assertEqual(replay.memory, memory)

    def test_sample(self):
        replay = ReplayMemory(maxlen=10)
        experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])
        for ct in range(10):
            state = numpy.random.rand(4, 1)
            action = numpy.random.rand(2, 1)
            reward = numpy.random.rand(1, 1)
            next_state = numpy.random.rand(4, 1)
            done = numpy.array(0)
            entry = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
            replay.append(entry)
        minibatch = replay.sample(batch_size=5)
        self.assertEqual(len(minibatch), 5)


if __name__ == '__main__':
    unittest.main()
