import unittest
import numpy
from utils.replay import PrioritizedExperienceReplay
from collections import namedtuple


class PrioritizedExperienceReplayTestCase(unittest.TestCase):
    def test_construction(self):
        replay = PrioritizedExperienceReplay(max_len=10)
        self.assertEqual(replay.memory, {})
        self.assertEqual(replay.max_len, 10)

    def test_append(self):
        replay = PrioritizedExperienceReplay(max_len=10)
        experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])
        state = numpy.random.rand(4, 1)
        action = numpy.random.rand(2, 1)
        reward = numpy.random.rand(1, 1)
        next_state = numpy.random.rand(4, 1)
        done = numpy.array(0)
        memory = {}

        for ct in range(5):
            memory[ct] = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
            replay.append(state, action, reward, next_state, done)
        self.assertEqual(list(replay.memory.keys()), [0, 1, 2, 3, 4])
        self.assertDictEqual(replay.memory, memory)

        for ct in range(5, 10):
            memory[ct] = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
            replay.append(state, action, reward, next_state, done)
        replay.append(state, action, reward, next_state, done)
        memory.pop(0)
        memory[10] = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.assertEqual(list(replay.memory.keys()), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertDictEqual(replay.memory, memory)

    def test_sample(self):
        replay = PrioritizedExperienceReplay(max_len=10)
        for ct in range(10):
            state = numpy.random.rand(4, 1)
            action = numpy.random.rand(2, 1)
            reward = numpy.random.rand(1, 1)
            next_state = numpy.random.rand(4, 1)
            done = numpy.array(0)
            replay.append(state, action, reward, next_state, done)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, priority_batch = \
            replay.sample(batch_size=5)
        self.assertEqual(len(state_batch), 2)
        self.assertEqual(len(action_batch), 2)
        self.assertEqual(len(reward_batch), 2)
        self.assertEqual(len(next_state_batch), 2)
        self.assertEqual(len(done_batch), 2)
        self.assertEqual(len(priority_batch), 2)


if __name__ == '__main__':
    unittest.main()
