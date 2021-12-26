import numpy as np
from collections import namedtuple


class ReplayMemory:

    def __init__(self, state_dim, action_dim, max_len=int(1e6)):
        """
        Initialize a replay memory for storing experiences.
        Parameters
        ----------
        state_dim : tuple
            Dimension of states. E.g. (4,) or (20,20,3)
        action_dim : number
            Dimension of actions. E.g. (2,)
        max_len : number
            Capacity of memory.
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.last_idx = -1

        # memory is a dictionary that stores various elements as numpy arrays.
        # each array has dimensions max_len x <element_dimension>
        self.memory = dict(states=np.empty((self.max_len, ) + self.state_dim),
                           actions=np.empty((self.max_len, ) + self.action_dim),
                           rewards=np.empty((self.max_len, 1)),
                           next_states=np.empty((self.max_len, ) + self.state_dim),
                           dones=np.empty((self.max_len, 1)))

    def append(self, state, action, reward, next_state, done):
        """
        Add experiences to replay memory.
        Parameters
        ----------
        state : numpy array
            State of the environment.
        action : numpy array
            Actions.
        reward : numpy array
            Rewards.
        next_state : numpy array
            Next states.
        done : numpy array
            Termination flag.
        """

        # increment last index in circular fashion
        self.last_idx += 1
        if self.last_idx >= self.max_len:
            self.last_idx = 0

        # append experiences
        self.memory['states'][self.last_idx] = state
        self.memory['actions'][self.last_idx] = action
        self.memory['rewards'][self.last_idx] = reward
        self.memory['next_states'][self.last_idx] = next_state
        self.memory['dones'][self.last_idx] = done

    def sample(self, batch_size, device='cpu'):
        """
        Get randomly sampled experiences.
        Parameters
        ----------
        batch_size : number
            Batch size.
        device : char, optional
            cpu or gpu. The default is 'cpu'.
        Returns
        -------
        data : dictionary
            Dictionary with keys 'states', 'actions', 'rewards', 'next_states'
            and 'dones'
        """
        batch_idxs = np.random.choice(self.last_idx+1, batch_size)
        states_batch = self.memory['states'][batch_idxs]
        actions_batch = self.memory['actions'][batch_idxs]
        rewards_batch = self.memory['rewards'][batch_idxs]
        next_states_batch = self.memory['next_states'][batch_idxs]
        dones_batch = self.memory['dones'][batch_idxs]
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch

    def len(self):
        """
        Return the current size of internal memory.
        """
        return self.last_idx + 1


class PrioritizedExperienceReplay:

    def __init__(self, max_len=int(1e6)):
        self.max_len = max_len
        self.memory = {}

    def append(self, state, action, reward, next_state, done):
        """
        Append experience to replay memory.
        :param state: numpy array
            State of the environment
        :param action: numpy array
            Action from the agent
        :param reward: numpy array
            Reward from the environment
        :param next_state: numpy array
            Next state of the environment
        :param done: numpy array
            Done flag
        :return: None
        """
        # If replay is full, remove the experience with lowest priority
        if len(self.memory) >= self.max_len:
            self.memory.pop(min(self.memory.keys()))
        # Add experience with max priority. Higher the priority score, higher the priority
        if len(self.memory) > 0:
            priority = max(self.memory.keys()) + 1
        else:
            priority = 0
        experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])
        self.memory[priority] = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def sort(self):
        """
        Sort the prioritized replay memory.
        :return: None
        """
        self.memory = sorted(self.memory.items())

    def sample(self, batch_size, device='cpu'):
        """
        Prioritized sample from the replay memory.For prioritized sampling, the buffer is divided into K=batch_size
        equal segments. One experience is uniformly sampled from each segment to create a batch of K experiences.
        Assume that the replay memory is sorted always.
        :param batch_size: integer
            Batch size of the sample
        :param device: str
            Device
        :return: dict
            A dictionary containing batched experiences.
        """

        num_segments = int(len(self.memory) / batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        priority_batch = []
        for idx in range(num_segments):
            # choose a segment and a key
            seg_choice = np.random.choice(num_segments)
            key_choice = seg_choice * batch_size + np.random.choice(batch_size)
            key = list(self.memory.keys())[key_choice]
            experience = self.memory[key]
            # append to batch
            state_batch.append(experience.state)
            action_batch.append(experience.action)
            reward_batch.append(experience.reward)
            next_state_batch.append(experience.next_state)
            done_batch.append(experience.done)
            priority_batch.append(key)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, priority_batch

    def update(self, old_priority, new_priority):
        """
        Update the old priorities with new priorities. This also sorts the replay memory.
        :param old_priority: list
            List of old priorities
        :param new_priority: list
            List of new priorities
        :return: None
        """
        for old_p, new_p in zip(old_priority, new_priority):
            experience = self.memory[old_p]
            self.memory.pop(old_p)
            self.memory[new_p] = experience
        self.sort()

    def len(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
