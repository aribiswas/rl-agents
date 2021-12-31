import numpy as np
import heapq


class ReplayMemory:
    """
    Replay memory. This implementation uses a list to store experiences.
    """

    def __init__(self, maxlen=int(1e6)):
        """
        Initialize a replay memory for storing experiences.
        :param maxlen: integer
        """
        self.memory = [[] for _ in range(maxlen)]
        self.maxlen = maxlen
        self.mem_idx = 0

    def append(self, entry):
        """
        Add experiences to replay memory.

        Time complexity : O(1)

        :param entry: object
        :return: None
        """
        # Append entry to memory. If replay is full, append from beginning
        # O(1) complexity
        if self.mem_idx >= self.maxlen:
            self.mem_idx = 0
        self.memory[self.mem_idx] = entry

        # increment last index
        self.mem_idx += 1

    def sample(self, batch_size):
        """
        Get randomly sampled experiences.

        Time complexity : O(K) where K = batch size

        :param batch_size: integer
        :return: minibatch (list)
        """
        batch_idxs = np.random.choice(self.mem_idx + 1, batch_size)
        minibatch = [self.memory[i] for i in batch_idxs]
        return minibatch

    def len(self):
        """
        Return the current size of internal memory.
        :return: current length of replay memory (integer)
        """
        return self.mem_idx + 1


class PrioritizedReplayMemory:
    """
    Prioritized replay memory. This implementation utilizes a list to store experiences and a sum tree to store
    experience priorities.
        append - O(logN)
        sample - O(KlogN)
        update - O(KlogN)
    where N = replay length and K = batch size
    """

    def __init__(self, alpha=0.6, beta=0.5, beta_factor=0.001, eps=0.0001, maxlen=int(1e6)):
        """
        Initialize prioritized replay memory.

        :param alpha: Priority exponent (float)
        :param beta: Importance sampling weight exponent (float)
        :param beta_factor: Beta rise factor (float)
        :param eps: TD error offset (float)
        :param maxlen: Maximum capacity of replay memory (integer)
        """
        # memory stores the experiences in circular fashion
        self.memory = [[] for _ in range(maxlen)]
        # tree stores the priority p^a at each leaf node. Parent nodes store the sum of child node priorities.
        # There are N=maxlen leaf nodes, and 2*N-1 total nodes.
        self.tree = [0 for _ in range(2 * maxlen - 1)]
        self.alpha = alpha
        self.beta = beta
        self.beta_factor = beta_factor
        self.eps = eps
        self.maxlen = maxlen
        self.mem_index = 0
        self.max_priority = 1

    def sum_priority(self):
        """
        Obtain sum(p^a) across all elements in the sum tree. This is obtained in O(1) time by indexing into the first
        element of the tree.

        Time complexity : O(1)

        :return: sum priority (float)
        """
        return self.tree[0]

    def append(self, entry):
        """
        Append new entry to the replay memory with maximal priority. After the append operation, the sum tree priorities
        are updated.

        Time complexity : O(logN)

        :param entry: object
        :return: None
        """
        # Append entry to memory. If replay is full, append from beginning
        # O(1) complexity
        if self.mem_index >= self.maxlen:
            self.mem_index = 0
        self.memory[self.mem_index] = entry

        # store entry with maximal priority in sum tree
        # O(logN) complexity
        tree_index = self.mem_index + self.maxlen - 1
        self._store_priority(tree_index, self.max_priority)

        # incremement index
        self.mem_index += 1

    def sample(self, batch_size):
        """
        Obtain a prioritized sample from the replay memory.

        Prioritized sampling samples from the probability distribution P = p^a / sum(p^a) where p = priority of entry,
        and a = priority exponent. To efficiently sample from this distribution, the range [0, sum(p^a)] is divided
        into K equal segments, where K = batch size, and one entry is sampled uniformly from each segment. This type
        of band sampling is an approximation of the above distribution. The resultant minibatch has elements with
        diverse priorities low, med, high, etc.

        Time complexity:
        The priority sum sum(p^a) is obtained from the sum tree's first element in O(1) time.
        For each segment, the sampled uniform priority is searched in the sum tree and the entry with the closest
        priority is chosen. This occurs in O(logN) time.

        The overall complexity of sampling is O(KlogN) where K = batch size, and N = size of replay.

        :param batch_size: Batch size of sample (integer)
        :return: mini batch (list), batch indices (integer), importance sampling weights (float)
        """
        minibatch = []
        indices = []
        weights = []
        segment_size = self.sum_priority() / batch_size

        for idx in range(batch_size):
            # compute the edges of the segment
            prio_low, prio_high = idx * segment_size, (idx+1) * segment_size

            # perform uniform sampling to get a priority for this segment
            prio_sample = np.random.uniform(prio_low, prio_high)

            # sample an entry from the sum tree using the priority - O(logN) complexity
            index, priority, entry = self._search_priority(0, prio_sample)

            # compute importance sampling weight
            w = (1 / (self.mem_index * priority) ** self.beta)
            self.beta *= self. beta_factor

            # append data to mini batch
            minibatch.append(entry)
            indices.append(index)
            weights.append(w)

        return minibatch, indices, weights

    def update(self, indices, errors):
        """
        Update the replay memory with new TD errors. This will update the priorities in the sum tree at the specified
        indices.

        Time complexity O(KlogN).

        :param indices: Batch indices (integer)
        :param errors: New TD errors (float)
        :return: None
        """
        for idx, err in zip(indices, errors):
            abs_error = abs(err) + self.eps
            new_priority = abs_error ** self.alpha
            self._store_priority(idx, new_priority)

    def _store_priority(self, tree_index, priority):
        """
        Update the sum tree with new priority at a specified index.

        Time complexity : O(logN)

        :param tree_index: Index to update in sum tree (integer)
        :param priority: Priority value (float)
        :return: None
        """
        # update the priority at the tree_index
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # traverse upward through the sum tree from the tree index and update priorities of the parent nodes.
        # This involves traversing one node at each level of the sum tree. The time complexity is equal to the height
        # of the tree (complete binary tree), i.e. O(logT) or O(logN)
        # where T = 2*N-1 is the total number of nodes in the tree, and N = replay length.
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

        # update max priority
        if priority > self.max_priority:
            self.max_priority = priority

    def _search_priority(self, root_index, value):
        """
        Search the sum tree for a priority value.

        Time complexity : O(logN)

        :param root_index: Start tree index of search (integer)
        :param value: Search value (float)
        :return: tree index (index), priority (float), entry (object)
        """
        left_index = 2 * root_index + 1
        right_index = left_index + 1

        if left_index >= len(self.tree):
            tree_index = root_index
            mem_index = tree_index - self.maxlen + 1
            return tree_index, self.tree[tree_index], self.memory[mem_index]

        # recursively search for the priority value. The time complexity depends on the height of the tree,
        # i.e. is O(logT) or O(logN)
        if value < self.tree[left_index]:
            return self._search_priority(left_index, value)
        else:
            return self._search_priority(right_index, value-self.tree[left_index])

    def len(self):
        """
        Return the current size of internal memory.
        :return: current length of replay memory (integer)
        """
        return self.mem_index + 1


class PrioritizedReplayMemoryV2:
    """
    Prioritized replay memory. This implementation utilizes a min heap to store entries and samples directly from the
    probability distribution P = p^a / sum(p^a) in O(N) time. This is a simpler implementation but is less efficient.

    Time complexity:
        append - O(logN)
        sample - O(N)
        update - O(K)
    where N = replay length and K = batch size
    """

    def __init__(self, alpha=0.5, beta=0.5, beta_factor=0.001, eps=0.0001, maxlen=int(1e6)):
        self.memory = []
        self.alpha = alpha
        self.beta = beta
        self.beta_factor = beta_factor
        self.eps = eps
        self.maxlen = maxlen
        self.index = -1
        self.max_priority = 1

    def append(self, entry):
        # increment the index
        self.index += 1

        # if buffer is full, pop the lowest priority entry - complexity is O(log N)
        if self.index > self.maxlen:
            self.index = self.maxlen
            heapq.heappop(self.memory)

        # push the entry to memory, maintaining the heap invariant - complexity is O(log N)
        prio = self.max_priority
        heapq.heappush(self.memory, (prio, self.index, entry))

    def sample(self, batch_size):
        # compute probabilities - complexity is O(N) where N = len(memory)
        prios = [self.memory[i][0] ** self.alpha for i in range(self.index+1)]
        sum_prios = sum(prios)
        probs = [p/sum_prios for p in prios]

        # sample prioritized batch indices
        indices = np.random.choice(self.index+1, size=batch_size, p=probs, replace=False)

        # entry batch - complexity is O(K) where K = len(indices)
        minibatch = [self.memory[i][2] for i in indices]

        # importance sampling weights batch - complexity is O(K) where K = len(indices)
        weights = [1/((self.index+1) * probs[i]) ** self.beta for i in indices]

        # update beta
        self.beta *= self.beta_factor

        return minibatch, indices, weights

    def update(self, indices, errors):
        # update the priorities - complexity is O(K) where K = len(indices)
        for idx, err in zip(indices, errors):
            prio = abs(err) + self.eps
            _, i, entry = self.memory[idx]
            self.memory[idx] = (prio, i, entry)
            if prio > self.max_priority:
                self.max_priority = prio

        # heapify the memory
        heapq.heapify(self.memory)

    def len(self):
        """
        Return the current size of internal memory.
        :return: current length of replay memory (integer)
        """
        return self.index + 1
