# This code is shamelessly stolen from
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import numpy as np
import random
import pickle

class ReplayBuffer:
    BUFFER_SAVE_DIR = "replay_buffers"

    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        self.SAVE_BUCKET = 5000

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        # fill the data cyclically, if the list is not yet complete append it
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones)
        )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)
    
    def clear_buffer(self):
        del self._storage[:]

    def save_buffer(self):
        bucket_counter = 0
        while bucket_counter * self.SAVE_BUCKET < len(self._storage):
            with open(f"{ReplayBuffer.BUFFER_SAVE_DIR}/replay_buffer_{bucket_counter}.txt", 'wb') as f:
                pickle.dump(self._storage[bucket_counter * self.SAVE_BUCKET : (bucket_counter + 1) * self.SAVE_BUCKET], f)
            bucket_counter += 1
    
    def load_buffer(self):
        self.clear_buffer()
        bucket_counter = 0
        while bucket_counter * self.SAVE_BUCKET < len(self._storage):
            with open(f"{ReplayBuffer.BUFFER_SAVE_DIR}/replay_buffer_{bucket_counter}.txt", 'rb') as f:
                self._storage += pickle.load(f)
            bucket_counter += 1