from torch.utils.data import Sampler
from typing import List


class EpisodeBatchSampler():
    """
    Sampler that yields indices corresponding to entire episodes.
    """
    def __init__(self, replay_buffer, episode_mask,keys=None):
        """
        Args:
            replay_buffer (ReplayBuffer): The replay buffer containing episodes.
            episode_mask (np.ndarray): A boolean mask indicating which episodes to sample.
        """
        if keys is None:
            keys = list(replay_buffer.keys())
        self.replay_buffer = replay_buffer
        self.episode_mask = episode_mask
        self.episodes = self._get_episode_indices()
        self.batch_size = 1  # Each batch corresponds to one episode
        self.drop_last = False
        self.keys = list(keys)
        print(f"EpisodeSampler initialized with {len(self.episodes)} episodes.")

    def _get_episode_indices(self) -> List[List[int]]:
        """
        Retrieves a list of lists, where each sublist contains indices for a full episode.
        """
        episodes = []
        for i, include in enumerate(self.episode_mask):
            if not include:
                print(f"Episode {i + 1} is excluded by the episode_mask.")
                continue
            start_idx = 0
            if i > 0:
                start_idx = self.replay_buffer.episode_ends[i-1]
                # print(f"Episode {i + 1} start index: {start_idx}")
            # else:
                # print(f"Episode {i + 1} start index: {start_idx} (first episode)")
            end_idx = self.replay_buffer.episode_ends[i]
            # print(f"Episode {i + 1} end index: {end_idx}")
            indices = list(range(start_idx, end_idx))
            num_indices = len(indices)
            # print(f"Episode {i + 1} has {num_indices} indices.")
            episodes.append(indices)
        return episodes
    
    # def __getitem__(self, idx: int) -> List[int]:
    #     return self.episodes[idx]
    def __len__(self):
        return len(self.episodes)

    def __iter__(self):
        for episode_num, episode in enumerate(self.episodes, start=1):
            num_indices = len(episode)
            print(f"Yielding Episode {episode_num} with {num_indices} indices.")
            print('episode',episode)
            yield episode
    
    def get_episode_index(self, idx: int):
        return self.episodes[idx]
    
    def sample_sequence(self, sequence_index):
        
        buffer_start_idx, buffer_end_idx = sequence_index[0], sequence_index[-1]

        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            sample = input_arr[buffer_start_idx:buffer_end_idx+1]
            data = sample
            result[key] = data
        return result


