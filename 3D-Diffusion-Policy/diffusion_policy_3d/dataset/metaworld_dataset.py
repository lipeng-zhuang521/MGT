from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.episode_sampler import EpisodeBatchSampler


class MetaworldDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        # self.sampler= EpisodeBatchSampler(
        #     replay_buffer=self.replay_buffer,
        #     episode_mask = train_mask)
        
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    # def get_validation_full_dataset(self):
    #     val_set = copy.copy(self)
    #     val_set.sampler = EpisodeBatchSampler(
    #         replay_buffer=self.replay_buffer,
    #         episode_mask=~self.train_mask
    #     )
    #     val_set.train_mask = ~self.train_mask
    #     val_set.horizon = None  # Horizon is irrelevant for full episodes
    #     return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud, 
                'agent_pos': agent_pos, 
            },
            'action': sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        # print(f"[Debug] Sample index {idx}: 'point_cloud' shape: {torch_data['obs']['point_cloud'].shape}") 4 1024 6
        # print(f"[Debug] Sample index {idx}: 'agent_pos' shape: {torch_data['obs']['agent_pos'].shape}") 4 9 
        # print(f"[Debug] Sample index {idx}: 'action' shape: {torch_data['action'].shape}") 4 4
        return torch_data

    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     # sample = self.sampler.sample_sequence(idx)
    #     sequence_index = self.sampler.get_episode_index(idx)
    #     # print('sequence_index', sequence_index)
    #     sequence_sample = self.sampler.sample_sequence(sequence_index)
         
    #     data = self._sample_to_data(sequence_sample)
    #     # print('data', data['action'].shape)
    #     torch_data = dict_apply(data, torch.from_numpy)
    #     # print(f"[Debug] Sample index {idx}: 'point_cloud' shape: {torch_data['obs']['point_cloud'].shape}")
    #     # print(f"[Debug] Sample index {idx}: 'agent_pos' shape: {torch_data['obs']['agent_pos'].shape}")
    #     # print(f"[Debug] Sample index {idx}: 'action' shape: {torch_data['action'].shape}")
    #     # [Debug] Sample index 8: 'point_cloud' shape: torch.Size([200, 1024, 6])
    #     # [Debug] Sample index 8: 'agent_pos' shape: torch.Size([200, 9])
    #     # [Debug] Sample index 8: 'action' shape: torch.Size([200, 4])
    #     return torch_data
 