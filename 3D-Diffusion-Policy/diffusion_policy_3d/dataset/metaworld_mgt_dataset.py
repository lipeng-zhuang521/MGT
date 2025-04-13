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
            # codebook_dir, 
            horizon=12,
            pad_before=11,
            pad_after=11,
            seed=42,
            val_ratio=0.0,
            # phase='train',
            max_train_episodes=None,
            cond_p =0.5,
            pad_token = 1025,
            eos_token = 1024
            ):
        super().__init__()
        self.cond_p = cond_p
        self.pad_token = pad_token
        self.eos_token = eos_token
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
     
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        # self.codebook_dir = codebook_dir
        # self.phase = phase
        # self._load_codebook_data()

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
    
    # def _load_codebook_data(self):
    #     # Load all codebook embeddings
    #     if self.phase == 'train':
    #         prefix = 'train_'
    #     else:
    #         prefix = 'val_'
        
    #     codebook_files = sorted(
    #         [f for f in os.listdir(self.codebook_dir) 
    #         if f.startswith(prefix) and f.endswith('.npy')],
    #         key=lambda x: int(x.split('_')[1].split('.')[0])
    #     )
    #     print(f"{self.phase.capitalize()} codebook files: {codebook_files}")
    
    #     self.code_embeddings = []
    #     # code_emb = torch.empty(0)
    #     for fname in codebook_files:
    #         # print(os.path.exists(os.path.join(self.codebook_dir, fname)))
    #         path = os.path.join(self.codebook_dir, fname)
    #         self.code_embeddings.append(np.load(path))
            
    #     self.code_embeddings = np.concatenate(self.code_embeddings, axis=1)
    #     self.code_embeddings = torch.from_numpy(self.code_embeddings).squeeze(0)

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
        # token_length = 3
        # p = torch.rand(1).item()
        # token_length = 3 if p >= self.cond_p else torch.randint(3, full_token_length, (1,)).item()
        # pad_length = 50 - token_length
        # pad_tokens = torch.full((pad_length,), self.pad_token, dtype=code_clip.dtype)

        # tokens = torch.cat([code_clip, pad_tokens])
        # tokens[token_length] = self.eos_token
        
        return torch_data
    
        # return torch_data


 