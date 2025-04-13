from typing import Dict
import torch
import numpy as np
import copy
import os
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.episode_sampler import EpisodeBatchSampler


class MetaworldDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            codebook_dir, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            phase='train',
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
        print(f"Train mask: {train_mask}")
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
        self.codebook_dir = codebook_dir
        self.phase = phase
        self._load_codebook_data()

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
    
    def _load_codebook_data(self):
        # Load all codebook embeddings
        if self.phase == 'train':
            prefix = 'train_'
        else:
            prefix = 'val_'
        
        codebook_files = sorted(
            [f for f in os.listdir(self.codebook_dir) 
            if f.startswith(prefix) and f.endswith('.npy')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        print(f"{self.phase.capitalize()} codebook files: {codebook_files}")
    
        self.code_embeddings = []
        # code_emb = torch.empty(0)
        for fname in codebook_files:
            # print(os.path.exists(os.path.join(self.codebook_dir, fname)))
            path = os.path.join(self.codebook_dir, fname)
            self.code_embeddings.append(np.load(path))
            
        self.code_embeddings = np.concatenate(self.code_embeddings, axis=1)
        self.code_embeddings = torch.from_numpy(self.code_embeddings).squeeze(0)

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
        if (idx+1)%50==0:
            sample = self.sampler.sample_sequence(idx-1)
            code_clip = self.code_embeddings[idx-2:idx+1]
        elif idx%50==0:
            sample = self.sampler.sample_sequence(idx+1)
            code_clip = self.code_embeddings[idx: idx + 3]
            # first_token = torch.tensor([0], dtype=code_clip.dtype)
            # code_clip = torch.cat([first_token, code_clip], dim=0)
        else:
            sample = self.sampler.sample_sequence(idx) # idx 0 -449
            # print('idx',idx)
            # print(f"[Debug] Sample index {idx}: 'point_cloud' shape: {torch_data['obs']['point_cloud'].shape}") 4 1024 6
            # print(f"[Debug] Sample index {idx}: 'agent_pos' shape: {torch_data['obs']['agent_pos'].shape}") 4 9 
            # print(f"[Debug] Sample index {idx}: 'action' shape: {torch_data['action'].shape}") 4 4
            
            # code_clip = self.code_embeddings.view(-1, 1)[idx + 49:idx + 52] # if state and pc correspondence to 2nd code
            # code emedding 10,50 - 500,1
            # print('code_embeddings', self.code_embeddings.shape) 450
            code_clip = self.code_embeddings[idx-1:idx + 2]
        
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
    
        token_length = 3
        # p = torch.rand(1).item()
        # token_length = 3 if p >= self.cond_p else torch.randint(3, full_token_length, (1,)).item()
        pad_length = 50 - token_length
        pad_tokens = torch.full((pad_length,), self.pad_token, dtype=code_clip.dtype)

        tokens = torch.cat([code_clip, pad_tokens])
        tokens[token_length] = self.eos_token
        
        return {
            'm_tokens': tokens,
            # 'state_clip': state_clip,
            # Preserve original DP3 data structure
            'pc': torch_data['obs']['point_cloud'],
            'state': torch_data['obs']['agent_pos'],
            # 'obs': torch_data['obs'],
            'm_tokens_len': torch.tensor(token_length)
            # 'action': torch_data['action']
        }
    
        # return torch_data


 