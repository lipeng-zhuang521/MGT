from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops
import os
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder

from diffusion_policy_3d.model.mgt.act_vq import ActVQ
from diffusion_policy_3d.model.mgt_utils.config import vq_args_parser,trans_args_parser
from diffusion_policy_3d.model.mgt.transformer import ActTransformer
import diffusion_policy_3d.model.mgt_utils.losses as losses
from diffusion_policy_3d.model.mgt_utils.utils_model import generate_src_mask, gumbel_sample, cosine_schedule
from pathlib import Path

class MGT(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            # noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            # # VQ parameters
            # vq_num_code: int = 512,
            # vq_code_dim: int = 512,
            # vq_output_dim: int = 512,
            # vq_commit_weight: float = 0.25,
            # # Transformer parameters
            # trans_embed_dim: int = 512,
            # trans_num_layers: int = 6,
            # trans_num_heads: int = 8,
            # Observation encoder
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # parameters passed to step
            **kwargs):
        super().__init__()


        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
 
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        self.obs_encoder = obs_encoder

        self.args_vq = vq_args_parser()
        self.args_trans = trans_args_parser()
        self.vq_model = self.build_vq(self.args_vq) 
        self.trans_model = self.build_trans(self.args_trans)       
        try:
            self.load_vq_checkpoint(device='cuda' if torch.cuda.is_available() else 'cpu')
        except FileNotFoundError as e:
            print(f"Warning: {str(e)}, training from scratch")
        
        
        self.losses = losses
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        # if num_inference_steps is None:
        #     num_inference_steps = noise_scheduler.config.num_train_timesteps
        # self.num_inference_steps = num_inference_steps


        print_params(self)
    
    def build_vq(self, args_vq):
        # args = vq_args_parser()
        torch.manual_seed(args_vq.seed)
        vq_model = ActVQ(args_vq,  ## use args to define different parameters in different quantizers
            args_vq.nb_code,
            args_vq.code_dim,
            args_vq.output_emb_width,
            args_vq.down_t,
            args_vq.stride_t,
            args_vq.width,
            args_vq.depth,
            args_vq.dilation_growth_rate,
            args_vq.vq_act,
            args_vq.vq_norm)
        return vq_model

    def build_trans(self, args_trans):
        # args = trans_args_parser()
        vq_model = self.vq_model
        trans = ActTransformer(vqvae=vq_model,
                             num_vq=args_trans.nb_code,
                             embed_dim=args_trans.embed_dim_gpt,
                             comb_state_dim=args_trans.comb_state_dim,
                            #  pc_dim=args_trans.pc_dim,
                             cond_length=args_trans.cond_length,
                             block_size=args_trans.block_size,
                             num_layers=args_trans.num_layers,
                             num_local_layer=args_trans.num_local_layer,
                             n_head=args_trans.n_head_gpt,
                             drop_out_rate=args_trans.drop_out_rate,
                             fc_rate=args_trans.ff_rate)
        return trans
    
    # def load_vq(self, vq_model_dir):
    #     self.vq_model.load_state_dict(self.vq_model.state_dict())
    #     self.vq_model.to(self.device)
    #     self.vq_model.eval()
    #     # print('load vq model')
   
    def load_vq_checkpoint(self, nb_iter=None, device='cpu'):
        vq_out_dir = os.path.join(self.args_vq.out_dir, f'vq')
        if nb_iter is None:
            checkpoints = [f for f in os.listdir(vq_out_dir) if f.endswith('_net_last.pth')]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {vq_out_dir}")
                
            iters = [int(f.split('_')[0]) for f in checkpoints]
            nb_iter = max(iters)       
        ckpt_path = os.path.join(vq_out_dir, f'{nb_iter}_net_last.pth')
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

        checkpoint = torch.load(ckpt_path, map_location=device)      
        # Load weights into model
        state_dict = checkpoint['net']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('vq_model.'):
                new_k = k[len('vq_model.'):]  # Strip 'vq_model.' prefix
                new_state_dict[new_k] = v
            # else:
            #     print(f"Warning: Key {k} not found in model")
        self.vq_model.load_state_dict(new_state_dict)   
        print(f"Loaded checkpoint from iteration {nb_iter}")
        return 

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory
    
    def vq_encode(self, x):
        """
        Encode the input data using the VQ model.
        Args:
            x: Input data to be encoded.
        Returns:
            Encoded data.
        """
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0
        x_d = self.vq_model.encode(x)
        return x_d


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result



    def predict_MGT_action(self, batch):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if 'action' in batch:
            action = batch['action'].to(self.device)
            target_token = self.vq_model.encode(action) # new code
            batch_size = action.shape[0]# new code
            m_tokens = self.regression_token_process(target_token) # new code
            m_tokens_len = torch.full((batch_size,), 2, dtype=m_tokens.dtype).to(m_tokens.device)# new code
            # m_tokens_len = batch['m_tokens'].to(self.device)
        else:
            m_tokens = batch['m_tokens'].to(self.device)
            m_tokens_len = batch['m_tokens_len'].to(self.device)
        # pc = batch['pc'].to(self.device)
        # state = batch['state'].to(self.device)
        # m_tokens_len = batch['m_tokens_len'].to(self.device)      
        # print("Type of m_tokens:", type(m_tokens))
        # print("Type of m_tokens_len:", type(m_tokens_len))
        # batch['obs'] = {}
        # batch['obs']['point_cloud'] = batch['pc']
        # batch['obs']['agent_pos'] = batch['state']
        # print('obs shape:', batch['obs']['point_cloud'].shape) 128,12,1024,6
        batch['obs']['point_cloud'] = batch['obs']['point_cloud'][:, :2, ...]  # only use first 5 frames
        batch['obs']['agent_pos'] = batch['obs']['agent_pos'][:, :2, ...]

        # m_tokens = batch['m_tokens'].to(self.device)  
        # pc = batch['pc'].to(self.device)
        # # print('pc',pc.shape)
        # state = batch['state'].to(self.device)
        # # print('state',state.shape)
        # # pc torch.Size([128, 4, 1024, 6])
        # # state torch.Size([128, 4, 9])
        # # eval pc torch.Size([1, 4, 1024, 6])
        # # eval state torch.Size([1, 4, 9])

        # m_tokens_len = batch['m_tokens_len'].to(self.device) 
        target = m_tokens.int()  
        target = target.cuda() 
          
        obs_dict = {
        'point_cloud': batch['obs']['point_cloud'],
        'agent_pos': batch['obs']['agent_pos']  
    }
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
    
        # need to check!!!!!!!!!
        
            # value = next(iter(nobs.values()))
            # B, To = value.shape[:2]
        this_nobs = dict_apply(nobs, lambda x: x[:,:2,...].reshape(-1,*x.shape[2:]))
        
        nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            # nobs_features = nobs_features.reshape(B, To, -1)
               
        # first_tokens = target[:, 0]
        # print('first_tokens',first_tokens) # first_tokens torch.Size([128]) 
        batch_size, max_len = target.shape[:2]
        nobs_features = nobs_features.reshape(batch_size, 2, -1)
        # nobs_features = nobs_features.reshape(batch_size, -1, 4)
        # print('nobs_features',nobs_features.shape) 128 128 4
        # nobs_features torch.Size([128, 4, 128])
        # in eval nobs_features torch.Size([1, 4, 128])
    # Generate source mask (adapt based on your sequence length handling)
        src_mask = generate_src_mask(
            max_len, m_tokens_len + 1
        )
        # unnormalize prediction
        # naction_pred = nsample[...,:Da]
        with torch.no_grad():
            sampled_tokens = self.trans_model.fast_sample(
                # first_tokens=first_tokens,
                src_mask=src_mask,
                # pc = nobs['point_cloud'],
                comb_state= nobs_features,
                m_length=None,
                step=1,
                gt=target
            )
        
        decoded_tokens = sampled_tokens[:,0:2]  # (B, 3)
               
        decoded_actions = self.vq_model.decode(decoded_tokens)  # (128,12,4)
        # print('mgt',decoded_actions.shape)
        decoded_target = target[:,0:2]  # (B, 3)
        # print('decoded_target',decoded_target)
        target = self.vq_model.decode(decoded_target)   # (128,12,4)
        # print('mgt',target.shape)
        norm_decoded_actions = self.normalizer['action'].unnormalize(decoded_actions)
        norm_target = self.normalizer['action'].unnormalize(target)
    # target also need to detoken
    # Decode latents to action space
        # latents = latents.permute(0, 2, 1)  # (B, code_dim, T)
        # decoded_actions = self.vq_model.decoder(latents).permute(0, 2, 1)  # (B, T, action_dim)
              
        # action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        # get action
        # start = To - 1
        # end = start + self.n_action_steps
        # action = action_pred[:,start:end]
        
        # get prediction
        return {
        'action_pred': norm_decoded_actions,
        'sampled_tokens': sampled_tokens,
        'action_gt': norm_target
    }

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        

        loss_dict = {
                'bc_loss': loss.item(),
            }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict
    
    def compute_vq_loss(self, batch):
        # Process observations
        # obs_features = self.obs_encoder(batch['obs'])
        Loss = self.losses.ReConsLoss(recons_loss=self.args_vq.recons_loss, pos_dim=[0, 1, 2], rot_state=False)
        # Process actions through VQ-Transformer
        # actions = batch['action']
        actions = self.normalizer['action'].normalize(batch['action'])
        pred_actions, loss_commit, perplexity = self.vq_model(actions)        
        # Calculate reconstruction loss
        loss_action = Loss(
            pred_actions, 
            batch['action']
        )
        
        # Total loss
        total_loss = loss_action + self.args_vq.commit * loss_commit
        
        return total_loss, {
            'loss_recon': loss_action.item(),
            'loss_commit': loss_commit.item(),
            'perplexity': perplexity.item()
        }
    
    @staticmethod
    def get_acc(cls_pred, target, mask):
        cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
        target_all = torch.masked_select(target, mask)
        probs = torch.softmax(cls_pred, dim=-1)
        _, cls_pred_index = torch.max(probs, dim=-1)
        right_num = (cls_pred_index == target_all).sum()
        return right_num*100/mask.sum()
    
    def regression_token_process(self, token):

        token_length = torch.tensor(2).int()
        batch_size = token.shape[0]
        
        # else:
        #     token_length = torch.randint(3,  full_token_length, (1,)).view([])
        pad_length = 50 - token_length
        pad_tokens = torch.full((batch_size, pad_length,), self.vq_model.nb_code+1, dtype=token.dtype).to(token.device)
        
        padded_tokens = torch.cat([token, pad_tokens], dim=1)
        padded_tokens[:, token_length] = self.vq_model.nb_code

        return padded_tokens
    

    def compute_trans_loss(self,batch):
        # m_tokens = batch['m_tokens'].to(self.device)
        action = batch['action'].to(self.device)
        action = self.normalizer['action'].normalize(action)
        target_token = self.vq_model.encode(action) # new code
        batch_size = action.shape[0]# new code
        m_tokens = self.regression_token_process(target_token) # new code
        m_tokens_len = torch.full((batch_size,), 2, dtype=m_tokens.dtype).to(m_tokens.device)# new code
        # pc = batch['pc'].to(self.device)
        # state = batch['state'].to(self.device)
        # m_tokens_len = batch['m_tokens_len'].to(self.device)      
        # print("Type of m_tokens:", type(m_tokens))
        # print("Type of m_tokens_len:", type(m_tokens_len))
        # batch['obs'] = {}
        # batch['obs']['point_cloud'] = batch['pc']
        # batch['obs']['agent_pos'] = batch['state']
        batch['obs']['point_cloud'] = batch['obs']['point_cloud'][:, :2, ...]  # only use first 5 frames
        batch['obs']['agent_pos'] = batch['obs']['agent_pos'][:, :2, ...]
        nobs = self.normalizer.normalize(batch['obs'])
        # nactions = self.normalizer['action'].normalize(batch['action'])
        
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

            # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # print('this_nobs',this_nobs['point_cloud'].shape)
        # print('nobs_features',nobs_features.shape)

        # print('tokens',m_tokens[0]) # tokens torch.Size([128, 50])
        # print('token length',m_tokens_len[0]) # tokens torch.Size([128, 50])
        target = m_tokens.int()
        # print('clean target[0]:', target[0])
        batch_size, max_len = target.shape[:2]
        nobs_features = nobs_features.reshape(batch_size, 2, -1)
        mask = torch.bernoulli(self.args_trans.pkeep * torch.ones(target.shape, device=target.device))  # random (0,1) mask
        # mask[:, 0] = 1 # first token always 1; 1 - 'keep the token',0 - 'drop it'
        # print('mask[0]:', mask[0]) mask[0]: tensor([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0',
        seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)  # bool mask for the action length
        # print('seq_mask',seq_mask_no_end.shape,seq_mask_no_end[0]) 
        # seq_mask torch.Size([24, 16]) tensor([ True,  True,  True, False, False, False, False, False, False, False,
        #     False, False, False, False, False, False], device='cuda:0')
        mask = torch.logical_or(mask, ~seq_mask_no_end).int()
        # print('mask[0]:', mask[0])
        # mask[0]: tensor([1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0',
        #    dtype=torch.int32)
        r_indices = torch.randint_like(target, self.args_trans.nb_code)
        input_indices = mask * target + (1 - mask) * r_indices
        # print('input_indices[0]:', input_indices[0])  For positions where mask is 1, the target token is kept; where mask is 0, a random token is used.
        # input_indices[0]: tensor([4436,  812, 5235, 8192, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193,
        #     8193, 8193, 8193, 8193], device='cuda:0', dtype=torch.int32)
        mask_id = self.args_trans.nb_code + 2 # a special token
        rand_mask_probs = torch.zeros(batch_size, device=m_tokens_len.device).float().uniform_(0.5, 1)
        num_token_masked = (m_tokens_len * rand_mask_probs).round().clamp(min=1)
        # the number of tokens to force-mask in that sample
        seq_mask = generate_src_mask(max_len, m_tokens_len + 1)
        # it “extends” the valid region by one token, thereby including the end token as a valid position.
        # seq_mask = generate_src_mask(max_len, m_tokens_len)
        batch_randperm = torch.rand((batch_size, max_len), device=target.device) - seq_mask_no_end.int()
        batch_randperm = batch_randperm.argsort(dim=-1)
        # print('batch_randperm:', batch_randperm)
        mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        # print('mask_token[0]:', mask_token[0])
        # mask_token[0]: tensor([ True,  True,  True, False, False, False, False, False, False, False,
        #     False, False, False, False, False, False], device='cuda:0')
        # mask_token[..., 0] = False
        masked_input_indices = torch.where(mask_token, mask_id, input_indices)
        # print('masked_input_indices[0]:', masked_input_indices[0])
        # masked_input_indices[0]: tensor([4137, 8194, 8194, 8192, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193,
        #     8193, 8193, 8193, 8193], device='cuda:0', dtype=torch.int32)
        # seq_mask[..., 0] = False
        
        cls_pred = self.trans_model(masked_input_indices, src_mask=seq_mask, comb_state=nobs_features)[:, 0:]
        weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
        cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
        target_seq_masked = target[seq_mask_no_end]
        weight_seq_masked = weights[seq_mask_no_end]
        # print('cls_pred_seq_masked:', cls_pred_seq_masked)
        # print('target_seq_masked:', target_seq_masked)
        loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked.long(), reduction='none')
        loss_cls = (loss_cls * weight_seq_masked).sum()
    
        probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
        _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
        target_seq_masked = torch.masked_select(target, seq_mask_no_end)
        right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()
        no_mask_token = ~mask_token * seq_mask_no_end
        acc_masked = self.get_acc(cls_pred, target, mask_token) 
        acc_no_masked = self.get_acc(cls_pred, target, no_mask_token)
        acc_overall = right_seq_masked*100/seq_mask_no_end.sum()
  
        return loss_cls, {
            'loss_recon': loss_cls.item(),
            'acc_masked': acc_masked.item() ,  
            'acc_no_masked': acc_no_masked.item() ,
            'acc_overall': acc_overall.item()}


