import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint
import os
import imageio

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                #  num_points=512
                 num_points=1024
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, save_video=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            is_start = False
            batch_size = 1
            dummy_m_tokens = torch.zeros((batch_size, 50), dtype=torch.long, device=device)

            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    #  need check !!!!!!!
                    
                    # seq_length = self.n_action_steps  # Adjust based on policy's expected sequence length
                    
                    # if is_start == False:
                    #     dummy_action = torch.zeros((batch_size, 4, 4), dtype=dtype, device=device)
                    #     dummy_m_tokens[:, 0] = policy.vq_model.encode(dummy_action)
                        # print(f'dummy_m_tokens 0',dummy_m_tokens[:, 0]) # dummy_m_tokens (1, 50)
                        # dummy_m_tokens[:, 0] = 0  # EOS token
                    dummy_m_tokens_len = torch.tensor([50] * batch_size, device=device)
                    
                    pc = obs_dict['point_cloud'].unsqueeze(0)  # Add batch dimension (1,5,1024,6)
                    state = obs_dict['agent_pos'].unsqueeze(0)  # Add batch dimension (1,5,9)
                    
                    batch = {
                        'm_tokens': dummy_m_tokens,
                        # 'pc': pc,
                        # 'state': state,
                        'm_tokens_len': dummy_m_tokens_len
                    } 
                    batch['obs'] = {
                        'point_cloud': pc,
                        'agent_pos': state
                    }
                    # obs_dict_input = {}
                    # obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    # obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                   
                    action_dict = policy.predict_MGT_action(batch)
                    is_start = True

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action_pred'].squeeze(0)
                # print(f'action',action.shape) action (12, 4)
                # print(f'action', action) # action (12, 4)
                # action_his = action[-4:, ...]
                # action_his_tensor = torch.from_numpy(action_his).to(device=device, dtype=dtype)
                # token_his = policy.vq_encode(action_his_tensor.unsqueeze(0))
                # print(f'token_his',token_his)
                # print(f'token_his',token_his.shape) # token_his (1, 1)
                # token_his = token_his[..., 0:1]
                action_exec = action[1:, ...]
                obs, reward, done, info = env.step(action_exec)
                # dummy_m_tokens[:, 0] = token_his

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])
             
            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            videos = env.env.get_video()
            # if len(videos.shape) == 5:
            #     videos = videos[:, 0]  # select first frame
            
            # if save_video:
            #     videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            #     log_data[f'sim_video_eval'] = videos_wandb
            video_filename = f"metaworld_bsk_ep{episode_idx}.mp4"
            local_video_path = os.path.join(
                "/home/lipeng/Downloads/Lipeng_human_demonstration_icra/3D-Diffusion-Policy-master-MGT/3D-Diffusion-Policy/data/outputs/video",
                video_filename
            )

            videos = videos.transpose(0, 2, 3, 1)  # Uncomment if channels-first
            imageio.mimwrite(uri=local_video_path, ims=videos, fps=self.fps)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        # videos = env.env.get_video()
        # # if len(videos.shape) == 5:
        # #     videos = videos[:, 0]  # select first frame
        
        # # if save_video:
        # #     videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        # #     log_data[f'sim_video_eval'] = videos_wandb
        # local_video_path = os.path.join("/home/lipeng/Downloads/Lipeng_human_demonstration_icra/3D-Diffusion-Policy-master-MGT/3D-Diffusion-Policy/data/outputs/video/metaworld_bsk7.mp4")
        # videos = videos.transpose(0, 2, 3, 1)  # Uncomment if channels-first
        # imageio.mimwrite(uri=local_video_path, ims=videos, fps=self.fps)
        _ = env.reset()
        videos = None

        return log_data

    def run_test(self, policy: BasePolicy,first_obs=None):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            is_start = False
            batch_size = 1
            dummy_m_tokens = torch.zeros((batch_size, 50), dtype=torch.long, device=device)
            np_obs_dict = dict(obs)
            obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))
            obs = first_obs
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    #  need check !!!!!!!
                    
                    # seq_length = self.n_action_steps  # Adjust based on policy's expected sequence length
                    
                    if is_start == False:
                        dummy_action = torch.zeros((batch_size, 4, 4), dtype=dtype, device=device)
                        dummy_m_tokens[:, 0] = policy.vq_model.encode(dummy_action)
                        # print(f'dummy_m_tokens 0',dummy_m_tokens[:, 0]) # dummy_m_tokens (1, 50)
                        # dummy_m_tokens[:, 0] = 0  # EOS token
                    dummy_m_tokens_len = torch.tensor([50] * batch_size, device=device)                   
                    pc = obs_dict['point_cloud'].unsqueeze(0)  # Add batch dimension
                    state = obs_dict['agent_pos'].unsqueeze(0)  # Add batch dimension
                    
                    batch = {
                        'm_tokens': dummy_m_tokens,
                        # 'pc': pc,
                        # 'state': state,
                        'm_tokens_len': dummy_m_tokens_len
                    } 
                    batch['obs'] = {
                        'point_cloud': pc,
                        'agent_pos': state
                    }
                    # obs_dict_input = {}
                    # obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    # obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                   
                    action_dict = policy.predict_MGT_action(batch)
                    is_start = True

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action_pred'].squeeze(0)
                # print(f'action',action.shape) action (12, 4)
                # print(f'action', action) # action (12, 4)
                action_his = action[-4:, ...]
                action_his_tensor = torch.from_numpy(action_his).to(device=device, dtype=dtype)
                token_his = policy.vq_encode(action_his_tensor.unsqueeze(0))
                # print(f'token_his',token_his)
                # print(f'token_his',token_his.shape) # token_his (1, 1)
                token_his = token_his[..., 0:1]
                action_exec = action[4:, ...]
                obs, reward, done, info = env.step(action_exec)
                dummy_m_tokens[:, 0] = token_his

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])
             
            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            videos = env.env.get_video()
            # if len(videos.shape) == 5:
            #     videos = videos[:, 0]  # select first frame
            
            # if save_video:
            #     videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            #     log_data[f'sim_video_eval'] = videos_wandb
            video_filename = f"metaworld_bsk_ep{episode_idx}.mp4"
            local_video_path = os.path.join(
                "/home/lipeng/Downloads/Lipeng_human_demonstration_icra/3D-Diffusion-Policy-master-MGT/3D-Diffusion-Policy/data/outputs/video",
                video_filename
            )

            videos = videos.transpose(0, 2, 3, 1)  # Uncomment if channels-first
            imageio.mimwrite(uri=local_video_path, ims=videos, fps=self.fps)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        # videos = env.env.get_video()
        # # if len(videos.shape) == 5:
        # #     videos = videos[:, 0]  # select first frame
        
        # # if save_video:
        # #     videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        # #     log_data[f'sim_video_eval'] = videos_wandb
        # local_video_path = os.path.join("/home/lipeng/Downloads/Lipeng_human_demonstration_icra/3D-Diffusion-Policy-master-MGT/3D-Diffusion-Policy/data/outputs/video/metaworld_bsk7.mp4")
        # videos = videos.transpose(0, 2, 3, 1)  # Uncomment if channels-first
        # imageio.mimwrite(uri=local_video_path, ims=videos, fps=self.fps)
        _ = env.reset()
        videos = None

        return log_data
