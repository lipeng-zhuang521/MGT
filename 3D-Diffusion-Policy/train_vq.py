if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from pathlib import Path

import torch.optim as optim
from diffusion_policy_3d.mgt_policy.mgt import MGT

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        # self.model: DP3 = hydra.utils.instantiate(cfg.policy)
        self.model = MGT(shape_meta=cfg.policy.shape_meta,
            # noise_scheduler: DDPMScheduler,
            horizon=cfg.policy.horizon, 
            n_action_steps=cfg.policy.horizon, 
            n_obs_steps=cfg.policy.horizon,
            encoder_output_dim=cfg.policy.encoder_output_dim,
            crop_shape=cfg.policy.crop_shape,
            use_pc_color=cfg.policy.use_pc_color,
            pointnet_type=cfg.policy.pointnet_type,
            pointcloud_encoder_cfg=cfg.policy.pointcloud_encoder_cfg)
        self.optimizer = optim.AdamW(self.model.vq_model.parameters(), lr=self.model.args_vq.lr, betas=(0.9, 0.99), weight_decay=self.model.args_vq.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.model.args_vq.lr_scheduler, gamma=self.model.args_vq.gamma)
        # configure training state
        self.global_step = 0
        self.epoch = 0

    @staticmethod
    def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        return optimizer, current_lr
    @staticmethod
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    # @staticmethod
    # def save_checkpoint(self, path=None,nb_iter=None):
    #     if path is None:
    #         checkpoint_dir = Path("checkpoints")
    #         checkpoint_dir.mkdir(exist_ok=True)
    #         path = checkpoint_dir / f"checkpoint_iter_{nb_iter}.pth"
        
    #     torch.save({
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #     }, path)
    #     print(f"Saved checkpoint to {path}")

    def run(self):
        cfg = copy.deepcopy(self.cfg)       
        RUN_VALIDATION = False # reduce time cost
        
        # resume training
        # if cfg.training.resume:
        #     lastest_ckpt_path = self.get_checkpoint_path()
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.vq_dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        train_dataloader_iter = self.cycle(train_dataloader)
        test_dataloader_iter = self.cycle(val_dataloader)
        self.model.set_normalizer(normalizer)

        # cprint("-----------------------------", "yellow")
        # cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        # cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        # cprint("-----------------------------", "yellow")
        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint

        # topk_manager = TopKCheckpointManager(
        #     save_dir=os.path.join(self.output_dir, 'checkpoints'),
        #     **cfg.checkpoint.topk
        # )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        vq_out_dir = os.path.join(self.model.args_vq.out_dir, f'vq')  # /{args.exp_name}
        
        print(f"Starting {self.model.args_vq.warm_up_iter} warmup iterations...")
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        
        for nb_iter in tqdm(range(1, self.model.args_vq.warm_up_iter + 1), desc="Warmup"):
            # Update learning rate
            self.optimizer, current_lr = self.update_lr_warm_up(self.optimizer, nb_iter, self.model.args_vq.warm_up_iter, self.model.args_vq.lr)

           # Get batch
            batch = next(train_dataloader_iter)
            batch = dict_apply(batch, lambda x: x.to(device))

            # Forward + Backward
            self.optimizer.zero_grad()
            total_loss, loss_dict = self.model.compute_vq_loss(batch)
            total_loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            avg_recons += loss_dict['loss_recon']
            avg_perplexity += loss_dict['perplexity']
            avg_commit += loss_dict['loss_commit']

            # Logging
            if nb_iter % self.model.args_vq.print_iter == 0:
                avg_recons /= self.model.args_vq.print_iter
                avg_perplexity /= self.model.args_vq.print_iter
                avg_commit /= self.model.args_vq.print_iter
                
                print(f"[Warmup] Iter {nb_iter}/{self.model.args_vq.warm_up_iter} | "
                    f"LR: {current_lr:.2e} | "
                    f"Recon: {avg_recons:.4f} | "
                    f"Commit: {avg_commit:.4f} | "
                    f"PPL: {avg_perplexity:.2f}")
                
                avg_recons, avg_perplexity, avg_commit = 0., 0., 0.


        # ========== Main Training Phase ==========
        print("Starting main training...")

        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        for nb_iter in tqdm(range(1, self.model.args_vq.total_iter + 1)):
            # step_log = dict()
            # ========= train for this epoch ==========
            batch = next(train_dataloader_iter)
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            self.optimizer.zero_grad()
            total_loss, loss_dict = self.model.compute_vq_loss(batch)
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()        
            
            if nb_iter % self.model.args_vq.print_iter == 0:
                avg_recons += loss_dict['loss_recon']
                avg_perplexity += loss_dict['perplexity']
                avg_commit += loss_dict['loss_commit']

                print(
                    f"Train. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
            
                # step_log = {
                #         'lr': current_lr,
                #         'train_loss': total_loss.item(),
                #         'loss_recon': loss_dict['loss_recon'],
                #         'loss_commit': loss_dict['loss_commit'],
                #         'perplexity': loss_dict['perplexity']
                #     }

                # step_log.update(loss_dict)
                # step_log['train_loss'] = total_loss
                avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
        
            # ========= eval for this epoch ==========
                policy = self.model
                policy.eval()
               
           # run validation     
                batch = next(test_dataloader_iter)
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                total_loss, loss_dict = self.model.compute_vq_loss(batch) 
                avg_recons += loss_dict['loss_recon']
                avg_perplexity += loss_dict['perplexity']
                avg_commit += loss_dict['loss_commit']
                print(
                    f"Test. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
                avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
                # step_log = {
                #         'lr': current_lr,
                #         'test_loss': total_loss.item(),
                #         'test_loss_recon': loss_dict['loss_recon'],
                #         'test_loss_commit': loss_dict['loss_commit'],
                #         'test_perplexity': loss_dict['perplexity']
                #     }
                    
                # ========= eval end for this epoch ==========

                # end of epoch
                # log of last step is combined with validation and rollout
                # wandb_run.log(step_log, step=self.global_step)
                # del step_log
        
            if nb_iter % self.model.args_vq.save_iter == 0:
                torch.save({'net': policy.state_dict()}, os.path.join(vq_out_dir, f'{nb_iter}_net_last.pth'))
   

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    print("Start training...")
    workspace = TrainDP3Workspace(cfg)
    workspace.run()
    # vq_model = workspace.model.vq_model
    # vq_model.eval()
    # vq_model.cuda()
    
    # args_vq = workspace.model.args_vq
    # codebook_dir = os.path.join(args_vq.out_dir, 'codebook')
    # os.makedirs(codebook_dir, exist_ok=True)
    # print(f'Generating codebook in {codebook_dir}')

    # if len(os.listdir(codebook_dir)) == 0:
    #     # Use DP3's existing dataset configuration
    #     dataset = hydra.utils.instantiate(cfg.task.vq_dataset)
    #     train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    #     # Get normalizer from workspace
    #     # normalizer = workspace.model.normalizer
    #     val_dataset = dataset.get_validation_dataset()
    #     val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    #     dataloaders = [
    #         ('train', train_dataloader),
    #         ('val', val_dataloader)
    #     ]
    #     for dataloader_name, dataloader in dataloaders:
    #         for i, batch in tqdm(enumerate(dataloader), 
    #                         desc=f"Generating {dataloader_name} codebook"):
    #             # Move data to GPU and normalize
    #             batch = dict_apply(batch, lambda x: x.to(device='cuda'))
    #             action = batch['action'].float()
                
    #             with torch.no_grad():
    #                 target = vq_model.encode(action)
                
    #             # Save with dataset prefix to avoid name collisions
    #             target = target.cpu().numpy()
    #             fname = os.path.join(codebook_dir, f'{dataloader_name}_{i}.npy')
    #             np.save(fname, target)


        

if __name__ == "__main__":
    main()
