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
from diffusion_policy_3d.model.mgt_utils.utils_model import initial_optim

import torch.optim as optim
from diffusion_policy_3d.mgt_policy.mgt import MGT
from pathlib import Path

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
        
        self.optimizer = initial_optim(self.model.args_trans.decay_option, self.model.args_trans.lr, self.model.args_trans.weight_decay, self.model.trans_model, self.model.args_trans.optimizer)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.model.args_trans.lr_scheduler, gamma=self.model.args_trans.gamma)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    @staticmethod
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        RUN_VALIDATION = False # reduce time cost     
        # # resume training
        # if cfg.training.resume:
        #     lastest_ckpt_path = self.get_checkpoint_path()
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.trans_dataset)
        # dataset = hydra.utils.instantiate(
        #     cfg.task.trans_dataset, 
        #     phase='train'  # Explicit training phase
        # )
        # val_dataset = hydra.utils.instantiate(
        #     cfg.task.trans_dataset,
        #     phase='val'    # Explicit validation phase
        # )
        # assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        train_dataloader_iter = self.cycle(train_dataloader)
        test_dataloader_iter = self.cycle(val_dataloader)
        # configure validation dataset


        self.model.set_normalizer(normalizer)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
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
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # training loop
        for nb_iter in tqdm(range(1, self.model.args_trans.total_iter + 1), position=0, leave=True):
            step_log = dict()
            # ========= train for this epoch ==========
            batch = next(train_dataloader_iter)
            loss_cls, loss_dict = self.model.compute_trans_loss(batch)
            self.optimizer.zero_grad()
            loss_cls.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            if nb_iter % self.model.args_trans.print_iter == 0:
                print(f'Iter {nb_iter} : Loss. {loss_cls:.5f}, ACC. {loss_dict["acc_overall"]:.4f}',
                    f'ACC_masked. {loss_dict["acc_masked"]:.4f}', f'ACC_no_masked. {loss_dict["acc_no_masked"]:.4f}')
            
            # ========= eval for this epoch ==========
            # policy = self.model
            # policy.eval()          
          # run validation
            if nb_iter % self.model.args_trans.eval_rand_iter == 0:
                policy = self.model
                policy.eval()
                test_loss_total = 0.0
                test_acc_total =  0.0
                test_mask_acc_total =  0.0
                test_no_mask_acc_total =  0.0
                num_batches = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        loss_cls, loss_dict = self.model.compute_trans_loss(val_batch)
                
                        test_loss_total += loss_dict['loss_recon']
                        test_acc_total += loss_dict['acc_overall']
                        test_mask_acc_total += loss_dict['acc_masked']
                        test_no_mask_acc_total += loss_dict['acc_no_masked']
                        num_batches +=1
                
                test_mask_acc_mean = test_mask_acc_total/ num_batches
                test_no_mask_acc_mean = test_no_mask_acc_total/ num_batches
                test_loss_mean = test_loss_total/ num_batches
                test_acc_mean = test_acc_total/ num_batches
                
                print(f'Iter {nb_iter} : Val_Rand_Loss. {test_loss_mean:.5f}, Val_Rand_ACC. {test_acc_mean:.4f}',
              f'ACC_masked. {test_mask_acc_mean:.4f}', f'ACC_no_masked. {test_no_mask_acc_mean:.4f}')
                
                # sample trajectory from training set, and evaluate difference
                # batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                # obs_dict = batch['obs']
                # gt_action = batch['action']
                batch = next(train_dataloader_iter) 
                result = policy.predict_MGT_action(batch)
                gt_action = result['action_gt']
                # print(f'Iter {nb_iter} : Val_Rand_GT_Action. {gt_action}')
                # print(gt_action.shape) 50 12 4
                pred_action = result['action_pred']
                # print(f'Iter {nb_iter} : Val_Rand_Pred_Action. {pred_action}')
                # print(pred_action.shape) 50 12 4
                # print('pred_action', pred_action[0])
                # print('gt_action', gt_action[0])
                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                step_log['train_action_mse_error'] = mse.item()
                print(f'Iter {nb_iter} : train_action_mse_error. {step_log["train_action_mse_error"]:.5f}')
                del batch
                # del obs_dict
                del gt_action
                del result
                del pred_action
                del mse

                policy.train()
            # checkpoint
            if nb_iter % self.model.args_trans.save_iter == 0:
                # checkpointing
                
                self.save_checkpoint(path=None,nb_iter=nb_iter)
                
                # # sanitize metric names
                # metric_dict = dict()
                # for key, value in step_log.items():
                #     new_key = key.replace('/', '_')
                #     metric_dict[new_key] = value
                
                # # We can't copy the last checkpoint here
                # # since save_checkpoint uses threads.
                # # therefore at this point the file might have been empty!
                # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                # if topk_ckpt_path is not None:
                #     self.save_checkpoint(path=topk_ckpt_path)
     



    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        checkpoint_path = "checkpoints/checkpoint_iter_40000.pth"  # or your specific path
        
        if Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        else:
            print("No checkpoint found, starting from scratch")
        # lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        # if lastest_ckpt_path.is_file():
        #     cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
        #     self.load_checkpoint(path=lastest_ckpt_path)
        
        cfg = copy.deepcopy(self.cfg)
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.trans_dataset)
        first_sample = dataset[0]
        first_obs = {
            'point_cloud': first_sample['obs']['point_cloud'][0:5],
            'agent_pos': first_sample['obs']['agent_pos'][0:5]
        }
        # configure env
        env_runner_MGT: BaseRunner
        env_runner_MGT = hydra.utils.instantiate(
            cfg.task.env_runner_MGT,
            output_dir=self.output_dir)
        assert isinstance(env_runner_MGT, BaseRunner)
        policy = self.model
        policy.eval()
        policy.cuda()

        # runner_log = env_runner_MGT.run(policy)
        runner_log = env_runner_MGT.run_test(policy,first_obs)
        
        
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir


    def save_original_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def save_checkpoint(self, path=None,nb_iter=None):
        if path is None:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            path = checkpoint_dir / f"checkpoint_iter_{nb_iter}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")

    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_original_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload

    def load_checkpoint(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint {path} not found")
        
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path})")
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
