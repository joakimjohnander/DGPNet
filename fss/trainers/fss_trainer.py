import glob
import time
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from fss.utils.stats import AverageMeter
from fss.utils.debugging import print_tensor_statistics, revert_imagenet_normalization, COLOR_RED, COLOR_WHITE
from fss.utils.recursive_functions import recursive_detach, recursive_to


class FSSTrainer:
    def __init__(self, model, optimizer, lr_sched, train_loader, val_loaders,
                 checkpoint_path, visualization_path, save_name, device,
                 checkpoint_epochs, print_interval, visualization_epochs,
                 gradient_clip_value=None, print_param_on_training_end=False):
        self._model = model
        self._optimizer = optimizer
        self._lr_sched = lr_sched
        
        self._train_loader = train_loader
        self._val_loaders = val_loaders
        if isinstance(self._val_loaders, list):
            self._val_loaders = {f'val{idx}': ldr for idx, ldr in enumerate(self._val_loaders)}

        self._gradient_clip_value = gradient_clip_value
        assert gradient_clip_value is None or isinstance(gradient_clip_value, (int, float))

        self._checkpoint_path = checkpoint_path
        self._visualization_path = visualization_path
        self._save_name = save_name
        self._device = device

        self._checkpoint_epochs = checkpoint_epochs
        self._print_interval = print_interval
        self._visualization_epochs = visualization_epochs
        self._print_param_on_training_end = print_param_on_training_end
        
        # Initialize statistics variables @todo should we also add some KPI s.a. mIoU?
        self._stats = {}
        modes = ['train'] + list(self._val_loaders.keys())
        for mode in modes:
            for loss_key in model.get_loss_idfs():
                self._stats[f'{mode} {loss_key} loss'] = AverageMeter()

        self._epoch = 0

    def train(self, max_epochs):
        print(f"Training epochs {self._epoch + 1} to {max_epochs}. Moving model to {self._device}.")
        self._model.to(self._device)
        for epoch in range(self._epoch + 1, max_epochs + 1):
            self._epoch = epoch
            print(f"Starting epoch {epoch} with lr={self._lr_sched.get_last_lr()}")
            self._train_epoch()
            if self._epoch in self._checkpoint_epochs:
                print("Saving Checkpoint, current statistics are:")
                for key, val in self._stats.items():
                    strout = ["{:.3f}".format(elem) for elem in val.history]
                    print(key, strout, flush=True)
                self.save_checkpoint()
            self._lr_sched.step()
        print(f"Finished training!")
        if self._print_param_on_training_end:
            print("Final parameter statistics are:")
            for name, param in self._model.named_parameters():
                print_tensor_statistics(param, name)

    def _train_epoch(self):
        """Do one epoch of training and validation."""
        self._model.train(True)
        self._run_epoch(mode='train', data_loader=self._train_loader)

        self._model.train(False)
        with torch.no_grad():
            for loader_name, data_loader in self._val_loaders.items():
                self._run_epoch(mode=f'{loader_name}', data_loader=data_loader)

        # Update all stat values
        for stat_value in self._stats.values():
            if isinstance(stat_value, AverageMeter):
                stat_value.new_epoch()
                
    def _run_epoch(self, mode, data_loader):
        for i, data in enumerate(data_loader):
            data = recursive_to(data, self._device)
            if hasattr(self._model, 'supervisor'):
                data = self._model.supervisor.augment_data(data, mode)

            self._optimizer.zero_grad()

            # For debugging purposes we feed in annotations
            visualize_this_iteration = (i == 0 and self._epoch in self._visualization_epochs)
            model_output, state, losses = self._model(data, visualize=visualize_this_iteration, epoch=self._epoch)
            loss, partial_losses = losses['total'], losses['partial']
            
            if mode == 'train':
                loss.backward()
                self._optimizer.step()

            partial_losses = {key: {'val': loss['val'].detach().to('cpu').item(), 'N': loss['N']}
                              for key, loss in partial_losses.items()}
            self.save_stats(partial_losses, model_output, data, mode)

            if visualize_this_iteration:
                print(f"Visualization at split {mode}, epoch {self._epoch}, and in-epoch-iteration {i}.")
                self.visualize_batch(data, model_output, mode)

            del model_output, state

            if (i + 1) % self._print_interval == 0:
                loss_str = [(self._stats[f'{mode} {key} loss'].avg, key) for key in partial_losses.keys()]
                loss_str = [f"{val:.5f} ({name})" for val, name in loss_str]
                loss_str = "  ".join(loss_str)
                gpu_mem = torch.cuda.max_memory_allocated()//1000**2
                now = datetime.now()
                print(f"{now} [{mode}: {self._epoch}, {i+1:4d}] Loss: {loss_str}, GPU memory:{gpu_mem} MB")
                torch.cuda.reset_peak_memory_stats()

        # end for
        loss_str = [(self._stats[f'{mode} {key} loss'].avg, key) for key in partial_losses.keys()]
        loss_str = [f"{val:.5f} ({name})" for val, name in loss_str]
        loss_str = "  ".join(loss_str)
        print(f"[{mode}: {self._epoch}] Loss: {loss_str}")
        return

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""
        state = {
            'epoch': self._epoch,
            'net_type': type(self._model).__name__,
            'net': self._model.state_dict(),
            'optimizer' : self._optimizer.state_dict(),
            'stats' : self._stats,
            'device' : self._device,
        }
        file_path = '{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_path, self._save_name, self._epoch)
        torch.save(state, file_path)

    def load_checkpoint(self, checkpoint=None):
        """Loads a network checkpoint file.
        """
        if checkpoint is None: # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}_ep*.pth.tar'.format(self._checkpoint_path, self._save_name)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int): # Checkpoint is the epoch number            
            checkpoint_path = '{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_path, self._save_name, checkpoint)
        elif isinstance(checkpoint, str): # checkpoint is the epoch file path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError
        if not os.path.isfile(checkpoint_path):
            print(f"WARNING: Attempted to load checkpoint at epoch {checkpoint}, but it does not"
                  + " exist. Continuing without loading. If runfile is correctly set up, there will"
                  + " be an upcoming training stage that will begin from scratch.")
            return
        checkpoint_dict = torch.load(checkpoint_path)
        assert type(self._model).__name__ == checkpoint_dict['net_type'], 'Network is not of correct type'
        self._epoch = checkpoint_dict['epoch']
        self._model.load_state_dict(checkpoint_dict['net'])
        self._optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self._stats = checkpoint_dict['stats']
        self._device = checkpoint_dict['device']
        self._lr_sched.step(self._epoch)
        print("Loaded: {}".format(checkpoint_path))

    def save_stats(self, partial_losses, model_output, data, mode):
        for name, loss in partial_losses.items():
            self._stats[f'{mode} {name} loss'].update(loss['val'], loss['N'])

    def _get_visualization(self, images, segmentations):
        B, N, H, W = segmentations.size()
        background = (segmentations == 0).cpu().detach().float().view(B, N, 1, H, W)
        target = (segmentations == 1).cpu().detach().float().view(B, N, 1, H, W)
        ignore = (segmentations == 255).cpu().detach().float().view(B, N, 1, H, W)
        visualization = (background * images
                         + target * (0.5 * images + 0.5 * COLOR_RED)
                         + ignore * COLOR_WHITE)
        visualization = (visualization * 255).byte()
        return visualization
            
    def visualize_batch(self, data, model_output, mode):
        query_images = revert_imagenet_normalization(data['query_images'].cpu().detach())
        support_images = revert_imagenet_normalization(data['support_images'].cpu().detach())
        support_anno_vis = self._get_visualization(support_images, data['support_segmentations'])
        query_pred_vis = self._get_visualization(query_images, model_output['query_segmentations'])
        query_anno_vis = self._get_visualization(query_images, data['query_segmentations'])
        B, Q, _, H, W = query_images.size()
        _, S, _, _, _ = support_images.size()

        if not os.path.exists(os.path.join(self._visualization_path, self._save_name)):
            os.makedirs(os.path.join(self._visualization_path, self._save_name))
        for b in range(B):
            for s in range(S):
                fpath = os.path.join(self._visualization_path, self._save_name, f"{mode}_b{b}_n{s}_qsupp_anno.png")
                tv.io.write_png(support_anno_vis[b, s], fpath)
            for q in range(Q):
                fpath = os.path.join(self._visualization_path, self._save_name, f"{mode}_b{b}_n{q}_query_pred.png")
                tv.io.write_png(query_pred_vis[b, q], fpath)
                fpath = os.path.join(self._visualization_path, self._save_name, f"{mode}_b{b}_n{q}_query_anno.png")
                tv.io.write_png(query_anno_vis[b, q], fpath)
        
