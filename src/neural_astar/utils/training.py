"""Helper functions for training
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

import random
import re
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
from neural_astar.planner.astar import VanillaAstar


def load_from_ptl_checkpoint(checkpoint_path: str) -> dict:
    """
    Load model weights from PyTorch Lightning checkpoint.

    Args:
        checkpoint_path (str): (parent) directory where .ckpt is stored.

    Returns:
        dict: model state dict
    """

    ckpt_file = sorted(glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))[-1]
    print(f"load {ckpt_file}")
    state_dict = torch.load(ckpt_file, weights_only=True)["state_dict"]
    state_dict_extracted = dict()
    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted


class PlannerModule(pl.LightningModule):
    def __init__(self, planner, config):
        super().__init__()
        self.planner = planner
        self.vanilla_astar = VanillaAstar()
        self.config = config

    def forward(self, map_designs, start_maps, goal_maps):
        return self.planner(map_designs, start_maps, goal_maps)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(self.planner.parameters(), self.config.params.lr)

    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.L1Loss()(outputs.histories, opt_trajs)
        self.log("metrics/train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.L1Loss()(outputs.histories.to(opt_trajs.device), opt_trajs)

        self.log("metrics/val_loss", loss)

        # For shortest path problems:
        if map_designs.shape[1] == 1:
            va_outputs = self.vanilla_astar(map_designs, start_maps, goal_maps)
            pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            p_opt = (pathlen_astar == pathlen_model).mean()

            exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            exp_na = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()

            h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))

            self.log("metrics/p_opt", p_opt)
            self.log("metrics/p_exp", p_exp)
            self.log("metrics/h_mean", h_mean)

        return loss


class PlannerModuleWithFields(PlannerModule):
    """Distance field loss를 포함하는 training module (개선 버전)"""
    
    def __init__(self, planner, config):
        super().__init__(planner, config)
        
        # Loss weights from config
        self.geodesic_loss_weight = getattr(config, 'geodesic_loss_weight', 0.5)
        self.obstacle_loss_weight = getattr(config, 'obstacle_loss_weight', 0.3)
        self.smooth_l1_beta = getattr(config, 'smooth_l1_beta', 1.0)
        self.use_log_space = getattr(config, 'use_log_space', True)
        
        # Adaptive loss balancing
        self.adaptive_weights = getattr(config, 'adaptive_weights', False)
        if self.adaptive_weights:
            # Track running statistics for loss balancing
            self.register_buffer('loss_stats', torch.zeros(3))  # [path, geo, obs]
            self.register_buffer('loss_count', torch.zeros(1))
    
    def compute_distance_field_losses(
        self,
        geodesic_pred: Optional[torch.Tensor],
        obstacle_pred: Optional[torch.Tensor],
        geodesic_gt: Optional[torch.Tensor],
        obstacle_gt: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distance field losses with log-space Smooth L1
        
        Args:
            geodesic_pred, obstacle_pred: [B, 1, H, W] predictions (already normalized)
            geodesic_gt, obstacle_gt: [B, 1, H, W] ground truth (already normalized)
        """
        device = geodesic_pred.device if geodesic_pred is not None else 'cpu'
        
        geodesic_loss = torch.tensor(0.0).to(device)
        obstacle_loss = torch.tensor(0.0).to(device)
        
        # Geodesic Distance Field Loss
        if geodesic_pred is not None and geodesic_gt is not None:
            if self.use_log_space:
                # Log-space: log(1 + x) for better handling of large distances
                pred_log = torch.log1p(geodesic_pred)  # log(1 + pred)
                gt_log = torch.log1p(geodesic_gt)      # log(1 + gt)
                geodesic_loss = nn.SmoothL1Loss(beta=self.smooth_l1_beta)(
                    pred_log, gt_log
                )
            else:
                # Direct space
                geodesic_loss = nn.SmoothL1Loss(beta=self.smooth_l1_beta)(
                    geodesic_pred, geodesic_gt
                )
        
        # Obstacle Distance Field Loss
        if obstacle_pred is not None and obstacle_gt is not None:
            if self.use_log_space:
                pred_log = torch.log1p(obstacle_pred)
                gt_log = torch.log1p(obstacle_gt)
                obstacle_loss = nn.SmoothL1Loss(beta=self.smooth_l1_beta)(
                    pred_log, gt_log
                )
            else:
                obstacle_loss = nn.SmoothL1Loss(beta=self.smooth_l1_beta)(
                    obstacle_pred, obstacle_gt
                )
        
        return geodesic_loss, obstacle_loss
    
    def update_adaptive_weights(
        self, 
        path_loss: torch.Tensor, 
        geo_loss: torch.Tensor, 
        obs_loss: torch.Tensor
    ):
        """Update loss statistics for adaptive weighting"""
        if not self.adaptive_weights:
            return
        
        with torch.no_grad():
            losses = torch.stack([
                path_loss.detach(), 
                geo_loss.detach(), 
                obs_loss.detach()
            ])
            
            # Exponential moving average
            alpha = 0.1
            self.loss_stats = (1 - alpha) * self.loss_stats + alpha * losses
            self.loss_count += 1
            
            # Adjust weights every N steps
            if self.loss_count % 100 == 0:
                # Normalize by path loss (main objective)
                if self.loss_stats[0] > 0:
                    ratio_geo = self.loss_stats[0] / (self.loss_stats[1] + 1e-8)
                    ratio_obs = self.loss_stats[0] / (self.loss_stats[2] + 1e-8)
                    
                    # Gentle adjustment
                    self.geodesic_loss_weight = 0.5 * ratio_geo
                    self.obstacle_loss_weight = 0.3 * ratio_obs
                    
                    # Clip to reasonable range
                    self.geodesic_loss_weight = np.clip(
                        self.geodesic_loss_weight, 0.1, 2.0
                    )
                    self.obstacle_loss_weight = np.clip(
                        self.obstacle_loss_weight, 0.1, 2.0
                    )
    
    def training_step(self, train_batch, batch_idx):
        (map_designs, start_maps, goal_maps, opt_trajs, 
         geodesic_gt, obstacle_gt) = train_batch
        
        # Forward pass
        outputs, geodesic_pred, obstacle_pred = self.planner(
            map_designs, start_maps, goal_maps
        )
        
        # Path loss (main objective)
        path_loss = nn.L1Loss()(outputs.histories, opt_trajs)
        
        # Distance field losses
        geodesic_loss, obstacle_loss = self.compute_distance_field_losses(
            geodesic_pred, obstacle_pred, geodesic_gt, obstacle_gt
        )
        
        # Update adaptive weights if enabled
        self.update_adaptive_weights(path_loss, geodesic_loss, obstacle_loss)
        
        # Total loss
        total_loss = (
            path_loss 
            + self.geodesic_loss_weight * geodesic_loss
            + self.obstacle_loss_weight * obstacle_loss
        )
        
        # Logging
        self.log("metrics/train_loss", total_loss)
        self.log("metrics/train_path_loss", path_loss)
        self.log("metrics/train_geodesic_loss", geodesic_loss)
        self.log("metrics/train_obstacle_loss", obstacle_loss)
        self.log("weights/lambda_geo", self.geodesic_loss_weight)
        self.log("weights/lambda_obs", self.obstacle_loss_weight)
        
        return total_loss
    
    def validation_step(self, val_batch, batch_idx):
        (map_designs, start_maps, goal_maps, opt_trajs,
         geodesic_gt, obstacle_gt) = val_batch
        
        outputs, geodesic_pred, obstacle_pred = self.planner(
            map_designs, start_maps, goal_maps
        )
        
        path_loss = nn.L1Loss()(outputs.histories, opt_trajs)
        geodesic_loss, obstacle_loss = self.compute_distance_field_losses(
            geodesic_pred, obstacle_pred, geodesic_gt, obstacle_gt
        )
        
        total_loss = (
            path_loss
            + self.geodesic_loss_weight * geodesic_loss
            + self.obstacle_loss_weight * obstacle_loss
        )
        
        self.log("metrics/val_loss", total_loss)
        self.log("metrics/val_path_loss", path_loss)
        self.log("metrics/val_geodesic_loss", geodesic_loss)
        self.log("metrics/val_obstacle_loss", obstacle_loss)
        
        # Additional metrics
        if geodesic_pred is not None:
            # Mean Absolute Error in original space
            geo_mae = torch.abs(geodesic_pred - geodesic_gt).mean()
            self.log("metrics/val_geo_mae", geo_mae)
        
        if obstacle_pred is not None:
            obs_mae = torch.abs(obstacle_pred - obstacle_gt).mean()
            self.log("metrics/val_obs_mae", obs_mae)
        
        # Optimality metrics (original Neural A*)
        if map_designs.shape[1] == 1:
            va_outputs = self.vanilla_astar(map_designs, start_maps, goal_maps)
            pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            p_opt = (pathlen_astar == pathlen_model).mean()
            
            exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            exp_na = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()
            
            h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))
            
            self.log("metrics/p_opt", p_opt)
            self.log("metrics/p_exp", p_exp)
            self.log("metrics/h_mean", h_mean)
        
        return total_loss


def set_global_seeds(seed: int) -> None:
    """
    Set random seeds

    Args:
        seed (int): random seed
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
