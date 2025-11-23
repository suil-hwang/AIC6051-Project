"""Training Neural A* with Distance Fields
"""
from __future__ import annotations

import os
import hydra
import pytorch_lightning as pl
import torch
from neural_astar.planner import NeuralAstar
from neural_astar.planner.astar import NeuralAstarField
from neural_astar.utils.data import create_dataloader_with_fields
from neural_astar.utils.training import (
    PlannerModule, 
    PlannerModuleWithFields,
    set_global_seeds
)
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_path="config", config_name="train_with_fields")
def main(config):
    set_global_seeds(config.seed)
    
    # Dataloaders with distance fields
    train_loader = create_dataloader_with_fields(
        config.dataset + ".npz",
        "train",
        config.params.batch_size,
        shuffle=True,
        precompute_fields=True,
        cache_dir=config.get("cache_dir", "cache/distance_fields"),
    )
    
    val_loader = create_dataloader_with_fields(
        config.dataset + ".npz",
        "valid",
        config.params.batch_size,
        shuffle=False,
        precompute_fields=True,
        cache_dir=config.get("cache_dir", "cache/distance_fields"),
    )
    
    # Model selection
    model_type = config.get("model_type", "field")  # "vanilla", "neural", "field"
    
    if model_type == "field":
        planner = NeuralAstarField(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=False,
            Tmax=config.Tmax,
            use_geodesic_loss=config.get("use_geodesic_loss", True),
            use_obstacle_loss=config.get("use_obstacle_loss", True),
            geodesic_loss_weight=config.get("geodesic_loss_weight", 0.5),
            obstacle_loss_weight=config.get("obstacle_loss_weight", 0.3),
        )
        module = PlannerModuleWithFields(planner, config)
    else:
        planner = NeuralAstar(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=False,
            Tmax=config.Tmax,
        )
        module = PlannerModule(planner, config)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/h_mean" if model_type != "field" else "metrics/val_loss",
        save_weights_only=True,
        mode="max" if model_type != "field" else "min",
    )
    
    logdir = f"{config.logdir}/{model_type}_{os.path.basename(config.dataset)}"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()