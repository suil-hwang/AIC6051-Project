"""Training Neural A* variants (Vanilla, Neural, Field-based)
Author: Ryo Yonetani (modified)
Affiliation: OSX
"""
from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader, create_dataloader_with_fields
from neural_astar.utils.training import (
    PlannerModule, 
    PlannerModuleWithFields,
    set_global_seeds
)
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(config):
    torch.set_float32_matmul_precision('medium')
    set_global_seeds(config.seed)
    
    # Model type selection
    model_type = config.get("model_type", "neural")  # "vanilla", "neural", "field"
    
    print(f"\n{'='*60}")
    print(f"Training Model: {model_type.upper()}")
    print(f"{'='*60}\n")
    
    # ==========================================
    # 1. VANILLA A* (No training needed)
    # ==========================================
    if model_type == "vanilla":
        print("Vanilla A* does not require training.")
        print("Performing validation-only run to establish baseline metrics...")
        
        # Create dataloaders
        val_loader = create_dataloader(
            config.dataset + ".npz", "valid", config.params.batch_size, shuffle=False
        )
        
        # Create vanilla planner
        vanilla_astar = VanillaAstar(
            g_ratio=config.get("g_ratio", 0.5),
            use_differentiable_astar=False  # Use faster PQ-based A*
        )
        
        # Evaluation only
        module = PlannerModule(vanilla_astar, config)
        logdir = f"{config.logdir}/vanilla_{os.path.basename(config.dataset)}"
        
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            log_every_n_steps=1,
            default_root_dir=logdir,
            max_epochs=1,  # Just one epoch for validation
        )
        
        # Run validation only
        trainer.validate(module, val_loader)
        
        print("\nVanilla A* baseline established.")
        print(f"Results saved to: {logdir}")
        return
    
    # ==========================================
    # 2. NEURAL A* (Original)
    # ==========================================
    elif model_type == "neural":
        print("Training Neural A* (Original)...")
        
        # Create standard dataloaders
        train_loader = create_dataloader(
            config.dataset + ".npz", "train", config.params.batch_size, shuffle=True
        )
        val_loader = create_dataloader(
            config.dataset + ".npz", "valid", config.params.batch_size, shuffle=False
        )
        
        # Create Neural A* planner
        neural_astar = NeuralAstar(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=config.encoder.get("learn_obstacles", False),
            Tmax=config.Tmax,
        )
        
        # Training module
        module = PlannerModule(neural_astar, config)
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="metrics/h_mean",
            save_weights_only=True,
            mode="max",
            filename="best-{epoch:02d}-{metrics/h_mean:.4f}",
        )
        
        logdir = f"{config.logdir}/neural_{os.path.basename(config.dataset)}"
        
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            log_every_n_steps=1,
            default_root_dir=logdir,
            max_epochs=config.params.num_epochs,
            callbacks=[checkpoint_callback],
        )
        
        trainer.fit(module, train_loader, val_loader)
        
        print(f"\nNeural A* training completed.")
        print(f"Best model saved to: {logdir}")
    
    # ==========================================
    # 3. NEURAL A* WITH DISTANCE FIELDS (Ours)
    # ==========================================
    elif model_type == "field":
        print("Training Neural A* with Distance Fields (Ours)...")
        
        # Import NeuralAstarField
        try:
            from neural_astar.planner.astar import NeuralAstarField
        except ImportError:
            print("ERROR: NeuralAstarField not found!")
            print("Please ensure you have implemented NeuralAstarField in src/neural_astar/planner/astar.py")
            return
        
        # Create dataloaders with distance fields
        cache_dir = config.get("cache_dir", "cache/distance_fields")
        
        print(f"\nPrecomputing distance fields (this may take a while)...")
        print(f"Cache directory: {cache_dir}")
        
        train_loader = create_dataloader_with_fields(
            config.dataset + ".npz",
            "train",
            config.params.batch_size,
            shuffle=True,
            precompute_fields=True,
            cache_dir=cache_dir,
        )
        
        val_loader = create_dataloader_with_fields(
            config.dataset + ".npz",
            "valid",
            config.params.batch_size,
            shuffle=False,
            precompute_fields=True,
            cache_dir=cache_dir,
        )
        
        print("Distance fields ready!\n")
        
        # Create NeuralAstarField planner
        field_planner = NeuralAstarField(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=config.encoder.get("learn_obstacles", False),
            Tmax=config.Tmax,
            use_geodesic_loss=config.get("use_geodesic_loss", True),
            use_obstacle_loss=config.get("use_obstacle_loss", True),
            geodesic_loss_weight=config.get("geodesic_loss_weight", 0.5),
            obstacle_loss_weight=config.get("obstacle_loss_weight", 0.3),
        )
        
        # Training module with distance field losses
        module = PlannerModuleWithFields(field_planner, config)
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="metrics/h_mean",
            save_weights_only=True,
            mode="max",
            filename="best-{epoch:02d}-{metrics/h_mean:.4f}",
        )
        
        logdir = f"{config.logdir}/field_{os.path.basename(config.dataset)}"
        
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            log_every_n_steps=1,
            default_root_dir=logdir,
            max_epochs=config.params.num_epochs,
            callbacks=[checkpoint_callback],
        )
        
        trainer.fit(module, train_loader, val_loader)
        
        print(f"\nNeural A* with Distance Fields training completed.")
        print(f"Best model saved to: {logdir}")
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'vanilla', 'neural', or 'field'")


if __name__ == "__main__":
    main()

# python scripts/train.py --multirun model_type=vanilla,neural,field
