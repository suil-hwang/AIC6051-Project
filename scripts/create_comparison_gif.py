"""Create comparison GIF showing Vanilla A*, Neural A*, and Field-based Neural A*
Author: Modified for comparison
Affiliation: OSX
"""

import os
from typing import Dict, Any

import hydra
import moviepy.editor as mpy
import numpy as np
import torch
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader, visualize_results
from neural_astar.utils.training import load_from_ptl_checkpoint


def load_model(model_type: str, config: Any, dataname: str):
    """Load a specific model based on type
    
    Args:
        model_type: "vanilla", "neural", or "field"
        config: Hydra config
        dataname: Dataset name
        
    Returns:
        Loaded planner model
    """
    if model_type == "vanilla":
        return VanillaAstar()
    
    elif model_type == "neural":
        planner = NeuralAstar(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=False,
            Tmax=1.0,
        )
        checkpoint_path = f"{config.modeldir}/neural_{dataname}"
        if os.path.exists(checkpoint_path):
            planner.load_state_dict(load_from_ptl_checkpoint(checkpoint_path))
        else:
            print(f"Warning: Checkpoint not found for neural model: {checkpoint_path}")
        return planner
    
    elif model_type == "field":
        try:
            from neural_astar.planner.astar import NeuralAstarField
            
            planner = NeuralAstarField(
                encoder_input=config.encoder.input,
                encoder_arch=config.encoder.arch,
                encoder_depth=config.encoder.depth,
                learn_obstacles=False,
                Tmax=1.0,
                use_geodesic_loss=True,
                use_obstacle_loss=True,
            )
            checkpoint_path = f"{config.modeldir}/field_{dataname}"
            if os.path.exists(checkpoint_path):
                planner.load_state_dict(load_from_ptl_checkpoint(checkpoint_path))
            else:
                print(f"Warning: Checkpoint not found for field model: {checkpoint_path}")
            return planner
        except ImportError:
            print("Error: NeuralAstarField not found. Skipping field model.")
            return None
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def extract_astar_output(outputs):
    """Handle models that return either AstarOutput or a tuple including it."""
    return outputs[0] if isinstance(outputs, tuple) else outputs


def create_side_by_side_frame(frames_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Create a side-by-side comparison frame
    
    Args:
        frames_dict: Dictionary with keys as model names and values as frames
        
    Returns:
        Combined frame with all models side-by-side
    """
    # Add text labels to each frame
    labeled_frames = []
    for model_name, frame in frames_dict.items():
        # Add a label bar at the top (black background with white text)
        label_height = 30
        h, w = frame.shape[:2]
        label_bar = np.zeros((label_height, w, 3), dtype=np.uint8)
        
        # Simple text rendering (just use the frame as-is with title in filename)
        # For production, you might want to use PIL or cv2 for text rendering
        labeled_frame = np.vstack([label_bar, frame])
        labeled_frames.append(labeled_frame)
    
    # Concatenate horizontally
    return np.hstack(labeled_frames)


@hydra.main(config_path="config", config_name="create_gif", version_base="1.2")
def main(config):
    dataname = os.path.basename(config.dataset)
    problem_id = config.get("problem_id", 1)
    
    # Models to compare
    model_types = ["vanilla", "neural", "field"]
    
    # Load test data
    dataloader = create_dataloader(
        config.dataset + ".npz",
        "test",
        100,
        shuffle=False,
        num_starts=1,
    )
    map_designs, start_maps, goal_maps, opt_trajs = next(iter(dataloader))
    
    # Get single problem
    map_design = map_designs[problem_id : problem_id + 1]
    start_map = start_maps[problem_id : problem_id + 1]
    goal_map = goal_maps[problem_id : problem_id + 1]
    
    print(f"\n{'='*60}")
    print(f"Creating Comparison GIF for Problem {problem_id}")
    print(f"{'='*60}\n")
    
    # Generate outputs for each model
    all_outputs = {}
    for model_type in model_types:
        print(f"Processing {model_type.upper()} model...")
        
        planner = load_model(model_type, config, dataname)
        if planner is None:
            continue
            
        outputs = planner(
            map_design,
            start_map,
            goal_map,
            store_intermediate_results=True,
        )

        astar_output = extract_astar_output(outputs)
        all_outputs[model_type] = astar_output
        print(f"  - {model_type}: {len(astar_output.intermediate_results)} frames generated")
    
    if not all_outputs:
        print("Error: No models were successfully loaded!")
        return
    
    # Create individual GIFs for each model
    savedir = f"{config.resultdir}/comparison"
    os.makedirs(savedir, exist_ok=True)
    
    for model_type, outputs in all_outputs.items():
        frames = [
            visualize_results(map_design, intermediate_results, scale=4)
            for intermediate_results in outputs.intermediate_results
        ]
        
        # Add pause at the end
        frames_with_pause = frames + [frames[-1]] * 15
        
        clip = mpy.ImageSequenceClip(frames_with_pause, fps=30)
        gif_path = f"{savedir}/{model_type}_{dataname}_{problem_id:04d}.gif"
        clip.write_gif(gif_path, verbose=False, logger=None)
        print(f"  ✓ Saved: {gif_path}")
    
    # Create side-by-side comparison GIF
    print("\nCreating side-by-side comparison GIF...")
    
    # Find the maximum number of frames
    max_frames = max(len(outputs.intermediate_results) for outputs in all_outputs.values())
    
    combined_frames = []
    for frame_idx in range(max_frames):
        frame_dict = {}
        for model_type, outputs in all_outputs.items():
            # Use last frame if this model has fewer frames
            idx = min(frame_idx, len(outputs.intermediate_results) - 1)
            frame = visualize_results(
                map_design, 
                outputs.intermediate_results[idx], 
                scale=4
            )
            frame_dict[model_type.upper()] = frame
        
        combined_frame = create_side_by_side_frame(frame_dict)
        combined_frames.append(combined_frame)
    
    # Add pause at the end
    combined_frames_with_pause = combined_frames + [combined_frames[-1]] * 15
    
    combined_clip = mpy.ImageSequenceClip(combined_frames_with_pause, fps=30)
    comparison_path = f"{savedir}/comparison_{dataname}_{problem_id:04d}.gif"
    combined_clip.write_gif(comparison_path, verbose=False, logger=None)
    
    print(f"\n{'='*60}")
    print(f"✓ Comparison GIF saved: {comparison_path}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("\nSummary:")
    print(f"  - Individual GIFs: {len(all_outputs)} files")
    print(f"  - Comparison GIF: 1 file")
    print(f"  - Output directory: {savedir}")
    print(f"\nModels compared: {', '.join(all_outputs.keys())}")


if __name__ == "__main__":
    main()
