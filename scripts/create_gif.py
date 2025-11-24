
import os
import warnings

# Suppress deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_lite")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import hydra
from moviepy import ImageSequenceClip
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader, visualize_results
from neural_astar.utils.training import load_from_ptl_checkpoint


def extract_astar_output(outputs):
    """Handle models that return either AstarOutput or a tuple including it."""
    from neural_astar.planner.differentiable_astar import AstarOutput
    # Check for AstarOutput FIRST before tuple, because AstarOutput is a NamedTuple
    # which is a subclass of tuple, so isinstance(outputs, tuple) would return True
    if isinstance(outputs, AstarOutput):
        return outputs
    elif isinstance(outputs, tuple):
        # This is from NeuralAstarField which returns (AstarOutput, geodesic_pred, obstacle_pred)
        return outputs[0]
    else:
        raise TypeError(f"Unexpected output type: {type(outputs)}")


def create_gif_for_planner(planner_type, config, dataloader, map_designs, start_maps, goal_maps):
    """Create GIF for a specific planner type."""
    dataname = os.path.basename(config.dataset)
    problem_id = config.problem_id
    
    print(f"\n{'='*60}")
    print(f"Processing {planner_type.upper()} planner...")
    print(f"{'='*60}")
    
    # Create planner based on type
    if planner_type == "na":
        # Neural A* (Original)
        planner = NeuralAstar(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=False,
            Tmax=1.0,
        )
        try:
            planner.load_state_dict(
                load_from_ptl_checkpoint(f"{config.modeldir}/neural_{dataname}")
            )
            print(f"✓ Loaded Neural A* model")
        except Exception as e:
            print(f"✗ Failed to load Neural A* model: {e}")
            return False
    
    elif planner_type == "field":
        # Neural A* with Distance Fields
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
            planner.load_state_dict(
                load_from_ptl_checkpoint(f"{config.modeldir}/field_{dataname}")
            )
            print(f"✓ Loaded Field-based Neural A* model")
        except ImportError:
            print("✗ NeuralAstarField not found!")
            print("  Please ensure NeuralAstarField is implemented in src/neural_astar/planner/astar.py")
            return False
        except Exception as e:
            print(f"✗ Failed to load Field model: {e}")
            return False
    
    else:  # vanilla
        # Vanilla A*
        planner = VanillaAstar()
        print(f"✓ Using Vanilla A*")

    # Create output directory
    savedir = f"{config.resultdir}/{planner_type}"
    os.makedirs(savedir, exist_ok=True)

    # Generate GIF
    try:
        outputs = extract_astar_output(
            planner(
                map_designs[problem_id : problem_id + 1],
                start_maps[problem_id : problem_id + 1],
                goal_maps[problem_id : problem_id + 1],
                store_intermediate_results=True,
            )
        )
        
        frames = [
            visualize_results(
                map_designs[problem_id : problem_id + 1], intermediate_results, scale=4
            )
            for intermediate_results in outputs.intermediate_results
        ]
        
        output_path = f"{savedir}/video_{dataname}_{problem_id:04d}.gif"
        clip = ImageSequenceClip(frames + [frames[-1]] * 15, fps=30)
        clip.write_gif(output_path)
        
        print(f"✓ GIF created: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create GIF: {e}")
        return False


@hydra.main(config_path="config", config_name="create_gif", version_base="1.2")
def main(config):
    dataname = os.path.basename(config.dataset)
    
    # Load dataset once for all planners
    print(f"\nLoading dataset: {config.dataset}.npz")
    dataloader = create_dataloader(
        config.dataset + ".npz",
        "test",
        100,
        shuffle=False,
        num_starts=1,
    )
    map_designs, start_maps, goal_maps, opt_trajs = next(iter(dataloader))
    print(f"✓ Dataset loaded successfully")
    print(f"  Map size: {map_designs.shape}")
    print(f"  Number of test samples: {len(map_designs)}")
    print(f"  Problem ID to visualize: {config.problem_id}")
    
    # Process all three planners
    planners_to_process = ["va", "na", "field"]
    results = {}
    
    for planner_type in planners_to_process:
        success = create_gif_for_planner(
            planner_type, config, dataloader, 
            map_designs, start_maps, goal_maps
        )
        results[planner_type] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for planner_type, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {planner_type.upper():8s}: {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

# python scripts/create_gif.py 