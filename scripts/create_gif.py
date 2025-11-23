import os

import hydra
import moviepy.editor as mpy
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader, visualize_results
from neural_astar.utils.training import load_from_ptl_checkpoint


def extract_astar_output(outputs):
    """Handle models that return either AstarOutput or a tuple including it."""
    return outputs[0] if isinstance(outputs, tuple) else outputs


@hydra.main(config_path="config", config_name="create_gif", version_base="1.2")
def main(config):
    dataname = os.path.basename(config.dataset)

    # Support for vanilla (va), neural (na), and field-based (field) planners
    if config.planner == "na":
        # Neural A* (Original)
        planner = NeuralAstar(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=False,
            Tmax=1.0,
        )
        planner.load_state_dict(
            load_from_ptl_checkpoint(f"{config.modeldir}/neural_{dataname}")
        )
    
    elif config.planner == "field":
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
        except ImportError:
            print("Error: NeuralAstarField not found!")
            print("Please ensure NeuralAstarField is implemented in src/neural_astar/planner/astar.py")
            return

    else:
        # Vanilla A*
        planner = VanillaAstar()

    problem_id = config.problem_id
    savedir = f"{config.resultdir}/{config.planner}"
    os.makedirs(savedir, exist_ok=True)

    dataloader = create_dataloader(
        config.dataset + ".npz",
        "test",
        100,
        shuffle=False,
        num_starts=1,
    )
    map_designs, start_maps, goal_maps, opt_trajs = next(iter(dataloader))
    outputs = extract_astar_output(
        planner(map_designs, start_maps, goal_maps, store_intermediate_results=True)
    )

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
    clip = mpy.ImageSequenceClip(frames + [frames[-1]] * 15, fps=30)
    clip.write_gif(f"{savedir}/video_{dataname}_{problem_id:04d}.gif")


if __name__ == "__main__":
    main()
