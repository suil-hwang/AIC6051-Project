import os
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from neural_astar.planner import VanillaAstar, NeuralAstar
from neural_astar.planner.astar import NeuralAstarField  # User's custom class
from neural_astar.utils.data import create_dataloader, visualize_results
from neural_astar.utils.training import load_from_ptl_checkpoint


def extract_astar_output(outputs):
    """Handle models that return either AstarOutput or a tuple including it."""
    from neural_astar.planner.differentiable_astar import AstarOutput

    if isinstance(outputs, AstarOutput):
        return outputs
    if isinstance(outputs, tuple):
        return outputs[0]
    raise TypeError(f"Unexpected output type: {type(outputs)}")


def calculate_paper_metrics(pred_paths, pred_histories, opt_lengths, vanilla_explorations=None):
    """
    Compute paper metrics.
    Opt: Path Optimality Ratio (% of shortest path predictions)
    Exp: Reduction Ratio of Node Explorations
    """
    path_lengths = pred_paths.sum((1, 2, 3)).detach().cpu().numpy()
    explorations = pred_histories.sum((1, 2, 3)).detach().cpu().numpy()
    opt_lengths_np = opt_lengths.sum((1, 2, 3)).detach().cpu().numpy()

    # percentage of shortest path predictions
    is_optimal = (path_lengths <= opt_lengths_np * 1.001) | (path_lengths <= opt_lengths_np + 0.5)

    # Exp = max(100 * (E* - E) / E*, 0)
    if vanilla_explorations is not None:
        reduction = (vanilla_explorations - explorations) / (vanilla_explorations + 1e-8)
        exp_ratios = np.maximum(100 * reduction, 0)
    else:
        exp_ratios = np.zeros_like(explorations)

    return is_optimal, exp_ratios, path_lengths, explorations


def run_comparison(models, dataloader, device="cuda"):
    """
    Evaluate all models on the dataloader and collect metrics.
    """
    results = {name: {"is_optimal": [], "exp": [], "times": [], "explorations": []} for name in models}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Models"):
            map_designs, start_maps, goal_maps, opt_trajs = batch
            map_designs = map_designs.to(device)
            start_maps = start_maps.to(device)
            goal_maps = goal_maps.to(device)

            batch_vanilla_exp = None

            # 1. Vanilla A* (baseline)
            if "Vanilla" in models:
                planner = models["Vanilla"]

                if device == "cuda":
                    torch.cuda.synchronize()
                start_t = time.time()
                outputs = planner(map_designs, start_maps, goal_maps)
                if device == "cuda":
                    torch.cuda.synchronize()
                runtime = time.time() - start_t

                is_opt, exp, _, expls = calculate_paper_metrics(
                    outputs.paths, outputs.histories, opt_trajs, vanilla_explorations=None
                )

                results["Vanilla"]["is_optimal"].extend(is_opt)
                results["Vanilla"]["exp"].extend(exp)
                results["Vanilla"]["times"].append(runtime / map_designs.size(0))  # Per sample time avg
                results["Vanilla"]["explorations"].extend(expls)

                batch_vanilla_exp = expls

            # 2. Other Models (Neural, Field)
            for name, planner in models.items():
                if name == "Vanilla":
                    continue

                if device == "cuda":
                    torch.cuda.synchronize()
                start_t = time.time()

                if isinstance(planner, NeuralAstarField):
                    outputs, _, _ = planner(map_designs, start_maps, goal_maps)
                else:
                    outputs = planner(map_designs, start_maps, goal_maps)

                if device == "cuda":
                    torch.cuda.synchronize()
                runtime = time.time() - start_t

                ref_exp = batch_vanilla_exp if batch_vanilla_exp is not None else None

                is_opt, exp, _, expls = calculate_paper_metrics(
                    outputs.paths, outputs.histories, opt_trajs, vanilla_explorations=ref_exp
                )

                results[name]["is_optimal"].extend(is_opt)
                results[name]["exp"].extend(exp)
                results[name]["times"].append(runtime / map_designs.size(0))
                results[name]["explorations"].extend(expls)

    return results


def create_comparison_animation(models, sample_batch, device, avg_times=None, save_path="comparison.gif"):
    """
    Use intermediate_results (same as create_gif.py) to build the animation and play it in the plot.
    """
    print(f"\nGenerating animation to {save_path}...")
    map_designs, start_maps, goal_maps, _ = sample_batch

    map_d = map_designs[0:1].to(device)
    start_m = start_maps[0:1].to(device)
    goal_m = goal_maps[0:1].to(device)
    map_d_cpu = map_d.cpu()

    frames_by_model = {}
    runtimes_ms = {}

    # Collect frames using the same logic as scripts/create_gif.py
    with torch.no_grad():
        for name, planner in models.items():
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            try:
                raw_outputs = planner(
                    map_d, start_m, goal_m, store_intermediate_results=True
                )
            except (TypeError, AssertionError):
                raw_outputs = planner(map_d, start_m, goal_m)
            if device == "cuda":
                torch.cuda.synchronize()
            runtimes_ms[name] = (time.time() - t0) * 1000

            outputs = extract_astar_output(raw_outputs)
            intermediate_results = getattr(outputs, "intermediate_results", None)
            if intermediate_results:
                frames = [
                    visualize_results(map_d_cpu, intermediate_results, scale=4)
                    for intermediate_results in intermediate_results
                ]
            else:
                # Fallback: build frames manually from histories/paths
                hist_np = outputs.histories[0].detach().cpu().numpy()  # (T, H, W)
                path_np = outputs.paths[0].detach().cpu().numpy().squeeze()
                frames = []
                for step_idx in range(hist_np.shape[0]):
                    current_hist = hist_np[step_idx]
                    current_path = path_np if step_idx >= hist_np.shape[0] - 1 else np.zeros_like(path_np)
                    h_tensor = torch.from_numpy(current_hist).unsqueeze(0).unsqueeze(0)
                    p_tensor = torch.from_numpy(current_path).unsqueeze(0).unsqueeze(0)
                    planner_outputs = {"histories": h_tensor, "paths": p_tensor}
                    frames.append(visualize_results(map_d_cpu, planner_outputs, scale=4))
            frames_by_model[name] = frames

    max_frames = max(len(frames) for frames in frames_by_model.values())
    for name, frames in frames_by_model.items():
        if len(frames) < max_frames:
            frames_by_model[name] = frames + [frames[-1]] * (max_frames - len(frames))

    # Add a 1s pause at the end by repeating the final frame (interval is 1 ms).
    interval_ms = 1
    pause_ms = 500
    pause_tail_frames = max(1, pause_ms // interval_ms)
    for name, frames in frames_by_model.items():
        frames_by_model[name] = frames + [frames[-1]] * pause_tail_frames
    total_frames = max(len(frames) for frames in frames_by_model.values())

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    img_artists = []
    for ax, name in zip(axes, models.keys()):
        img_artist = ax.imshow(frames_by_model[name][0])
        display_time = avg_times[name] if avg_times and name in avg_times else runtimes_ms[name]
        ax.set_title(f"{name}\nTime: {display_time:.2f}ms")
        ax.axis("off")
        img_artists.append(img_artist)

    def update(frame_idx):
        artists = []
        for i, name in enumerate(models.keys()):
            frame = frames_by_model[name][min(frame_idx, len(frames_by_model[name]) - 1)]
            img_artists[i].set_data(frame)
            artists.append(img_artists[i])
        return artists

    # 1 frame per 1ms playback in the plot
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        blit=True,
        interval=interval_ms,
        repeat=True,
        repeat_delay=pause_ms,
    )
    ani.save(save_path, writer="pillow", fps=1000)
    print("Animation saved.")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to .npz dataset (without .npz)")
    parser.add_argument("--model-path", type=str, required=True, help="Directory containing model subfolders")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading dataset from {args.dataset_path}...")
    test_loader = create_dataloader(args.dataset_path + ".npz", "test", batch_size=50, shuffle=False)

    # 2. Initialize Models
    models = {}

    # (A) Vanilla A*
    print("Initializing Vanilla A*...")
    # Enable differentiable A* so that intermediate_results are available for animation.
    models["Vanilla"] = VanillaAstar(use_differentiable_astar=True).to(args.device)
    models["Vanilla"].eval()

    dataset_name = os.path.basename(args.dataset_path)

    # (B) Neural A*
    na_path = os.path.join(args.model_path, f"neural_{dataset_name}")
    if os.path.exists(na_path):
        print(f"Loading Neural A* from {na_path}...")
        try:
            na_model = NeuralAstar(encoder_input="m+", encoder_arch="CNN", encoder_depth=4, Tmax=1.0)
            na_model.load_state_dict(load_from_ptl_checkpoint(na_path), strict=False)
            na_model.to(args.device).eval()
            models["Neural"] = na_model
        except Exception as e:
            print(f"Error loading Neural A*: {e}")

    # (C) Neural A* Field
    field_path = os.path.join(args.model_path, f"field_{dataset_name}")
    if os.path.exists(field_path):
        print(f"Loading Neural A* (Field) from {field_path}...")
        try:
            field_model = NeuralAstarField(encoder_input="m+", encoder_arch="CNN", encoder_depth=4, Tmax=1.0)
            field_model.load_state_dict(load_from_ptl_checkpoint(field_path), strict=False)
            field_model.to(args.device).eval()
            models["Field"] = field_model
        except Exception as e:
            print(f"Error loading Neural A* (Field): {e}")

    if len(models) < 1:
        print("No models to evaluate.")
        return

    # 3. Numerical Evaluation
    raw_results = run_comparison(models, test_loader, args.device)

    # Calculate average times for animation consistency (ms)
    avg_times = {name: np.mean(res["times"]) * 1000 for name, res in raw_results.items()}

    # 4. Run Animation on the first batch using create_gif.py style frames
    first_batch = next(iter(test_loader))
    create_comparison_animation(
        models, first_batch, args.device, avg_times=avg_times, save_path="search_process_comparison.gif"
    )

    # 5. Process & Print Final Metrics
    print("\n" + "=" * 60)
    print(f"{'Model':<10} | {'Opt (%)':<10} | {'Exp (%)':<10} | {'Hmean':<10} | {'Time (ms)':<10}")
    print("-" * 60)

    metrics_for_plot = {"Opt": [], "Exp": [], "Hmean": [], "Time": [], "Names": []}

    for name, res in raw_results.items():
        opt_mean = np.mean(res["is_optimal"]) * 100
        exp_mean = np.mean(res["exp"])
        time_mean = np.mean(res["times"]) * 1000

        if opt_mean + exp_mean > 0:
            hmean = 2 * (opt_mean * exp_mean) / (opt_mean + exp_mean)
        else:
            hmean = 0.0

        print(f"{name:<10} | {opt_mean:<10.2f} | {exp_mean:<10.2f} | {hmean:<10.2f} | {time_mean:<10.2f}")

        metrics_for_plot["Names"].append(name)
        metrics_for_plot["Opt"].append(opt_mean)
        metrics_for_plot["Exp"].append(exp_mean)
        metrics_for_plot["Hmean"].append(hmean)
        metrics_for_plot["Time"].append(time_mean)
    print("=" * 60)

    # 6. Plot Metrics
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

    axes[0].bar(metrics_for_plot["Names"], metrics_for_plot["Opt"], color=colors, alpha=0.8)
    axes[0].set_title("Path Optimality Ratio (Opt)\n(Higher is better)")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 105)

    axes[1].bar(metrics_for_plot["Names"], metrics_for_plot["Exp"], color=colors, alpha=0.8)
    axes[1].set_title("Exploration Reduction (Exp)\n(Higher is better)")
    axes[1].set_ylabel("%")

    axes[2].bar(metrics_for_plot["Names"], metrics_for_plot["Hmean"], color=colors, alpha=0.8)
    axes[2].set_title("Harmonic Mean (Hmean)\n(Higher is better)")

    axes[3].bar(metrics_for_plot["Names"], metrics_for_plot["Time"], color=colors, alpha=0.8)
    axes[3].set_title("Average Runtime\n(Lower is better)")
    axes[3].set_ylabel("ms")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# python scripts\compare_models.py --dataset-path data/maze/gaps_and_forest_032_moore_c8  --model-path model/
