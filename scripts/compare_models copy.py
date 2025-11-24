"""Compare Vanilla A*, Neural A*, and Neural A* with Fields
"""
import os
import warnings

# Suppress pkg_resources deprecation warning
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
import numpy as np
import torch
import matplotlib.pyplot as plt
from neural_astar.planner import VanillaAstar, NeuralAstar
from neural_astar.planner.astar import NeuralAstarField
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import load_from_ptl_checkpoint
from tqdm import tqdm
import argparse


def evaluate_planner(planner, dataloader, device='cuda'):
    """Evaluate planner performance"""
    planner.eval()
    planner.to(device)
    
    all_opt_ratios = []
    all_exp_ratios = []
    all_path_lengths = []
    all_explorations = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            map_designs, start_maps, goal_maps, opt_trajs = batch
            map_designs = map_designs.to(device)
            start_maps = start_maps.to(device)
            goal_maps = goal_maps.to(device)
            
            # Forward
            if isinstance(planner, NeuralAstarField):
                outputs, _, _ = planner(map_designs, start_maps, goal_maps)
            else:
                outputs = planner(map_designs, start_maps, goal_maps)
            
            # Metrics
            path_lengths = outputs.paths.sum((1, 2, 3)).cpu().numpy()
            explorations = outputs.histories.sum((1, 2, 3)).cpu().numpy()
            opt_lengths = opt_trajs.sum((1, 2, 3)).cpu().numpy()
            
            opt_ratios = path_lengths / (opt_lengths + 1e-8)
            
            all_opt_ratios.extend(opt_ratios)
            all_path_lengths.extend(path_lengths)
            all_explorations.extend(explorations)
    
    return {
        'opt_ratio_mean': np.mean(all_opt_ratios),
        'opt_ratio_std': np.std(all_opt_ratios),
        'opt_rate': (np.array(all_opt_ratios) <= 1.01).mean(),  # ≈optimal
        'exploration_mean': np.mean(all_explorations),
        'exploration_std': np.std(all_explorations),
        'path_length_mean': np.mean(all_path_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to .npz dataset (without .npz extension)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model directory containing subfolders for neural, field, vanilla")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path = args.dataset_path
    
    # Load test data
    test_loader = create_dataloader(
        dataset_path + ".npz",
        "test",
        batch_size=32,
        shuffle=False,
    )
    
    results = {}
"""Compare Vanilla A*, Neural A*, and Neural A* with Fields
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from neural_astar.planner import VanillaAstar, NeuralAstar
from neural_astar.planner.astar import NeuralAstarField
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import load_from_ptl_checkpoint
from tqdm import tqdm
import argparse


def evaluate_planner(planner, dataloader, device='cuda'):
    """Evaluate planner performance"""
    planner.eval()
    planner.to(device)
    
    all_opt_ratios = []
    all_exp_ratios = []
    all_path_lengths = []
    all_explorations = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            map_designs, start_maps, goal_maps, opt_trajs = batch
            map_designs = map_designs.to(device)
            start_maps = start_maps.to(device)
            goal_maps = goal_maps.to(device)
            
            # Forward
            if isinstance(planner, NeuralAstarField):
                outputs, _, _ = planner(map_designs, start_maps, goal_maps)
            else:
                outputs = planner(map_designs, start_maps, goal_maps)
            
            # Metrics
            path_lengths = outputs.paths.sum((1, 2, 3)).cpu().numpy()
            explorations = outputs.histories.sum((1, 2, 3)).cpu().numpy()
            opt_lengths = opt_trajs.sum((1, 2, 3)).cpu().numpy()
            
            opt_ratios = path_lengths / (opt_lengths + 1e-8)
            
            all_opt_ratios.extend(opt_ratios)
            all_path_lengths.extend(path_lengths)
            all_explorations.extend(explorations)
    
    return {
        'opt_ratio_mean': np.mean(all_opt_ratios),
        'opt_ratio_std': np.std(all_opt_ratios),
        'opt_rate': (np.array(all_opt_ratios) <= 1.01).mean(),  # ≈optimal
        'exploration_mean': np.mean(all_explorations),
        'exploration_std': np.std(all_explorations),
        'path_length_mean': np.mean(all_path_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to .npz dataset (without .npz extension)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model directory containing subfolders for neural, field, vanilla")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path = args.dataset_path
    
    # Load test data
    test_loader = create_dataloader(
        dataset_path + ".npz",
        "test",
        batch_size=32,
        shuffle=False,
    )
    
    results = {}
    
    # 1. Vanilla A*
    print("\n=== Evaluating Vanilla A* ===")
    vanilla_planner = VanillaAstar(use_differentiable_astar=False)
    results['vanilla'] = evaluate_planner(vanilla_planner, test_loader, device)
    
    # 2. Neural A*
    print("\n=== Evaluating Neural A* ===")
    dataset_name = os.path.basename(dataset_path)
    neural_model_dir = os.path.join(args.model_path, f"neural_{dataset_name}")
    
    if os.path.exists(neural_model_dir):
        try:
            neural_planner = NeuralAstar(
                encoder_input="m+",
                encoder_arch="CNN",
                encoder_depth=4,
                Tmax=1.0
            )
            state_dict = load_from_ptl_checkpoint(neural_model_dir)
            neural_planner.load_state_dict(state_dict, strict=False)
            results['neural'] = evaluate_planner(neural_planner, test_loader, device)
        except Exception as e:
            print(f"Failed to load Neural A*: {e}")
            results['neural'] = None
    else:
        print(f"Neural A* model not found at {neural_model_dir}")
        results['neural'] = None

    # 3. Neural A* with Fields (Ours)
    print("\n=== Evaluating Neural A* (Field) ===")
    field_model_dir = os.path.join(args.model_path, f"field_{dataset_name}")
    
    if os.path.exists(field_model_dir):
        try:
            field_planner = NeuralAstarField(
                encoder_input="m+",
                encoder_arch="CNN",
                encoder_depth=4,
                Tmax=1.0
            )
            state_dict = load_from_ptl_checkpoint(field_model_dir)
            field_planner.load_state_dict(state_dict, strict=False)
            results['field'] = evaluate_planner(field_planner, test_loader, device)
        except Exception as e:
            print(f"Failed to load Neural A* (Field): {e}")
            results['field'] = None
    else:
        print(f"Neural A* (Field) model not found at {field_model_dir}")
        results['field'] = None
    
    for name, metrics in results.items():
        if metrics is None:
            continue
        print(f"\n{name.upper()}:")
        print(f"  Optimality Ratio: {metrics['opt_ratio_mean']:.4f} ± {metrics['opt_ratio_std']:.4f}")
        print(f"  Optimal Rate: {metrics['opt_rate']:.2%}")
        print(f"  Exploration: {metrics['exploration_mean']:.2f} ± {metrics['exploration_std']:.2f}")
        print(f"  Path Length: {metrics['path_length_mean']:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics_to_plot = ['opt_ratio_mean', 'exploration_mean', 'opt_rate']
    titles = ['Optimality Ratio\n(lower is better)', 
              'Nodes Explored\n(lower is better)', 
              'Optimal Rate\n(higher is better)']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        values = [results[name][metric] for name in results if results[name] is not None]
        names = [name.capitalize() for name in results if results[name] is not None]
        
        # Colors: Vanilla (Blue), Neural (Orange), Field (Green)
        colors = []
        for name in names:
            if name == 'Vanilla': colors.append('#1f77b4')ㅊ
            elif name == 'Neural': colors.append('#ff7f0e')
            elif name == 'Field': colors.append('#2ca02c')
            else: colors.append('gray')

        axes[idx].bar(names, values, color=colors)
        axes[idx].set_title(title)
        axes[idx].set_ylabel('Value')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig('comparison_results.png', dpi=150)
    #print("\nVisualization saved to comparison_results.png")
    plt.show()


if __name__ == "__main__":
    main()

# python scripts\compare_models.py --dataset-path data/maze/gaps_and_forest_032_moore_c8  --model-path model/