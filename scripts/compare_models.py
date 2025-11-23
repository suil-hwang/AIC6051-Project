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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path = "planning-datasets/data/mpd/mazes_032_moore_c8"
    
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
    neural_planner = NeuralAstar(
        encoder_input="m+",
        encoder_arch="CNN",
        encoder_depth=4,
    )
    checkpoint_path = "model/mazes_032_moore_c8"
    if os.path.exists(checkpoint_path):
        neural_planner.load_state_dict(load_from_ptl_checkpoint(checkpoint_path))
        results['neural'] = evaluate_planner(neural_planner, test_loader, device)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        results['neural'] = None
    
    # 3. Neural A* with Fields
    print("\n=== Evaluating Neural A* with Fields ===")
    field_planner = NeuralAstarField(
        encoder_input="m+",
        encoder_arch="CNN",
        encoder_depth=4,
        use_geodesic_loss=True,
        use_obstacle_loss=True,
    )
    checkpoint_path_field = "model_field/field_mazes_032_moore_c8"
    if os.path.exists(checkpoint_path_field):
        field_planner.load_state_dict(load_from_ptl_checkpoint(checkpoint_path_field))
        results['field'] = evaluate_planner(field_planner, test_loader, device)
    else:
        print(f"Checkpoint not found: {checkpoint_path_field}")
        results['field'] = None
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
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
        
        axes[idx].bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(names)])
        axes[idx].set_title(title)
        axes[idx].set_ylabel('Value')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150)
    print("\nVisualization saved to comparison_results.png")
    plt.show()


if __name__ == "__main__":
    main()