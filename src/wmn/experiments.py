"""
Experiment execution for WMN placement
"""
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
from .config import WMNConfig
from .geometry import generate_clients, solution_to_positions
from .metrics import compute_fitness
from .coa import COAOptimizer


def create_objective_function(clients: np.ndarray, config: WMNConfig) -> callable:
    """
    Create objective function for optimization
    
    Args:
        clients: Client positions (fixed)
        config: WMN configuration
        
    Returns:
        Objective function that takes solution array and returns cost
    """
    def objective(solution: np.ndarray) -> float:
        routers = solution_to_positions(solution)
        _, cost, _ = compute_fitness(
            clients, routers, config.CR, config.comm_radius,
            config.lambda_weight, use_combined_connectivity=True
        )
        return cost
    
    return objective


def run_single_placement(config: WMNConfig, clients: Optional[np.ndarray] = None,
                        seed: Optional[int] = None) -> Tuple[np.ndarray, float, Dict, List[float]]:
    """
    Run single COA placement optimization
    
    Args:
        config: WMN configuration
        clients: Client positions (if None, generate new)
        seed: Random seed
        
    Returns:
        Tuple of (best_routers, best_fitness, metrics_dict, convergence_history)
    """
    # Generate or use provided clients
    if clients is None:
        client_seed = seed if config.fix_clients else None
        clients = generate_clients(config.n, config.W, config.H, seed=client_seed)
    
    # Create objective function
    objective = create_objective_function(clients, config)
    
    # Run COA
    optimizer = COAOptimizer(config, objective)
    best_solution, best_cost, convergence = optimizer.optimize(seed=seed)
    
    # Convert solution to router positions
    best_routers = solution_to_positions(best_solution)
    
    # Compute final metrics
    fitness, _, metrics = compute_fitness(
        clients, best_routers, config.CR, config.comm_radius,
        config.lambda_weight, use_combined_connectivity=True
    )
    
    return best_routers, fitness, metrics, convergence


def run_multiple_runs(config: WMNConfig, num_runs: Optional[int] = None,
                     clients: Optional[np.ndarray] = None) -> Dict:
    """
    Run multiple independent runs and collect statistics
    
    Args:
        config: WMN configuration
        num_runs: Number of runs (default: config.runs)
        clients: Client positions (if None, generate new)
        
    Returns:
        Dictionary with statistics
    """
    if num_runs is None:
        num_runs = config.runs
    
    # Generate clients once if not provided
    if clients is None:
        clients = generate_clients(config.n, config.W, config.H, seed=config.seed)
    
    # Collect results
    coverage_counts = []
    connectivity_counts = []
    fitnesses = []
    all_convergences = []
    
    for run_idx in range(num_runs):
        seed = config.seed + run_idx if config.seed is not None else None
        _, fitness, metrics, convergence = run_single_placement(
            config, clients=clients, seed=seed
        )
        
        coverage_counts.append(metrics['coverage_count'])
        connectivity_counts.append(metrics['connectivity_count'])
        fitnesses.append(fitness)
        all_convergences.append(convergence)
        
        if (run_idx + 1) % 10 == 0:
            logging.info(f"Completed {run_idx + 1}/{num_runs} runs")
    
    # Compute statistics
    stats = {
        'mean_coverage_count': np.mean(coverage_counts),
        'std_coverage_count': np.std(coverage_counts),
        'mean_connectivity_count': np.mean(connectivity_counts),
        'std_connectivity_count': np.std(connectivity_counts),
        'mean_fitness': np.mean(fitnesses),
        'std_fitness': np.std(fitnesses),
        'best_fitness': np.max(fitnesses),
        'worst_fitness': np.min(fitnesses),
        'coverage_counts': coverage_counts,
        'connectivity_counts': connectivity_counts,
        'fitnesses': fitnesses,
        'convergences': all_convergences
    }
    
    return stats

