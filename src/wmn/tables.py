"""
Table generation and export utilities
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os


def create_clients_sweep_table(results: Dict[int, Dict], save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create table for clients sweep experiment
    
    Args:
        results: Dictionary mapping n (number of clients) to statistics
        save_path: Path to save CSV (optional)
        
    Returns:
        DataFrame with results
    """
    rows = []
    for n, stats in sorted(results.items()):
        rows.append({
            'n': n,
            'Mean Coverage Count': f"{stats['mean_coverage_count']:.2f}",
            'Std Coverage Count': f"{stats['std_coverage_count']:.2f}",
            'Mean Connectivity Count': f"{stats['mean_connectivity_count']:.2f}",
            'Std Connectivity Count': f"{stats['std_connectivity_count']:.2f}",
            'Mean Fitness': f"{stats['mean_fitness']:.4f}",
            'Std Fitness': f"{stats['std_fitness']:.4f}"
        })
    
    df = pd.DataFrame(rows)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved table to {save_path}")
    
    return df


def create_routers_sweep_table(results: Dict[int, Dict], save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create table for routers sweep experiment
    
    Args:
        results: Dictionary mapping m (number of routers) to statistics
        save_path: Path to save CSV (optional)
        
    Returns:
        DataFrame with results
    """
    rows = []
    for m, stats in sorted(results.items()):
        rows.append({
            'm': m,
            'Mean Coverage Count': f"{stats['mean_coverage_count']:.2f}",
            'Std Coverage Count': f"{stats['std_coverage_count']:.2f}",
            'Mean Connectivity Count': f"{stats['mean_connectivity_count']:.2f}",
            'Std Connectivity Count': f"{stats['std_connectivity_count']:.2f}",
            'Mean Fitness': f"{stats['mean_fitness']:.4f}",
            'Std Fitness': f"{stats['std_fitness']:.4f}"
        })
    
    df = pd.DataFrame(rows)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved table to {save_path}")
    
    return df


def create_radius_sweep_table(results: Dict[float, Dict], save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create table for coverage radius sweep experiment
    
    Args:
        results: Dictionary mapping CR (coverage radius) to statistics
        save_path: Path to save CSV (optional)
        
    Returns:
        DataFrame with results
    """
    rows = []
    for CR, stats in sorted(results.items()):
        rows.append({
            'CR': CR,
            'Mean Coverage Count': f"{stats['mean_coverage_count']:.2f}",
            'Std Coverage Count': f"{stats['std_coverage_count']:.2f}",
            'Mean Connectivity Count': f"{stats['mean_connectivity_count']:.2f}",
            'Std Connectivity Count': f"{stats['std_connectivity_count']:.2f}",
            'Mean Fitness': f"{stats['mean_fitness']:.4f}",
            'Std Fitness': f"{stats['std_fitness']:.4f}"
        })
    
    df = pd.DataFrame(rows)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved table to {save_path}")
    
    return df


def create_convergence_table(results: List[Dict], save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create table for convergence analysis
    
    Args:
        results: List of dictionaries with instance results
        save_path: Path to save CSV (optional)
        
    Returns:
        DataFrame with results
    """
    rows = []
    for i, result in enumerate(results, 1):
        convergence = result['convergence']  # This is cost history
        best_fitness = result['best_fitness']
        
        # Find iteration where best was reached (minimum cost = maximum fitness)
        best_iteration = np.argmin(convergence) + 1  # +1 for 1-indexed
        
        # Convert final cost to fitness
        final_fitness = 1.0 - convergence[-1]
        
        rows.append({
            'Instance': f"Instance{i}",
            'W': result['W'],
            'H': result['H'],
            'm': result['m'],
            'n': result['n'],
            'CR': result['CR'],
            'Best Fitness': f"{best_fitness:.6f}",
            'Iteration to Best': best_iteration,
            'Final Fitness': f"{final_fitness:.6f}"
        })
    
    df = pd.DataFrame(rows)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved table to {save_path}")
    
    return df

