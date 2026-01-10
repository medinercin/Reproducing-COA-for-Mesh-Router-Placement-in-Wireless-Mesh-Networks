"""
Script to run single placement and generate placement plot (Figure 3 style)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.wmn.config import WMNConfig
from src.wmn.experiments import run_single_placement
from src.wmn.plotting import plot_placement
from src.wmn.geometry import generate_clients
import logging

# Setup logging
os.makedirs('results/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/placement.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Configuration for representative setup
    config = WMNConfig(
        W=2000, H=2000,
        m=20, n=50, CR=200,
        lambda_weight=0.5,
        iterations=1000,
        seed=42,
        runs=1
    )
    
    logging.info("Running single placement optimization...")
    logging.info(f"Configuration: W={config.W}, H={config.H}, m={config.m}, n={config.n}, CR={config.CR}")
    
    # Generate clients
    clients = generate_clients(config.n, config.W, config.H, seed=config.seed)
    
    # Run optimization
    best_routers, fitness, metrics, convergence = run_single_placement(
        config, clients=clients, seed=config.seed
    )
    
    logging.info(f"Best fitness: {fitness:.6f}")
    logging.info(f"Coverage: {metrics['coverage_count']}/{config.n} ({metrics['coverage_ratio']:.2%})")
    logging.info(f"Connectivity: {metrics['connectivity_count']}/{config.m + config.n} ({metrics['connectivity_ratio']:.2%})")
    
    # Plot placement
    os.makedirs('results/figures', exist_ok=True)
    plot_placement(
        clients, best_routers, config.CR, config.comm_radius,
        config.W, config.H,
        save_path='results/figures/fig_placement_coa.png',
        title=f'COA Router Placement (Fitness: {fitness:.4f})',
        algorithm='COA'
    )
    
    logging.info("Placement plot saved to results/figures/fig_placement_coa.png")

if __name__ == '__main__':
    main()

