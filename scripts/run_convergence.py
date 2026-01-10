"""
Script to run convergence analysis (Table 10, Figures 17-20, Table 11)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.wmn.config import WMNConfig
from src.wmn.experiments import run_single_placement
from src.wmn.plotting import plot_convergence_paper_style
from src.wmn.tables import create_convergence_table
import logging
import pandas as pd

# Setup logging
os.makedirs('results/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/convergence.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Define 4 instances consistent with paper's idea (different (m,n,CR) setups)
    # Instance 1: Small network
    # Instance 2: Medium network
    # Instance 3: Large network
    # Instance 4: Very large network
    instances = [
        {'W': 2000, 'H': 2000, 'm': 10, 'n': 50, 'CR': 200, 'name': 'Instance1'},
        {'W': 2000, 'H': 2000, 'm': 15, 'n': 100, 'CR': 200, 'name': 'Instance2'},
        {'W': 2000, 'H': 2000, 'm': 20, 'n': 150, 'CR': 200, 'name': 'Instance3'},
        {'W': 2000, 'H': 2000, 'm': 25, 'n': 200, 'CR': 200, 'name': 'Instance4'},
    ]
    
    results = []
    
    logging.info("Starting convergence analysis...")
    
    for i, instance in enumerate(instances, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing {instance['name']}")
        logging.info(f"W={instance['W']}, H={instance['H']}, m={instance['m']}, n={instance['n']}, CR={instance['CR']}")
        logging.info(f"{'='*60}")
        
        # Create config
        config = WMNConfig(
            W=instance['W'], H=instance['H'],
            m=instance['m'], n=instance['n'], CR=instance['CR'],
            lambda_weight=0.5,
            iterations=800,
            seed=42 + i,  # Different seed for each instance
            runs=1
        )
        
        # Run single optimization
        best_routers, fitness, metrics, convergence = run_single_placement(
            config, seed=config.seed
        )
        
        logging.info(f"Best fitness: {fitness:.6f}")
        logging.info(f"Coverage: {metrics['coverage_count']}/{config.n} ({metrics['coverage_ratio']:.2%})")
        logging.info(f"Connectivity: {metrics['connectivity_count']}/{config.m + config.n} ({metrics['connectivity_ratio']:.2%})")
        
        # Store results
        result = {
            'W': instance['W'],
            'H': instance['H'],
            'm': instance['m'],
            'n': instance['n'],
            'CR': instance['CR'],
            'best_fitness': fitness,
            'convergence': convergence,
            'metrics': metrics
        }
        results.append(result)
        
        # Plot convergence curve (use cost directly, as per paper style)
        os.makedirs('results/figures', exist_ok=True)
        fig_num = 16 + i  # Figures 17, 18, 19, 20
        plot_convergence_paper_style(
            convergence_history=convergence,  # Already cost values
            title=f"Convergence Curve - {instance['name']}",
            save_path=f'results/figures/fig{fig_num}_convergence_{instance["name"].lower()}.png',
            algorithm='COA'
        )
        
        # Save convergence CSV per instance
        os.makedirs('results/tables', exist_ok=True)
        conv_df = pd.DataFrame({
            'iteration': range(1, len(convergence) + 1),
            'cost': convergence
        })
        conv_df.to_csv(f'results/tables/convergence_{instance["name"].lower()}.csv', index=False)
        logging.info(f"Saved convergence data to results/tables/convergence_{instance['name'].lower()}.csv")
    
    # Create convergence table
    os.makedirs('results/tables', exist_ok=True)
    df = create_convergence_table(results, save_path='results/tables/table11_convergence.csv')
    print("\nTable 11 (Convergence Analysis):")
    print(df.to_string(index=False))
    
    logging.info("\nConvergence analysis completed!")
    logging.info("Results saved to:")
    for i, instance in enumerate(instances, 1):
        fig_num = 16 + i
        logging.info(f"  - results/figures/fig{fig_num}_convergence_{instance['name'].lower()}.png")
        logging.info(f"  - results/tables/convergence_{instance['name'].lower()}.csv")
    logging.info("  - results/tables/table11_convergence.csv")

if __name__ == '__main__':
    main()

