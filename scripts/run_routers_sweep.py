"""
Script to sweep number of routers (Table 8 + Figure 15)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.wmn.config import WMNConfig
from src.wmn.experiments import run_multiple_runs
from src.wmn.plotting import plot_bar_chart
from src.wmn.tables import create_routers_sweep_table
import logging

# Setup logging
os.makedirs('results/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/routers_sweep.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Base configuration
    base_config = WMNConfig(
        W=2000, H=2000,
        n=50, CR=200,
        lambda_weight=0.5,
        iterations=1000,
        runs=50,  # Can reduce for quick tests
        seed=42
    )
    
    # Values of m to test
    m_values = [5, 10, 15, 20, 25, 30, 35, 40]
    
    results = {}
    
    logging.info("Starting routers sweep experiment...")
    logging.info(f"Testing m values: {m_values}")
    
    for m in m_values:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing m = {m}")
        logging.info(f"{'='*60}")
        
        # Update config
        config = WMNConfig(
            W=base_config.W, H=base_config.H,
            m=m, n=base_config.n, CR=base_config.CR,
            lambda_weight=base_config.lambda_weight,
            iterations=base_config.iterations,
            runs=base_config.runs,
            seed=base_config.seed
        )
        
        # Run multiple runs
        stats = run_multiple_runs(config, num_runs=config.runs)
        results[m] = stats
        
        logging.info(f"m={m}: Mean fitness = {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
    
    # Create table
    os.makedirs('results/tables', exist_ok=True)
    df = create_routers_sweep_table(results, save_path='results/tables/table8_routers_sweep.csv')
    print("\nTable 8 (Routers Sweep):")
    print(df.to_string(index=False))
    
    # Create bar chart
    os.makedirs('results/figures', exist_ok=True)
    plot_data = {
        'Coverage Count': [results[m]['mean_coverage_count'] for m in m_values],
        'Connectivity Count': [results[m]['mean_connectivity_count'] for m in m_values],
        'Fitness': [results[m]['mean_fitness'] for m in m_values]
    }
    
    plot_bar_chart(
        plot_data,
        xlabel='Number of Routers (m)',
        ylabel='Value',
        title='Impact of Varying Number of Routers',
        save_path='results/figures/fig15_routers_sweep.png',
        group_labels=[str(m) for m in m_values]
    )
    
    logging.info("\nExperiment completed!")
    logging.info("Results saved to:")
    logging.info("  - results/tables/table8_routers_sweep.csv")
    logging.info("  - results/figures/fig15_routers_sweep.png")

if __name__ == '__main__':
    main()

