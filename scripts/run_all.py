"""
Script to run all experiments
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import time

# Setup logging
os.makedirs('results/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/run_all.log'),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("="*80)
    logging.info("Running all experiments for COA WMN placement")
    logging.info("="*80)
    
    scripts = [
        ('run_placement.py', 'Placement plot (Figure 3)'),
        ('run_clients_sweep.py', 'Clients sweep (Table 7, Figure 14)'),
        ('run_routers_sweep.py', 'Routers sweep (Table 8, Figure 15)'),
        ('run_radius_sweep.py', 'Radius sweep (Table 9, Figure 16)'),
        ('run_convergence.py', 'Convergence analysis (Table 11, Figures 17-20)'),
    ]
    
    start_time = time.time()
    
    for script_name, description in scripts:
        logging.info(f"\n{'='*80}")
        logging.info(f"Running: {script_name} - {description}")
        logging.info(f"{'='*80}\n")
        
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        
        try:
            # Import and run the script
            module_name = script_name.replace('.py', '')
            if module_name == 'run_placement':
                from scripts.run_placement import main as run_script
            elif module_name == 'run_clients_sweep':
                from scripts.run_clients_sweep import main as run_script
            elif module_name == 'run_routers_sweep':
                from scripts.run_routers_sweep import main as run_script
            elif module_name == 'run_radius_sweep':
                from scripts.run_radius_sweep import main as run_script
            elif module_name == 'run_convergence':
                from scripts.run_convergence import main as run_script
            
            run_script()
            logging.info(f"✓ Completed: {script_name}\n")
            
        except Exception as e:
            logging.error(f"✗ Error in {script_name}: {e}", exc_info=True)
            logging.info("Continuing with next script...\n")
    
    elapsed_time = time.time() - start_time
    
    logging.info("\n" + "="*80)
    logging.info("All experiments completed!")
    logging.info(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    logging.info("="*80)
    logging.info("\nResults are saved in:")
    logging.info("  - results/figures/ (all plots)")
    logging.info("  - results/tables/ (all CSV tables)")
    logging.info("  - results/logs/ (all log files)")

if __name__ == '__main__':
    main()

