"""
Parallel Script to sweep number of routers (Table 8 + Figure 15)
Uses CPU multiprocessing to speed up independent runs.
- Uses src.wmn.experiments.run_single_placement as the "single run" function.
- Keeps client positions FIXED per m (same clients across runs for that config value) to reduce variance.
- max_workers is set to 4 for usability while multitasking.
"""

import os
import sys
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.wmn.config import WMNConfig
from src.wmn.plotting import plot_paper_style_bars_for_sweep
from src.wmn.tables import create_routers_sweep_table
from src.wmn.geometry import generate_clients
from src.wmn.experiments import run_single_placement

# ---------------------------------------------------------------------
# Windows multiprocessing safety (IMPORTANT)
# ---------------------------------------------------------------------
import multiprocessing as mp

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
os.makedirs("results/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("results/logs/routers_sweep_parallel.log"),
        logging.StreamHandler()
    ]
)

def worker_run(cfg_dict: dict, clients: np.ndarray, run_idx: int) -> dict:
    """
    Run a single placement optimization and return the metrics we need.
    run_single_placement returns: (best_routers, fitness, metrics_dict, convergence_history)
    """
    cfg = WMNConfig(**cfg_dict)
    seed = (cfg.seed + run_idx) if (cfg.seed is not None) else None

    _, fitness, metrics, _ = run_single_placement(cfg, clients=clients, seed=seed)

    return {
        "fitness": float(fitness),
        "coverage_count": float(metrics["coverage_count"]),
        "connectivity_count": float(metrics["connectivity_count"]),
    }

def run_multiple_runs_parallel(config: WMNConfig, num_runs: int, clients: np.ndarray,
                               max_workers: int = 4, progress_every: int = 10) -> dict:
    """
    Parallel equivalent of experiments.run_multiple_runs.
    """
    cfg_dict = dict(config.__dict__)

    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker_run, cfg_dict, clients, i) for i in range(int(num_runs))]
        for f in as_completed(futures):
            results.append(f.result())
            completed += 1
            if progress_every and (completed % progress_every == 0):
                logging.info(f"  Completed {completed}/{num_runs} runs")

    fitness = np.array([r["fitness"] for r in results], dtype=float)
    cov = np.array([r["coverage_count"] for r in results], dtype=float)
    conn = np.array([r["connectivity_count"] for r in results], dtype=float)

    return {
        "mean_fitness": float(np.mean(fitness)),
        "std_fitness": float(np.std(fitness, ddof=1)) if len(fitness) > 1 else 0.0,
        "mean_coverage_count": float(np.mean(cov)),
        "std_coverage_count": float(np.std(cov, ddof=1)) if len(cov) > 1 else 0.0,
        "mean_connectivity_count": float(np.mean(conn)),
        "std_connectivity_count": float(np.std(conn, ddof=1)) if len(conn) > 1 else 0.0,
    }

def main():
    # Base config: n=100, CR=200 as specified
    base_config = WMNConfig(
        W=2000, H=2000,
        n=100, CR=200,
        lambda_weight=0.5,
        iterations=800,
        runs=50,
        seed=42
    )

    m_values = [5, 10, 15, 20, 25, 30, 35, 40]
    results = {}

    logging.info("Starting PARALLEL routers sweep experiment...")
    logging.info(f"Testing m values: {m_values}")
    logging.info("max_workers = 4")

    for m in m_values:
        logging.info("\n" + "=" * 60)
        logging.info(f"Processing m = {m}")
        logging.info("=" * 60)

        cfg = WMNConfig(
            W=base_config.W, H=base_config.H,
            m=m, n=base_config.n, CR=base_config.CR,
            lambda_weight=base_config.lambda_weight,
            iterations=base_config.iterations,
            runs=base_config.runs,
            seed=base_config.seed
        )

        # FIX clients per m (matches original run_multiple_runs)
        clients = generate_clients(cfg.n, cfg.W, cfg.H, seed=cfg.seed)

        stats = run_multiple_runs_parallel(cfg, num_runs=cfg.runs, clients=clients, 
                                          max_workers=4, progress_every=10)
        results[m] = stats

        logging.info(
            f"m={m}: Mean fitness = {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f} | "
            f"Coverage={stats['mean_coverage_count']:.2f} | Connectivity={stats['mean_connectivity_count']:.2f}"
        )

    os.makedirs("results/tables", exist_ok=True)
    df = create_routers_sweep_table(results, save_path="results/tables/table8_routers_sweep_coa.csv")
    print("\nTable 8 (Routers Sweep, COA):")
    print(df.to_string(index=False))

    os.makedirs("results/figures", exist_ok=True)
    coverage_vals = [results[m]["mean_coverage_count"] for m in m_values]
    connectivity_vals = [results[m]["mean_connectivity_count"] for m in m_values]
    fitness_vals = [results[m]["mean_fitness"] for m in m_values]

    plot_paper_style_bars_for_sweep(
        coverage_values=coverage_vals,
        connectivity_values=connectivity_vals,
        fitness_values=fitness_vals,
        x_values=m_values,
        xlabel="Number of Routers (m)",
        title_prefix="Impact of varying number of routers on",
        save_path="results/figures/fig15_routers_sweep_coa.png"
    )

    logging.info("\nExperiment completed!")
    logging.info("Results saved to:")
    logging.info("  - results/tables/table8_routers_sweep_coa.csv")
    logging.info("  - results/figures/fig15_routers_sweep_coa.png")
    logging.info("  - results/logs/routers_sweep_parallel.log")

if __name__ == "__main__":
    # Windows için güvenli başlatma
    mp.freeze_support()
    main()
