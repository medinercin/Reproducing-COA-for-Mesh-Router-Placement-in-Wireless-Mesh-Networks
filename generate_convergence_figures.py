"""
Generate convergence analysis figures (Figures 17-20) showing only:
- Reproduced COA Results (from CSV files - step plots)

This script reads convergence CSV files and creates plots matching paper style.

Usage:
    python generate_convergence_figures.py

Output files:
    - Fig17_instance1_COA_reproduced.png / .pdf
    - Fig18_instance2_COA_reproduced.png / .pdf
    - Fig19_instance3_COA_reproduced.png / .pdf
    - Fig20_instance4_COA_reproduced.png / .pdf
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Figure numbers mapping
FIGURE_NUMBERS = {
    1: 17,  # Instance1 -> Figure 17
    2: 18,  # Instance2 -> Figure 18
    3: 19,  # Instance3 -> Figure 19
    4: 20,  # Instance4 -> Figure 20
}

# Configuration
INPUT_DIR = "results/tables"
OUTPUT_DIR = "results/figures"
MAX_ITERATIONS = 1000  # Scale x-axis to 1000 iterations if CSV ends earlier


def find_column(df, possible_names, case_insensitive=True):
    """
    Find a column in DataFrame by trying multiple possible names.
    
    Args:
        df: pandas DataFrame
        possible_names: List of possible column names
        case_insensitive: If True, match column names case-insensitively
        
    Returns:
        Column name if found, None otherwise
    """
    df_columns = df.columns.tolist()
    
    if case_insensitive:
        df_columns_lower = [c.lower() for c in df_columns]
        for name in possible_names:
            name_lower = name.lower()
            if name_lower in df_columns_lower:
                idx = df_columns_lower.index(name_lower)
                return df_columns[idx]
    else:
        for name in possible_names:
            if name in df_columns:
                return name
    
    return None


def read_convergence_data(csv_path):
    """
    Read convergence CSV file and extract iteration and cost values.
    Convert cost to fitness (fitness = 1 - cost).
    
    Args:
        csv_path: Path to convergence CSV file
        
    Returns:
        Tuple of (iterations, fitness_values) as numpy arrays
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Find iteration column
    iter_col = find_column(df, ["iteration", "iter", "t"])
    
    # Find cost column (objective function value)
    cost_col = find_column(df, ["cost", "objective", "obj", "best_objective", "fitness", "best_fitness"])
    
    if cost_col is None:
        raise ValueError(f"Could not find cost/objective column in {csv_path}. Available columns: {df.columns.tolist()}")
    
    # Get iterations (1-based)
    if iter_col is not None:
        iterations = df[iter_col].values.astype(int)
        # Ensure 1-based indexing
        if iterations[0] == 0:
            iterations = iterations + 1
    else:
        # Create 1-based iterations from row index
        iterations = np.arange(1, len(df) + 1)
    
    # Get cost values
    cost_values = df[cost_col].values
    
    # Convert cost to fitness (fitness = 1 - cost)
    # If column is already named 'fitness', check if values are in [0,1] range
    # If so, assume it's already fitness; otherwise convert
    if "fitness" in cost_col.lower():
        # Might already be fitness, but let's check values
        # If all values are <= 1, assume it's fitness; else assume it's cost
        if np.all(cost_values <= 1.0) and np.all(cost_values >= 0.0):
            fitness_values = cost_values
        else:
            # Likely cost, convert to fitness
            fitness_values = 1.0 - cost_values
    else:
        # Definitely cost, convert to fitness
        fitness_values = 1.0 - cost_values
    
    return iterations, fitness_values


def plot_convergence_reproduced(instance_num, iterations, fitness_values, 
                                fig_num, output_dir):
    """
    Plot reproduced COA convergence curve (step plot only, no reference line).
    
    Args:
        instance_num: Instance number (1-4)
        iterations: Array of iteration numbers (1-based)
        fitness_values: Array of fitness values (1 - cost)
        fig_num: Figure number (17-20)
        output_dir: Directory to save output files
    """
    # Create figure with specific size
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Determine x-axis range (scale to 1000 if CSV ends earlier)
    actual_max_iter = int(iterations.max())
    if actual_max_iter < MAX_ITERATIONS:
        max_iter = MAX_ITERATIONS  # Scale to 1000 if CSV ends earlier
    else:
        max_iter = actual_max_iter  # Use actual max if CSV has more iterations
    x_min, x_max = 0, max_iter
    
    # Plot reproduced COA data as step plot (solid blue line, linewidth=2)
    ax.step(
        iterations, fitness_values,
        where='post',  # Right-continuous step (value stays until next point)
        color='#1f77b4',  # Blue color as specified
        linewidth=2,
        linestyle='-',
        label='COA (Reproduced)',
        alpha=1.0,
        zorder=2
    )
    
    # Set axis labels (exactly as specified)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective function value', fontsize=12)
    
    # Set x-axis limits (0 to 1000 or actual max)
    ax.set_xlim(x_min, x_max)
    
    # Set y-axis limits (with some padding)
    y_min = fitness_values.min() * 0.95
    y_max = fitness_values.max() * 1.05
    ax.set_ylim(y_min, y_max)
    
    # Configure grid (very light, alpha ~0.2)
    ax.grid(
        True,
        color='#E0E0E0',
        alpha=0.2,
        linestyle='--',
        linewidth=0.5
    )
    
    # Add legend at top center
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        framealpha=0.9,
        fontsize=10,
        ncol=1
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as PNG
    png_path = os.path.join(output_dir, f"Fig{fig_num}_instance{instance_num}_COA_reproduced.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved PNG: {png_path}")
    
    # Save as PDF
    pdf_path = os.path.join(output_dir, f"Fig{fig_num}_instance{instance_num}_COA_reproduced.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved PDF: {pdf_path}")
    
    plt.close()


def main():
    """Main function to generate all convergence figures (reproduced COA only)."""
    print("=" * 70)
    print("Generating Convergence Analysis Figures (17-20)")
    print("Reproduced COA Only")
    print("=" * 70)
    
    # Process each instance
    for instance_num in range(1, 5):
        print(f"\nProcessing Instance {instance_num}...")
        
        # File paths
        csv_path = os.path.join(INPUT_DIR, f"convergence_instance{instance_num}.csv")
        
        if not os.path.exists(csv_path):
            print(f"WARNING: Convergence file not found: {csv_path}")
            print(f"Skipping Instance {instance_num}")
            continue
        
        # Read convergence data
        try:
            iterations, fitness_values = read_convergence_data(csv_path)
            print(f"  Read {len(iterations)} iterations from {csv_path}")
            print(f"  Fitness range: [{fitness_values.min():.4f}, {fitness_values.max():.4f}]")
        except Exception as e:
            print(f"ERROR: Failed to read {csv_path}: {e}")
            continue
        
        # Get figure number
        fig_num = FIGURE_NUMBERS[instance_num]
        
        # Create plot
        try:
            plot_convergence_reproduced(
                instance_num=instance_num,
                iterations=iterations,
                fitness_values=fitness_values,
                fig_num=fig_num,
                output_dir=OUTPUT_DIR
            )
            print(f"  Generated Figure {fig_num} for Instance {instance_num} successfully")
        except Exception as e:
            print(f"ERROR: Failed to generate plot for Instance {instance_num}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("Convergence figure generation complete!")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
