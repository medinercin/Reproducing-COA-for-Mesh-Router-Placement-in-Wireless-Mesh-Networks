"""
Script to generate Table 11: Convergence Analysis (COA Reference vs Reproduced)

This script:
1. Reads convergence CSV files for 4 instances
2. Computes best fitness and iteration for reproduced COA
3. Creates comparison table with paper reference values
4. Exports as CSV, LaTeX, and PNG

Usage:
    python make_table11.py

To change input file paths, modify the INPUT_DIR variable below.
To change output paths, modify the OUTPUT_DIR variable below.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


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


def process_convergence_file(csv_path):
    """
    Process a single convergence CSV file and compute best fitness and iteration.
    
    Args:
        csv_path: Path to the convergence CSV file
        
    Returns:
        Tuple of (best_fitness, best_iteration) where:
        - best_fitness: Best (minimum) fitness value achieved (1 - min_cost)
        - best_iteration: 1-based iteration index where best was first achieved
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Find fitness/objective column (it might be named 'cost', 'fitness', etc.)
    # Based on the actual CSV structure, we have 'cost' column
    fitness_col = find_column(df, ["fitness", "best_fitness", "objective", "best_objective", "obj", "best", "cost"])
    
    if fitness_col is None:
        raise ValueError(f"Could not find fitness/objective/cost column in {csv_path}. Available columns: {df.columns.tolist()}")
    
    # Find iteration column
    iter_col = find_column(df, ["iteration", "iter", "t"])
    
    # Get fitness/objective values
    values = df[fitness_col].values
    
    # If column is 'cost', convert to fitness (fitness = 1 - cost, since lower cost = higher fitness)
    # If column is already fitness, use as-is (assuming higher is better)
    if fitness_col.lower() == "cost":
        # Cost values: lower is better, so best fitness = 1 - min_cost
        fitness_values = 1.0 - values
        best_idx = np.argmin(values)  # Find index with minimum cost
    else:
        # Fitness/objective values: if it's fitness, higher is better
        # If it's objective and we're minimizing, lower is better
        # For now, assume we want minimum if it's called 'objective' or maximum if 'fitness'
        if "fitness" in fitness_col.lower():
            fitness_values = values
            best_idx = np.argmax(values)  # Find index with maximum fitness
        else:
            # Objective (minimizing)
            fitness_values = 1.0 - values  # Convert to fitness
            best_idx = np.argmin(values)  # Find index with minimum objective
    
    best_fitness = fitness_values[best_idx]
    
    # Get iteration index (1-based)
    if iter_col is not None:
        # Use existing iteration column
        iterations = df[iter_col].values
        best_iteration = iterations[best_idx]
        # Ensure 1-based (in case CSV has 0-based)
        if best_iteration == 0 and best_idx == 0:
            # Might be 0-based, but we want 1-based
            # Check if all values start from 0 or 1
            if iterations[0] == 0:
                best_iteration = best_idx + 1
            else:
                best_iteration = int(best_iteration)
        else:
            best_iteration = int(best_iteration)
    else:
        # Create 1-based iteration from row index
        best_iteration = best_idx + 1
    
    return best_fitness, best_iteration


def create_table11(input_dir="results/tables", output_dir="results/tables"):
    """
    Create Table 11: Convergence Analysis (COA Reference vs Reproduced).
    
    Args:
        input_dir: Directory containing convergence_instance*.csv files
        output_dir: Directory to save output files
        
    Returns:
        pandas DataFrame with the table
    """
    # Paper reference values (constants)
    paper_values = {
        1: {"fitness": 0.85, "iteration": 308.67},
        2: {"fitness": 0.77, "iteration": 721.60},
        3: {"fitness": 0.68, "iteration": 817.26},
        4: {"fitness": 0.61, "iteration": 911.46},
    }
    
    # Process each instance
    rows = []
    for instance_num in range(1, 5):
        csv_path = os.path.join(input_dir, f"convergence_instance{instance_num}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Convergence file not found: {csv_path}")
        
        # Compute reproduced values
        reproduced_fitness, reproduced_iteration = process_convergence_file(csv_path)
        
        # Get paper values
        paper_fitness = paper_values[instance_num]["fitness"]
        paper_iteration = paper_values[instance_num]["iteration"]
        
        rows.append({
            "Instance": f"Instance{instance_num}",
            "COA (Paper) Fitness": paper_fitness,
            "COA (Paper) Iteration": paper_iteration,
            "COA (Reproduced) Fitness": reproduced_fitness,
            "COA (Reproduced) Iteration": reproduced_iteration,
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Format for display (rounding)
    df_display = df.copy()
    df_display["COA (Paper) Fitness"] = df_display["COA (Paper) Fitness"].apply(lambda x: f"{x:.2f}")
    df_display["COA (Paper) Iteration"] = df_display["COA (Paper) Iteration"].apply(lambda x: f"{x:.2f}")
    df_display["COA (Reproduced) Fitness"] = df_display["COA (Reproduced) Fitness"].apply(lambda x: f"{x:.2f}")
    # Iteration should be integer if it's integer, otherwise 2 decimals
    df_display["COA (Reproduced) Iteration"] = df_display["COA (Reproduced) Iteration"].apply(
        lambda x: f"{int(x)}" if x == int(x) else f"{x:.2f}"
    )
    
    return df, df_display


def save_csv(df, output_path):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to: {output_path}")


def save_latex(df, output_path):
    """Save DataFrame to LaTeX format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Manually create LaTeX table for better control
    latex_lines = []
    latex_lines.append("\\begin{tabular}{l|cc|cc}")
    latex_lines.append("\\hline")
    
    # Header row
    headers = [
        "\\textbf{Instance}",
        "\\textbf{COA (Paper) Fitness}",
        "\\textbf{COA (Paper) Iteration}",
        "\\textbf{COA (Reproduced) Fitness}",
        "\\textbf{COA (Reproduced) Iteration}",
    ]
    latex_lines.append(" & ".join(headers) + " \\\\")
    latex_lines.append("\\hline")
    
    # Data rows
    for _, row in df.iterrows():
        # Format values
        values = [
            str(row["Instance"]),
            f"{row['COA (Paper) Fitness']:.2f}",
            f"{row['COA (Paper) Iteration']:.2f}",
            f"{row['COA (Reproduced) Fitness']:.2f}",
            f"{int(row['COA (Reproduced) Iteration'])}" if row['COA (Reproduced) Iteration'] == int(row['COA (Reproduced) Iteration']) 
            else f"{row['COA (Reproduced) Iteration']:.2f}",
        ]
        latex_lines.append(" & ".join(values) + " \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    latex_str = "\n".join(latex_lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    print(f"Saved LaTeX to: {output_path}")


def save_png(df, output_path, title="Table 11: Convergence Analysis (COA)"):
    """Save DataFrame as PNG image with styling similar to paper."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("tight")
    ax.axis("off")
    
    # Prepare table data
    table_data = []
    for _, row in df.iterrows():
        # Format reproduced iteration (integer if integer, else 2 decimals)
        repro_iter = row['COA (Reproduced) Iteration']
        repro_iter_str = f"{int(repro_iter)}" if repro_iter == int(repro_iter) else f"{repro_iter:.2f}"
        
        table_data.append([
            str(row["Instance"]),
            f"{row['COA (Paper) Fitness']:.2f}",
            f"{row['COA (Paper) Iteration']:.2f}",
            f"{row['COA (Reproduced) Fitness']:.2f}",
            repro_iter_str,
        ])
    
    # Column headers
    col_labels = [
        "Instance",
        "COA (Paper)\nFitness",
        "COA (Paper)\nIteration",
        "COA (Reproduced)\nFitness",
        "COA (Reproduced)\nIteration",
    ]
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Make header bold and styled
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor("#D0D0D0")  # Darker gray for header
        cell.set_text_props(weight="bold", fontsize=11)
        cell.set_edgecolor("black")
        cell.set_linewidth(2.0)
    
    # Style data cells with proper grid lines
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
            cell.set_text_props(fontsize=11)
            if j == 0:
                # Instance column - slightly different background
                cell.set_facecolor("#F8F8F8")
            else:
                cell.set_facecolor("white")
    
    # Add vertical separator lines for better readability
    # Between Instance and Paper columns
    for i in range(len(table_data) + 1):
        cell = table[(i, 1)]
        cell.set_linewidth(2.0)
    # Between Paper and Reproduced columns
    for i in range(len(table_data) + 1):
        cell = table[(i, 3)]
        cell.set_linewidth(2.0)
    
    # Add title
    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    
    print(f"Saved PNG to: {output_path}")


def main():
    """Main function to generate Table 11."""
    # Configuration: Change these paths if needed
    INPUT_DIR = "results/tables"
    OUTPUT_DIR = "results/tables"
    
    print("=" * 70)
    print("Generating Table 11: Convergence Analysis (COA Reference vs Reproduced)")
    print("=" * 70)
    
    # Create table
    print("\nProcessing convergence CSV files...")
    df, df_display = create_table11(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    
    # Display table
    print("\n" + "=" * 70)
    print("Table 11: Convergence Analysis")
    print("=" * 70)
    print(df_display.to_string(index=False))
    print("=" * 70)
    
    # Save outputs
    print("\nSaving outputs...")
    
    # CSV (with formatted values for readability, but also save raw numeric version)
    csv_path = os.path.join(OUTPUT_DIR, "table11_coa_reference_vs_reproduced.csv")
    save_csv(df_display, csv_path)
    
    # Also save raw numeric version for further processing
    csv_raw_path = os.path.join(OUTPUT_DIR, "table11_coa_reference_vs_reproduced_raw.csv")
    save_csv(df, csv_raw_path)
    
    # LaTeX
    tex_path = os.path.join(OUTPUT_DIR, "table11_coa_reference_vs_reproduced.tex")
    save_latex(df, tex_path)
    
    # PNG
    png_path = os.path.join(OUTPUT_DIR, "table11_coa_reference_vs_reproduced.png")
    save_png(df, png_path)
    
    print("\n" + "=" * 70)
    print("Table 11 generation complete!")
    print("=" * 70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"  - {os.path.basename(csv_path)}")
    print(f"  - {os.path.basename(tex_path)}")
    print(f"  - {os.path.basename(png_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
