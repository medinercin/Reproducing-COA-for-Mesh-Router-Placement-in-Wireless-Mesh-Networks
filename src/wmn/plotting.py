"""
Plotting utilities for WMN experiments

Paper Figure Mapping:
- plot_placement: Figures 3-13 (placement visualization per algorithm)
- plot_bar_chart: Figures 14, 15, 16 (bar charts for sweeps)
- plot_convergence: Figures 17, 18, 19, 20 (convergence curves)
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import matplotlib.patches as mpatches
from .graph import build_router_graph, build_combined_graph
from .style import (
    MARKER_STYLES, LINE_STYLES, BAR_STYLES, AXES_STYLES,
    LEGEND_ORDER, COVERAGE_ALPHA,
    get_coverage_color, get_coverage_edge_color, get_algorithm_color
)


def plot_placement(clients: np.ndarray, routers: np.ndarray, CR: float, 
                  comm_radius: float, W: float, H: float,
                  save_path: Optional[str] = None, title: str = "Router Placement",
                  algorithm: str = "COA"):
    """
    Plot router placement with clients, routers, coverage circles, and links
    Matches paper Figures 3-13 styling exactly.
    
    Args:
        clients: Array of shape (n, 2) with client positions
        routers: Array of shape (m, 2) with router positions
        CR: Coverage radius
        comm_radius: Communication radius
        W: Area width
        H: Area height
        save_path: Path to save figure (optional)
        title: Plot title
        algorithm: Algorithm name for coverage circle color (default: "COA")
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(AXES_STYLES['background_color'])
    ax.set_facecolor(AXES_STYLES['background_color'])
    
    # Get algorithm-specific coverage colors
    coverage_color = get_coverage_color(algorithm)
    coverage_edge = get_coverage_edge_color(algorithm)
    
    # Draw coverage circles for each router (translucent filled circles)
    # Z-order: 1 (bottom layer)
    for router in routers:
        circle = mpatches.Circle(
            router, CR,
            facecolor=coverage_color,
            edgecolor=coverage_edge,
            linewidth=LINE_STYLES['coverage_circle']['linewidth'],
            alpha=COVERAGE_ALPHA,
            linestyle=LINE_STYLES['coverage_circle']['linestyle'],
            zorder=LINE_STYLES['coverage_circle']['zorder']
        )
        ax.add_patch(circle)
    
    # Draw router-router links (thin purple lines)
    # Z-order: 2
    router_graph = build_router_graph(routers, comm_radius)
    link_style = LINE_STYLES['router_link']
    for edge in router_graph.edges():
        r1_idx, r2_idx = edge
        pos1 = routers[r1_idx]
        pos2 = routers[r2_idx]
        ax.plot(
            [pos1[0], pos2[0]], [pos1[1], pos2[1]],
            color=link_style['color'],
            linewidth=link_style['linewidth'],
            alpha=link_style['alpha'],
            linestyle=link_style['linestyle'],
            zorder=link_style['zorder']
        )
    
    # Draw clients (small green dots)
    # Z-order: 3
    client_style = MARKER_STYLES['client']
    if len(clients) > 0:
        ax.scatter(
            clients[:, 0], clients[:, 1],
            marker=client_style['marker'],
            s=client_style['size'],
            c=client_style['color'],
            edgecolors=client_style['edgecolor'],
            alpha=client_style['alpha'],
            zorder=client_style['zorder'],
            label='Client'
        )
    
    # Draw routers (blue filled circles)
    # Z-order: 4 (top layer)
    router_style = MARKER_STYLES['router']
    ax.scatter(
        routers[:, 0], routers[:, 1],
        marker=router_style['marker'],
        s=router_style['size'],
        c=router_style['facecolor'],
        edgecolors=router_style['edgecolor'],
        alpha=router_style['alpha'],
        zorder=router_style['zorder'],
        label='Router'
    )
    
    # Set axes limits exactly as paper: x=[0,W], y=[0,H]
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xlabel('X (m)', fontsize=12, color=AXES_STYLES['axis_color'])
    ax.set_ylabel('Y (m)', fontsize=12, color=AXES_STYLES['axis_color'])
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Configure grid
    ax.grid(
        True,
        color=AXES_STYLES['grid_color'],
        alpha=AXES_STYLES['grid_alpha'],
        linestyle=AXES_STYLES['grid_linestyle'],
        linewidth=AXES_STYLES['grid_linewidth']
    )
    
    # Set equal aspect ratio (must match paper)
    ax.set_aspect('equal')
    
    # Legend in exact order: Client, Router, Coverage area, Link
    # Create legend handles in correct order
    handles, labels = ax.get_legend_handles_labels()
    # Add coverage area and link to legend if not already present
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    # Create legend elements in correct order
    legend_elements = [
        Line2D([0], [0], marker=client_style['marker'], color='w',
               markerfacecolor=client_style['color'], markersize=8, label='Client'),
        Line2D([0], [0], marker=router_style['marker'], color='w',
               markerfacecolor=router_style['facecolor'], markersize=10, label='Router'),
        Patch(facecolor=coverage_color, edgecolor=coverage_edge, alpha=COVERAGE_ALPHA, label='Coverage area'),
        Line2D([0], [0], color=link_style['color'], linewidth=link_style['linewidth'], label='Link')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=AXES_STYLES['background_color'])
        print(f"Saved placement plot to {save_path}")
    
    plt.close()


def plot_bar_chart(data: dict, xlabel: str, ylabel: str, title: str,
                   save_path: Optional[str] = None, 
                   group_labels: Optional[list] = None,
                   algorithm_colors: Optional[dict] = None):
    """
    Plot grouped bar chart matching paper Figures 14-16 styling.
    
    Args:
        data: Dictionary with keys as metric names and values as lists of values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        save_path: Path to save figure (optional)
        group_labels: Labels for groups (if None, uses dict keys)
        algorithm_colors: Optional dict mapping metric names to algorithm colors
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(AXES_STYLES['background_color'])
    ax.set_facecolor(AXES_STYLES['background_color'])
    
    if group_labels is None:
        group_labels = list(data.keys())
    
    # Get x positions
    x = np.arange(len(group_labels))
    width = BAR_STYLES['width_ratio'] / len(data)
    
    # Map metric names to algorithm colors
    # Default mapping: Coverage Count -> green, Connectivity Count -> blue, Fitness -> orange
    metric_color_map = {
        'Coverage Count': get_algorithm_color('WOA'),  # Green
        'Connectivity Count': get_algorithm_color('COA'),  # Blue
        'Fitness': get_algorithm_color('FA'),  # Orange
    }
    
    if algorithm_colors:
        metric_color_map.update(algorithm_colors)
    
    # Plot bars for each metric with explicit colors
    offset = -width * (len(data) - 1) / 2
    for i, (metric_name, values) in enumerate(data.items()):
        # Get color for this metric
        bar_color = metric_color_map.get(metric_name, get_algorithm_color('COA'))
        
        ax.bar(
            x + offset, values, width,
            label=metric_name,
            color=bar_color,
            edgecolor=BAR_STYLES['edgecolor'],
            linewidth=BAR_STYLES['edgewidth'],
            alpha=BAR_STYLES['alpha']
        )
        offset += width
    
    ax.set_xlabel(xlabel, fontsize=12, color=AXES_STYLES['axis_color'])
    ax.set_ylabel(ylabel, fontsize=12, color=AXES_STYLES['axis_color'])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend(fontsize=10, framealpha=0.9)
    
    # Configure grid
    ax.grid(
        True,
        axis='y',
        color=AXES_STYLES['grid_color'],
        alpha=AXES_STYLES['grid_alpha'],
        linestyle=AXES_STYLES['grid_linestyle'],
        linewidth=AXES_STYLES['grid_linewidth']
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=AXES_STYLES['background_color'])
        print(f"Saved bar chart to {save_path}")
    
    plt.close()


def plot_convergence(convergence_history: list, title: str,
                    save_path: Optional[str] = None,
                    xlabel: str = "Iteration", 
                    ylabel: str = "Best Objective Value",
                    algorithm: str = "COA"):
    """
    Plot convergence curve matching paper Figures 17-20 styling.
    
    Args:
        convergence_history: List of best objective values per iteration (can be cost or fitness)
        title: Plot title
        save_path: Path to save figure (optional)
        xlabel: X-axis label
        ylabel: Y-axis label
        algorithm: Algorithm name for line color (default: "COA")
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(AXES_STYLES['background_color'])
    ax.set_facecolor(AXES_STYLES['background_color'])
    
    # Get algorithm color for convergence line
    line_color = get_algorithm_color(algorithm)
    line_style = LINE_STYLES['convergence']
    
    iterations = np.arange(1, len(convergence_history) + 1)
    ax.plot(
        iterations, convergence_history,
        color=line_color,
        linewidth=line_style['linewidth'],
        alpha=line_style['alpha'],
        linestyle=line_style['linestyle'],
        marker=line_style['marker']
    )
    
    ax.set_xlabel(xlabel, fontsize=12, color=AXES_STYLES['axis_color'])
    ax.set_ylabel(ylabel, fontsize=12, color=AXES_STYLES['axis_color'])
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Configure grid
    ax.grid(
        True,
        color=AXES_STYLES['grid_color'],
        alpha=AXES_STYLES['grid_alpha'],
        linestyle=AXES_STYLES['grid_linestyle'],
        linewidth=AXES_STYLES['grid_linewidth']
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=AXES_STYLES['background_color'])
        print(f"Saved convergence plot to {save_path}")
    
    plt.close()


def plot_paper_style_bars_for_sweep(coverage_values: list, connectivity_values: list, 
                                    fitness_values: list, x_values: list,
                                    xlabel: str, title_prefix: str,
                                    save_path: Optional[str] = None):
    """
    Generate paper-style bar chart with 3 vertically stacked panels:
    (a) Coverage, (b) Connectivity, (c) Fitness
    
    Matches paper Figures 14-16 layout.
    
    Args:
        coverage_values: List of mean coverage counts for each x value
        connectivity_values: List of mean connectivity counts for each x value
        fitness_values: List of mean fitness values for each x value
        x_values: List of x-axis values (sweep parameter values)
        xlabel: X-axis label (e.g., "Number of Clients (n)")
        title_prefix: Prefix for title (e.g., "Impact of varying number of mesh clients on")
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.patch.set_facecolor(AXES_STYLES['background_color'])
    
    x_pos = np.arange(len(x_values))
    x_labels = [str(v) for v in x_values]
    
    # Panel (a): Coverage
    ax1 = axes[0]
    ax1.set_facecolor(AXES_STYLES['background_color'])
    ax1.bar(
        x_pos, coverage_values,
        color=get_algorithm_color('WOA'),  # Green
        edgecolor=BAR_STYLES['edgecolor'],
        linewidth=BAR_STYLES['edgewidth'],
        alpha=BAR_STYLES['alpha']
    )
    ax1.set_ylabel('Coverage Count', fontsize=12, color=AXES_STYLES['axis_color'])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.grid(True, axis='y', color=AXES_STYLES['grid_color'],
             alpha=AXES_STYLES['grid_alpha'], linestyle=AXES_STYLES['grid_linestyle'],
             linewidth=AXES_STYLES['grid_linewidth'])
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, 
             fontweight='bold', verticalalignment='top')
    
    # Panel (b): Connectivity
    ax2 = axes[1]
    ax2.set_facecolor(AXES_STYLES['background_color'])
    ax2.bar(
        x_pos, connectivity_values,
        color=get_algorithm_color('COA'),  # Blue
        edgecolor=BAR_STYLES['edgecolor'],
        linewidth=BAR_STYLES['edgewidth'],
        alpha=BAR_STYLES['alpha']
    )
    ax2.set_ylabel('Connectivity Count', fontsize=12, color=AXES_STYLES['axis_color'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.grid(True, axis='y', color=AXES_STYLES['grid_color'],
             alpha=AXES_STYLES['grid_alpha'], linestyle=AXES_STYLES['grid_linestyle'],
             linewidth=AXES_STYLES['grid_linewidth'])
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=14,
             fontweight='bold', verticalalignment='top')
    
    # Panel (c): Fitness
    ax3 = axes[2]
    ax3.set_facecolor(AXES_STYLES['background_color'])
    ax3.bar(
        x_pos, fitness_values,
        color=get_algorithm_color('FA'),  # Orange
        edgecolor=BAR_STYLES['edgecolor'],
        linewidth=BAR_STYLES['edgewidth'],
        alpha=BAR_STYLES['alpha']
    )
    ax3.set_xlabel(xlabel, fontsize=12, color=AXES_STYLES['axis_color'])
    ax3.set_ylabel('Fitness', fontsize=12, color=AXES_STYLES['axis_color'])
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels)
    ax3.grid(True, axis='y', color=AXES_STYLES['grid_color'],
             alpha=AXES_STYLES['grid_alpha'], linestyle=AXES_STYLES['grid_linestyle'],
             linewidth=AXES_STYLES['grid_linewidth'])
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=14,
             fontweight='bold', verticalalignment='top')
    
    # Set overall title
    full_title = f"{title_prefix}: (a) Coverage (b) Connectivity (c) Fitness"
    fig.suptitle(full_title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=AXES_STYLES['background_color'])
        print(f"Saved paper-style bar chart to {save_path}")
    
    plt.close()


def plot_convergence_paper_style(convergence_history: list, title: str,
                                 save_path: Optional[str] = None,
                                 algorithm: str = "COA"):
    """
    Plot convergence curve in paper style (Figures 17-20).
    Uses cost (objective = 1 - fitness) on y-axis.
    
    Args:
        convergence_history: List of cost values per iteration (from optimizer)
        title: Plot title
        save_path: Path to save figure (optional)
        algorithm: Algorithm name for line color (default: "COA")
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(AXES_STYLES['background_color'])
    ax.set_facecolor(AXES_STYLES['background_color'])
    
    # Get algorithm color for convergence line
    line_color = get_algorithm_color(algorithm)
    line_style = LINE_STYLES['convergence']
    
    iterations = np.arange(1, len(convergence_history) + 1)
    ax.plot(
        iterations, convergence_history,
        color=line_color,
        linewidth=line_style['linewidth'],
        alpha=line_style['alpha'],
        linestyle=line_style['linestyle'],
        marker=line_style['marker']
    )
    
    ax.set_xlabel('Iteration', fontsize=12, color=AXES_STYLES['axis_color'])
    ax.set_ylabel('Objective (cost = 1 - fitness)', fontsize=12, 
                  color=AXES_STYLES['axis_color'])
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Configure grid
    ax.grid(
        True,
        color=AXES_STYLES['grid_color'],
        alpha=AXES_STYLES['grid_alpha'],
        linestyle=AXES_STYLES['grid_linestyle'],
        linewidth=AXES_STYLES['grid_linewidth']
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=AXES_STYLES['background_color'])
        print(f"Saved convergence plot to {save_path}")
    
    plt.close()
