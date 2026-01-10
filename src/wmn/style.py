"""
Centralized style definitions matching the reference paper figures exactly.

Paper Figure Mapping:
- Placement plots: Figures 3-13 (one per algorithm)
- Bar charts: Figures 14-16 (grouped bars for different algorithms)
- Convergence plots: Figures 17-20 (line plots per instance)

All colors use MATLAB R2014b+ "lines" palette as reference for consistency.
"""

# MATLAB R2014b+ default "lines" color palette (exact HEX codes)
MATLAB_COLORS = {
    'blue': '#0072BD',      # (0.0000, 0.4470, 0.7410)
    'orange': '#D95319',    # (0.8500, 0.3250, 0.0980)
    'yellow': '#EDB120',    # (0.9290, 0.6940, 0.1250)
    'purple': '#7E2F8E',    # (0.4940, 0.1840, 0.5560)
    'green': '#77AC30',     # (0.4660, 0.6740, 0.1880)
    'cyan': '#4DBEEE',      # (0.3010, 0.7450, 0.9330)
    'red': '#A2142F',       # (0.6350, 0.0780, 0.1840)
    'magenta': '#ED0AD9',   # Extended palette
    'teal': '#20B2AA',      # Extended palette
    'brown': '#8B4513',     # Extended palette
    'olive': '#808000',     # Extended palette
    'pink': '#FFB6C1',      # Light pink
    'gray': '#808080',      # Gray
    'black': '#000000',     # Black
    'white': '#FFFFFF',      # White
}

# Algorithm color palette for bar charts (Figures 14-16)
# Matches paper legend order and colors
ALGORITHM_COLORS = {
    'COA': MATLAB_COLORS['blue'],      # Blue
    'FA': MATLAB_COLORS['orange'],     # Orange (Firefly Algorithm)
    'GA': MATLAB_COLORS['yellow'],     # Yellow (Genetic Algorithm)
    'PSO': MATLAB_COLORS['purple'],    # Purple (Particle Swarm Optimization)
    'WOA': MATLAB_COLORS['green'],     # Green (Whale Optimization Algorithm)
    'BA': MATLAB_COLORS['cyan'],       # Cyan (Bat Algorithm)
    'AVOA': MATLAB_COLORS['red'],      # Red (African Vulture Optimization)
    'AO': MATLAB_COLORS['teal'],       # Teal (Aquila Optimizer)
    'BES': MATLAB_COLORS['black'],     # Black (Bald Eagle Search)
    'CHIO': MATLAB_COLORS['brown'],   # Brown (Coronavirus Herd Immunity Optimizer)
    'SSA': MATLAB_COLORS['magenta'],   # Magenta (Salp Swarm Algorithm)
}

# Coverage circle colors for placement figures (Figures 3-13)
# Translucent filled circles with alpha ~0.3-0.4
COVERAGE_CIRCLE_COLORS = {
    'COA': '#77AC30',       # Green translucent
    'FA': '#8B4513',        # Brown translucent
    'GA': '#EDB120',        # Orange translucent
    'PSO': '#ED0AD9',       # Magenta/pink translucent
    'WOA': '#EDB120',       # Yellow translucent
    'BA': '#A2142F',        # Red translucent
    'BES': '#808080',       # Gray translucent
    'AO': '#808000',        # Olive/gold translucent
    'AVOA': '#FFB6C1',      # Light pink translucent
    'SSA': '#7E2F8E',       # Purple translucent
    'CHIO': '#F5F5F5',      # White/very light gray with dark edge
}

# Coverage circle edge colors (darker version of face color)
COVERAGE_CIRCLE_EDGES = {
    'COA': '#4A6B1F',       # Darker green
    'FA': '#5C2E0A',       # Darker brown
    'GA': '#9B7A0C',       # Darker orange
    'PSO': '#9B0691',      # Darker magenta
    'WOA': '#9B7A0C',      # Darker yellow
    'BA': '#6B0A12',       # Darker red
    'BES': '#505050',      # Darker gray
    'AO': '#555500',       # Darker olive
    'AVOA': '#CC8F9E',     # Darker pink
    'SSA': '#4F1C58',      # Darker purple
    'CHIO': '#CCCCCC',     # Medium gray edge
}

# Marker and line styles matching paper exactly
MARKER_STYLES = {
    'client': {
        'marker': '.',
        'size': 12,
        'color': '#66AA33',      # MATLAB-like green
        'edgecolor': 'none',
        'alpha': 1.0,
        'zorder': 3,
    },
    'router': {
        'marker': 'o',
        'size': 40,
        'facecolor': '#0072BD',  # MATLAB default blue
        'edgecolor': 'none',
        'alpha': 1.0,
        'zorder': 4,
    },
}

LINE_STYLES = {
    'router_link': {
        'color': '#7E2F8E',      # MATLAB default purple
        'linewidth': 1.0,
        'alpha': 0.6,
        'linestyle': '-',
        'zorder': 2,
    },
    'coverage_circle': {
        'linewidth': 1.0,
        'alpha': 0.35,           # Translucent fill
        'linestyle': '-',
        'zorder': 1,
    },
    'convergence': {
        'color': '#0072BD',       # Blue for COA convergence
        'linewidth': 2.0,
        'alpha': 1.0,
        'linestyle': '-',
        'marker': None,
    },
}

# Bar chart styling
BAR_STYLES = {
    'edgecolor': '#2C3E50',      # Dark gray edges
    'edgewidth': 1.0,
    'alpha': 0.85,
    'width_ratio': 0.8,          # Bar width relative to group spacing
}

# Grid and axes styling
AXES_STYLES = {
    'grid_color': '#E0E0E0',     # Light gray
    'grid_alpha': 0.3,
    'grid_linestyle': '--',
    'grid_linewidth': 0.5,
    'axis_color': '#333333',     # Dark gray
    'background_color': '#FFFFFF', # White
}

# Legend order (must match paper exactly)
LEGEND_ORDER = ['Client', 'Router', 'Coverage area', 'Link']

# Coverage circle alpha (translucency)
COVERAGE_ALPHA = 0.35

def get_algorithm_color(algorithm: str) -> str:
    """Get color for algorithm in bar charts"""
    return ALGORITHM_COLORS.get(algorithm.upper(), MATLAB_COLORS['blue'])

def get_coverage_color(algorithm: str) -> str:
    """Get coverage circle face color for placement figures"""
    return COVERAGE_CIRCLE_COLORS.get(algorithm.upper(), COVERAGE_CIRCLE_COLORS['COA'])

def get_coverage_edge_color(algorithm: str) -> str:
    """Get coverage circle edge color for placement figures"""
    return COVERAGE_CIRCLE_EDGES.get(algorithm.upper(), COVERAGE_CIRCLE_EDGES['COA'])

