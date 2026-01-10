# Visual Styling Guide

This document describes the visual styling implementation that matches the reference academic paper figures exactly.

## Centralized Style Module

All visual styling is defined in `src/wmn/style.py` to ensure consistency across all plots.

### Color Palette

The implementation uses MATLAB R2014b+ "lines" color palette as reference:

- **Blue**: `#0072BD` - COA algorithm, routers, convergence lines
- **Orange**: `#D95319` - FA algorithm, fitness bars
- **Yellow**: `#EDB120` - GA algorithm
- **Purple**: `#7E2F8E` - PSO algorithm, router links
- **Green**: `#77AC30` - WOA algorithm, clients, COA coverage circles
- **Cyan**: `#4DBEEE` - BA algorithm
- **Red**: `#A2142F` - AVOA algorithm
- **Magenta**: `#ED0AD9` - SSA algorithm
- **Teal**: `#20B2AA` - AO algorithm
- **Brown**: `#8B4513` - CHIO algorithm, FA coverage circles
- **Black**: `#000000` - BES algorithm

### Placement Figures (Figures 3-13)

**Client Points:**
- Marker: `.` (dot)
- Size: 12
- Color: `#66AA33` (MATLAB-like green)
- No edge color
- Z-order: 3

**Router Points:**
- Marker: `o` (circle)
- Size: 40
- Face color: `#0072BD` (MATLAB default blue)
- No edge color
- Z-order: 4 (top layer)

**Coverage Circles:**
- Translucent filled circles
- Alpha: 0.35
- Algorithm-specific colors:
  - COA: Green (`#77AC30`)
  - FA: Brown (`#8B4513`)
  - GA: Orange (`#EDB120`)
  - PSO: Magenta (`#ED0AD9`)
  - WOA: Yellow (`#EDB120`)
  - BA: Red (`#A2142F`)
  - BES: Gray (`#808080`)
  - AO: Olive (`#808000`)
  - AVOA: Light pink (`#FFB6C1`)
  - SSA: Purple (`#7E2F8E`)
  - CHIO: White/light gray (`#F5F5F5`)
- Edge colors are darker versions of face colors
- Z-order: 1 (bottom layer)

**Router Links:**
- Color: `#7E2F8E` (MATLAB default purple)
- Line width: 1.0
- Alpha: 0.6
- Z-order: 2

**Axes:**
- Limits: x=[0, W], y=[0, H] (exact, no padding)
- Aspect ratio: Equal
- Grid: Light gray dashed lines

**Legend Order:**
1. Client
2. Router
3. Coverage area
4. Link

### Bar Charts (Figures 14-16)

**Bar Colors:**
- Coverage Count: Green (`#77AC30` - WOA color)
- Connectivity Count: Blue (`#0072BD` - COA color)
- Fitness: Orange (`#D95319` - FA color)

**Bar Styling:**
- Edge color: `#2C3E50` (dark gray)
- Edge width: 1.0
- Alpha: 0.85
- Width ratio: 0.8

**Grid:**
- Color: `#E0E0E0` (light gray)
- Alpha: 0.3
- Style: Dashed
- Only on Y-axis

### Convergence Plots (Figures 17-20)

**Line Styling:**
- Color: Algorithm-specific (COA uses blue `#0072BD`)
- Line width: 2.0
- Alpha: 1.0
- No markers

**Grid:**
- Color: `#E0E0E0` (light gray)
- Alpha: 0.3
- Style: Dashed

## Paper Figure Mapping

| Function | Paper Figures | Description |
|----------|---------------|-------------|
| `plot_placement()` | Figures 3-13 | Placement visualization (one per algorithm) |
| `plot_bar_chart()` | Figures 14-16 | Bar charts for parameter sweeps |
| `plot_convergence()` | Figures 17-20 | Convergence curves per instance |

## Usage

All plotting functions accept an `algorithm` parameter to specify which algorithm's colors to use:

```python
from src.wmn.plotting import plot_placement

plot_placement(
    clients, routers, CR, comm_radius, W, H,
    algorithm='COA',  # Uses COA-specific colors
    save_path='fig_placement_coa.png'
)
```

## Customization

To update colors to match paper exactly:

1. Extract HEX color codes from paper figures using a color picker tool
2. Update `src/wmn/style.py`:
   - `ALGORITHM_COLORS` for bar chart colors
   - `COVERAGE_CIRCLE_COLORS` for placement figure coverage circles
   - `MARKER_STYLES` for client/router markers
   - `LINE_STYLES` for links and convergence lines

All colors use explicit HEX codes - no named colors or automatic color cycles.

