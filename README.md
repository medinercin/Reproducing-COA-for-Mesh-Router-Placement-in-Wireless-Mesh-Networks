# COA-based Wireless Mesh Network Router Placement

This project implements the Coyote Optimization Algorithm (COA) for solving the mesh router nodes placement problem in Wireless Mesh Networks (WMN), as described in the paper:

**"Solving the Mesh Router Nodes Placement in Wireless Mesh Networks Using Coyote Optimization Algorithm (COA)"**

## Project Structure

```
BBL512E_COA_WMN/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/wmn/                  # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── geometry.py           # Geometric utilities
│   ├── metrics.py            # Coverage, connectivity, fitness metrics
│   ├── graph.py              # Graph building utilities
│   ├── plotting.py           # Plotting functions
│   ├── coa.py                # COA algorithm implementation
│   ├── experiments.py        # Experiment execution
│   └── tables.py             # Table generation
├── scripts/                  # Experiment scripts
│   ├── run_placement.py      # Single placement plot (Fig 3)
│   ├── run_clients_sweep.py  # Clients sweep (Table 7, Fig 14)
│   ├── run_routers_sweep.py  # Routers sweep (Table 8, Fig 15)
│   ├── run_radius_sweep.py   # Radius sweep (Table 9, Fig 16)
│   ├── run_convergence.py    # Convergence analysis (Table 11, Figs 17-20)
│   └── run_all.py            # Run all experiments
└── results/                  # Output directory
    ├── figures/              # Generated plots
    ├── tables/               # Generated CSV tables
    └── logs/                 # Log files
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

The project uses a `WMNConfig` dataclass for configuration. Key parameters:

- **Area dimensions**: `W`, `H` (default: 2000x2000)
- **Network parameters**: 
  - `m`: Number of routers (default: 20)
  - `n`: Number of clients (default: 50)
  - `CR`: Coverage radius (default: 200)
  - `comm_radius`: Communication radius for router-router links (default: 2*CR)
- **Objective function**: `lambda_weight` (default: 0.5)
- **COA parameters**:
  - `pack_count` (Gp): Number of packs (default: 10)
  - `coyotes_per_pack` (Cc): Coyotes per pack (default: 10)
  - `migration_prob` (Pa): Migration probability (default: 0.1)
- **Optimization**: `iterations` (default: 1000), `runs` (default: 50)
- **Random seed**: `seed` (default: None)

## Running Experiments

### Single Placement Plot (Figure 3)

Generate a single placement visualization:

```bash
python scripts/run_placement.py
```

Output: `results/figures/fig_placement_coa.png`

### Clients Sweep (Table 7, Figure 14)

Sweep number of clients (n) from 50 to 300:

```bash
python scripts/run_clients_sweep.py
```

Outputs:
- `results/tables/table7_clients_sweep.csv`
- `results/figures/fig14_clients_sweep.png`

### Routers Sweep (Table 8, Figure 15)

Sweep number of routers (m) from 5 to 40:

```bash
python scripts/run_routers_sweep.py
```

Outputs:
- `results/tables/table8_routers_sweep.csv`
- `results/figures/fig15_routers_sweep.png`

### Radius Sweep (Table 9, Figure 16)

Sweep coverage radius (CR) from 50 to 400:

```bash
python scripts/run_radius_sweep.py
```

Outputs:
- `results/tables/table9_radius_sweep.csv`
- `results/figures/fig16_radius_sweep.png`

### Convergence Analysis (Table 11, Figures 17-20)

Run convergence analysis on 4 instances:

```bash
python scripts/run_convergence.py
```

Outputs:
- `results/figures/fig17_instance1.png`
- `results/figures/fig18_instance2.png`
- `results/figures/fig19_instance3.png`
- `results/figures/fig20_instance4.png`
- `results/tables/table11_convergence.csv`

### Run All Experiments

Execute all experiments sequentially:

```bash
python scripts/run_all.py
```

This will run all scripts and generate all figures and tables.

## Metrics

The implementation computes three main metrics:

### 1. Coverage
- **Coverage Count**: Number of clients within CR distance of at least one router
- **Coverage Ratio**: Coverage count / total clients

### 2. Connectivity
- **Connectivity Count**: Size of largest connected component in the combined graph (clients + routers)
- **Connectivity Ratio**: Connectivity count / (m + n)

The combined graph includes:
- Client-router edges: if client is within CR of router
- Router-router edges: if routers are within comm_radius

### 3. Fitness
- **Fitness**: `lambda * coverage_ratio + (1 - lambda) * connectivity_ratio`
- **Cost**: `1 - fitness` (minimized by COA)

## COA Algorithm

The Coyote Optimization Algorithm (COA) is implemented with:

- **Pack structure**: Gp packs with Cc coyotes each
- **Update rule**: Each coyote updates based on:
  - Alpha (best coyote in pack)
  - Cultural tendency (median of pack)
  - Random differences with other coyotes
- **Birth/death**: Generate pups and replace worst coyotes
- **Migration**: Swap coyotes between packs with probability Pa

## Output Mapping to Paper

| Output File | Paper Reference |
|------------|----------------|
| `fig_placement_coa.png` | Figure 3 (placement visualization) |
| `table7_clients_sweep.csv` | Table 7 (clients sweep results) |
| `fig14_clients_sweep.png` | Figure 14 (clients sweep bar chart) |
| `table8_routers_sweep.csv` | Table 8 (routers sweep results) |
| `fig15_routers_sweep.png` | Figure 15 (routers sweep bar chart) |
| `table9_radius_sweep.csv` | Table 9 (radius sweep results) |
| `fig16_radius_sweep.png` | Figure 16 (radius sweep bar chart) |
| `fig17_instance1.png` | Figure 17 (convergence Instance 1) |
| `fig18_instance2.png` | Figure 18 (convergence Instance 2) |
| `fig19_instance3.png` | Figure 19 (convergence Instance 3) |
| `fig20_instance4.png` | Figure 20 (convergence Instance 4) |
| `table11_convergence.csv` | Table 11 (convergence analysis) |

## Quick Test

For quick testing with fewer runs, modify the `runs` parameter in the scripts:

```python
config = WMNConfig(
    ...
    runs=5,  # Reduced from 50 for quick testing
    ...
)
```

## Dependencies

- numpy
- scipy
- matplotlib
- networkx
- pandas
- pyyaml

## Notes

- All experiments use deterministic seeds for reproducibility
- Clients are fixed across runs within a scenario (when `fix_clients=True`)
- Results are saved in `results/` directory with organized subdirectories
- Logs are written to `results/logs/` for debugging

## License

This implementation is for educational and research purposes.

