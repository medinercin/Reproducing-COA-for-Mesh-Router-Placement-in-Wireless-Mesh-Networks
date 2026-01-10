"""
Configuration management for WMN experiments
- Clean RNG behavior (no global seeding side-effects)
- Paper-aligned migration probability:
  Eq(6): Pa = 0.005 * Cc^2  (clamped to [0,1])
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class WMNConfig:
    """Configuration for WMN experiments"""

    # Area dimensions
    W: float = 2000.0  # Width
    H: float = 2000.0  # Height

    # Network parameters
    m: int = 20          # Number of routers
    n: int = 50          # Number of clients
    CR: float = 200.0    # Coverage radius
    comm_radius: Optional[float] = None  # Communication radius (default: 2*CR)

    # Objective function
    lambda_weight: float = 0.5  # Weight for coverage vs connectivity

    # Experiment parameters
    iterations: int = 1000
    runs: int = 50               # Independent runs for statistics
    seed: Optional[int] = None   # Base seed (do NOT globally seed in __post_init__)

    # COA algorithm parameters
    pack_count: int = 10         # Gp: number of packs
    coyotes_per_pack: int = 10   # Cc: coyotes per pack

    # Migration probability:
    # If None => compute from Eq(6): Pa = 0.005 * Cc^2 (clamped to [0,1])
    migration_prob: Optional[float] = None

    # Fixed clients flag
    fix_clients: bool = True  # Use same client positions across runs

    def __post_init__(self):
        """Set derived defaults after initialization."""
        if self.comm_radius is None:
            self.comm_radius = 2.0 * float(self.CR)

        # Paper-aligned migration probability (Eq.6)
        if self.migration_prob is None:
            pa = 0.005 * (float(self.coyotes_per_pack) ** 2)
            # Clamp to [0, 1] (probability)
            self.migration_prob = max(0.0, min(1.0, pa))
        else:
            # Ensure it's a valid probability
            self.migration_prob = max(0.0, min(1.0, float(self.migration_prob)))

        # IMPORTANT:
        # Do NOT set global random seeds here.
        # Seeding should be handled at call sites:
        # - generate_clients(seed=...)
        # - optimizer.optimize(seed=...)
        # This avoids side-effects and is safer for multiprocessing.

    @property
    def total_population(self) -> int:
        """Total population size = Gp * Cc"""
        return int(self.pack_count) * int(self.coyotes_per_pack)
