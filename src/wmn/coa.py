"""
Coyote Optimization Algorithm (COA) implementation for WMN router placement
Aligned with the paper "Solving the Mesh Router Nodes Placement in WMNs Using COA"

Implements key equations/steps:
- Eq(6): Pa from config (computed in WMNConfig as 0.005 * Cc^2)
- Eq(7): alpha = best coyote in pack (min cost)
- Eq(8): cultural tendency = median per dimension (ranked median)
- Eq(9): social update uses (alpha - SC_cr1) and (cult - SC_cr2)
- Eq(10)-(11): accept if improved (minimization on cost)
- Eq(12)-(14): pup generation with Ps=1/D and Pz=(1-Ps)/2, scatter/association
- Age update per iteration; pup starts at age=0
"""

import numpy as np
from typing import Callable, Tuple, Optional, List, Dict
from .config import WMNConfig
from .geometry import clip_to_bounds


class COA:
    """
    COA optimizer for continuous domain.
    Solution dimension D = 2*m  (x,y for each router).
    Objective returns COST to minimize (your compute_fitness returns cost = 1 - fitness).
    """

    def __init__(self, config: WMNConfig, objective_func: Callable[[np.ndarray], float]):
        self.config = config
        self.objective_func = objective_func

        # COA parameters
        self.Gp = int(config.pack_count)           # number of packs
        self.Cc = int(config.coyotes_per_pack)     # coyotes per pack
        self.Pa = float(config.migration_prob)     # migration probability (Eq6 via config)
        self.iterations = int(config.iterations)

        # Problem dimension
        self.dim = int(2 * config.m)

        # State
        self.packs: List[List[Dict]] = []
        self.global_best_solution: Optional[np.ndarray] = None
        self.global_best_cost: float = float("inf")
        self.convergence_history: List[float] = []

        self.rng: Optional[np.random.RandomState] = None

    # ----------------------------
    # Initialization
    # ----------------------------
    def initialize_population(self, seed: Optional[int] = None) -> None:
        """
        Initialize coyotes randomly in bounds and evaluate cost.
        """
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        self.packs = []
        self.global_best_solution = None
        self.global_best_cost = float("inf")
        self.convergence_history = []

        for _ in range(self.Gp):
            pack: List[Dict] = []
            for _ in range(self.Cc):
                sol = np.zeros(self.dim, dtype=float)
                sol[0::2] = self.rng.uniform(0.0, self.config.W, size=self.dim // 2)
                sol[1::2] = self.rng.uniform(0.0, self.config.H, size=self.dim // 2)
                sol = clip_to_bounds(sol, self.config.W, self.config.H)

                cost = float(self.objective_func(sol))

                # Age is used by the flowchart; initialize diverse ages
                age = int(self.rng.randint(0, max(1, self.Cc)))
                pack.append({"solution": sol, "cost": cost, "age": age})

            pack.sort(key=lambda x: x["cost"])
            self.packs.append(pack)

            if pack[0]["cost"] < self.global_best_cost:
                self.global_best_cost = pack[0]["cost"]
                self.global_best_solution = pack[0]["solution"].copy()

    # ----------------------------
    # Eq(7) alpha, Eq(8) cultural tendency
    # ----------------------------
    @staticmethod
    def _alpha(pack: List[Dict]) -> Dict:
        return pack[0]  # best (min cost)

    @staticmethod
    def _cultural_tendency(pack: List[Dict]) -> np.ndarray:
        """
        Median per dimension (ranked median).
        """
        sols = np.array([c["solution"] for c in pack], dtype=float)  # (Cc, D)
        sorted_sols = np.sort(sols, axis=0)
        mid = sorted_sols.shape[0] // 2
        if sorted_sols.shape[0] % 2 == 1:
            return sorted_sols[mid]
        return 0.5 * (sorted_sols[mid - 1] + sorted_sols[mid])

    # ----------------------------
    # Eq(9) social behavior update + Eq(10)-(11) accept
    # ----------------------------
    def update_coyote(self, pack: List[Dict], coyote_idx: int) -> Dict:
        coyote = pack[coyote_idx]
        alpha = self._alpha(pack)
        cult = self._cultural_tendency(pack)

        # Choose two random coyotes (cr1, cr2) distinct from current
        candidates = [i for i in range(len(pack)) if i != coyote_idx]
        if len(candidates) >= 2:
            cr1_idx, cr2_idx = self.rng.choice(candidates, size=2, replace=False)
        elif len(candidates) == 1:
            cr1_idx = cr2_idx = candidates[0]
        else:
            cr1_idx = cr2_idx = coyote_idx

        cr1 = pack[cr1_idx]
        cr2 = pack[cr2_idx]

        # Eq(9): newSC = SC + r1*(alpha - SC_cr1) + r2*(cult - SC_cr2)
        r1 = float(self.rng.rand())
        r2 = float(self.rng.rand())

        new_sol = (
            coyote["solution"]
            + r1 * (alpha["solution"] - cr1["solution"])
            + r2 * (cult - cr2["solution"])
        )
        new_sol = clip_to_bounds(new_sol, self.config.W, self.config.H)
        new_cost = float(self.objective_func(new_sol))

        # Eq(10)-(11): accept if improved
        if new_cost < coyote["cost"]:
            return {"solution": new_sol, "cost": new_cost, "age": int(coyote["age"])}
        return {"solution": coyote["solution"].copy(), "cost": float(coyote["cost"]), "age": int(coyote["age"])}

    # ----------------------------
    # Eq(12)-(14) Birth & Death
    # ----------------------------
    def birth_death(self, pack: List[Dict]) -> List[Dict]:
        D = self.dim
        Ps = 1.0 / float(D)          # Eq(13)
        Pz = (1.0 - Ps) / 2.0        # Eq(14)

        # Two parents uniformly
        p1_idx, p2_idx = self.rng.choice(len(pack), size=2, replace=False)
        p1 = pack[p1_idx]
        p2 = pack[p2_idx]

        pup = np.zeros(D, dtype=float)

        # Eq(12) per-dimension gene rule
        for j in range(D):
            r = float(self.rng.rand())
            if r < Ps:
                # scatter: random within bounds
                if j % 2 == 0:  # x
                    pup[j] = float(self.rng.uniform(0.0, self.config.W))
                else:           # y
                    pup[j] = float(self.rng.uniform(0.0, self.config.H))
            else:
                # association: choose parent genes with probabilities ~Pz, Pz, remaining
                r2 = float(self.rng.rand())
                if r2 < Pz:
                    pup[j] = float(p1["solution"][j])
                elif r2 < 2.0 * Pz:
                    pup[j] = float(p2["solution"][j])
                else:
                    pup[j] = float(p1["solution"][j] if self.rng.rand() < 0.5 else p2["solution"][j])

        pup = clip_to_bounds(pup, self.config.W, self.config.H)
        pup_cost = float(self.objective_func(pup))

        # Replacement policy:
        # Prefer replacing worst if pup better; otherwise consider oldest if pup better than oldest.
        worst_idx = int(np.argmax([c["cost"] for c in pack]))
        oldest_idx = int(np.argmax([c["age"] for c in pack]))

        replaced = False
        if pup_cost < pack[worst_idx]["cost"]:
            pack[worst_idx] = {"solution": pup, "cost": pup_cost, "age": 0}
            replaced = True
        elif pup_cost < pack[oldest_idx]["cost"]:
            pack[oldest_idx] = {"solution": pup, "cost": pup_cost, "age": 0}
            replaced = True

        if replaced:
            pack.sort(key=lambda x: x["cost"])

        return pack

    # ----------------------------
    # Migration among packs (Eq6 via Pa)
    # ----------------------------
    def migrate(self) -> None:
        if self.Gp < 2:
            return
        if self.rng.rand() > self.Pa:
            return

        p1, p2 = self.rng.choice(self.Gp, size=2, replace=False)
        pack1 = self.packs[p1]
        pack2 = self.packs[p2]

        i = int(self.rng.randint(0, len(pack1)))
        j = int(self.rng.randint(0, len(pack2)))

        pack1[i], pack2[j] = pack2[j], pack1[i]

        pack1.sort(key=lambda x: x["cost"])
        pack2.sort(key=lambda x: x["cost"])

    # ----------------------------
    # Main loop
    # ----------------------------
    def optimize(self, seed: Optional[int] = None) -> Tuple[np.ndarray, float, List[float]]:
        self.initialize_population(seed=seed)

        for _ in range(self.iterations):
            for p in range(len(self.packs)):
                pack = self.packs[p]

                # Update coyotes except alpha (common stable variant; paper updates packs iteratively)
                for idx in range(1, len(pack)):
                    pack[idx] = self.update_coyote(pack, idx)

                pack.sort(key=lambda x: x["cost"])

                # Birth & death
                pack = self.birth_death(pack)

                # Age update (flowchart)
                for c in pack:
                    c["age"] = int(c["age"]) + 1

                self.packs[p] = pack

            # Migration
            self.migrate()

            # Global best
            for pack in self.packs:
                if pack[0]["cost"] < self.global_best_cost:
                    self.global_best_cost = pack[0]["cost"]
                    self.global_best_solution = pack[0]["solution"].copy()

            self.convergence_history.append(float(self.global_best_cost))

        return self.global_best_solution, float(self.global_best_cost), self.convergence_history.copy()


class Optimizer:
    """
    Abstract base class for optimizers (for future extension)
    """
    def optimize(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement optimize()")


class COAOptimizer(Optimizer):
    """
    COA optimizer wrapper (keeps your existing API)
    """
    def __init__(self, config: WMNConfig, objective_func: Callable[[np.ndarray], float]):
        self.coa = COA(config, objective_func)
        self.config = config

    def optimize(self, seed: Optional[int] = None) -> Tuple[np.ndarray, float, List[float]]:
        return self.coa.optimize(seed=seed)
