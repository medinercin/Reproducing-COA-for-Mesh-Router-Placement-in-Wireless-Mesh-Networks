"""
Geometric utilities for WMN placement
"""
import numpy as np
from typing import Tuple, Optional


def generate_clients(n: int, W: float, H: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate n clients uniformly distributed in area [0, W] x [0, H]
    
    Args:
        n: Number of clients
        W: Area width
        H: Area height
        seed: Random seed
        
    Returns:
        Array of shape (n, 2) with client positions [x, y]
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    clients = np.column_stack([
        rng.uniform(0, W, n),
        rng.uniform(0, H, n)
    ])
    return clients


def generate_router_positions(m: int, W: float, H: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate m router positions uniformly distributed in area [0, W] x [0, H]
    
    Args:
        m: Number of routers
        W: Area width
        H: Area height
        seed: Random seed
        
    Returns:
        Array of shape (m, 2) with router positions [x, y]
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    routers = np.column_stack([
        rng.uniform(0, W, m),
        rng.uniform(0, H, m)
    ])
    return routers


def solution_to_positions(solution: np.ndarray) -> np.ndarray:
    """
    Convert solution vector [x1, y1, x2, y2, ..., xm, ym] to position array
    
    Args:
        solution: Array of shape (2*m,) with router coordinates
        
    Returns:
        Array of shape (m, 2) with router positions
    """
    m = len(solution) // 2
    return solution.reshape(m, 2)


def positions_to_solution(positions: np.ndarray) -> np.ndarray:
    """
    Convert position array to solution vector
    
    Args:
        positions: Array of shape (m, 2) with router positions
        
    Returns:
        Array of shape (2*m,) with router coordinates
    """
    return positions.flatten()


def clip_to_bounds(solution: np.ndarray, W: float, H: float) -> np.ndarray:
    """
    Clip solution coordinates to area bounds [0, W] x [0, H]
    
    Args:
        solution: Array of shape (2*m,) with router coordinates
        W: Area width
        H: Area height
        
    Returns:
        Clipped solution array
    """
    m = len(solution) // 2
    positions = solution_to_positions(solution)
    positions[:, 0] = np.clip(positions[:, 0], 0, W)
    positions[:, 1] = np.clip(positions[:, 1], 0, H)
    return positions_to_solution(positions)


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points
    
    Args:
        p1: Point 1, shape (2,)
        p2: Point 2, shape (2,)
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(p1 - p2)


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between positions
    
    Args:
        positions: Array of shape (n, 2) with positions
        
    Returns:
        Distance matrix of shape (n, n)
    """
    return np.sqrt(((positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2).sum(axis=2))

