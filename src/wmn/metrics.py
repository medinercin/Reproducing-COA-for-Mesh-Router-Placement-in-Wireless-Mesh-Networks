"""
Metrics for WMN placement evaluation
"""
import numpy as np
from typing import Tuple


def compute_coverage(clients: np.ndarray, routers: np.ndarray, CR: float) -> Tuple[int, float]:
    """
    Compute coverage metric: number and ratio of covered clients
    
    A client is covered if it lies within CR distance of at least one router.
    
    Args:
        clients: Array of shape (n, 2) with client positions
        routers: Array of shape (m, 2) with router positions
        CR: Coverage radius
        
    Returns:
        Tuple of (coverage_count, coverage_ratio)
    """
    n = len(clients)
    if n == 0:
        return 0, 0.0
    
    # Compute distances from each client to each router
    # clients: (n, 2), routers: (m, 2)
    # distances: (n, m)
    distances = np.sqrt(((clients[:, np.newaxis, :] - routers[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # For each client, check if any router is within CR
    min_distances = np.min(distances, axis=1)  # (n,)
    covered = min_distances <= CR
    
    coverage_count = int(np.sum(covered))
    coverage_ratio = coverage_count / n
    
    return coverage_count, coverage_ratio


def compute_connectivity_router_only(routers: np.ndarray, comm_radius: float) -> Tuple[int, float]:
    """
    Compute connectivity metric using router-only graph
    
    Builds a graph where routers are nodes and edges exist if distance <= comm_radius.
    Returns the size of the largest connected component.
    
    Args:
        routers: Array of shape (m, 2) with router positions
        comm_radius: Communication radius for router-router links
        
    Returns:
        Tuple of (LCC_size, LCC_ratio) where ratio is normalized by m
    """
    m = len(routers)
    if m == 0:
        return 0, 0.0
    
    # Compute pairwise distances
    distances = np.sqrt(((routers[:, np.newaxis, :] - routers[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Build adjacency matrix
    adj_matrix = (distances <= comm_radius).astype(int)
    np.fill_diagonal(adj_matrix, 0)  # No self-loops
    
    # Find largest connected component using BFS/DFS
    visited = np.zeros(m, dtype=bool)
    max_component_size = 0
    
    for start in range(m):
        if visited[start]:
            continue
        
        # BFS from start
        component_size = 0
        queue = [start]
        visited[start] = True
        
        while queue:
            node = queue.pop(0)
            component_size += 1
            
            # Find neighbors
            neighbors = np.where((adj_matrix[node] > 0) & (~visited))[0]
            for neighbor in neighbors:
                visited[neighbor] = True
                queue.append(neighbor)
        
        max_component_size = max(max_component_size, component_size)
    
    lcc_ratio = max_component_size / m if m > 0 else 0.0
    return max_component_size, lcc_ratio


def compute_connectivity_combined(clients: np.ndarray, routers: np.ndarray, 
                                   CR: float, comm_radius: float) -> Tuple[int, float]:
    """
    Compute connectivity metric using combined graph (clients + routers)
    
    Builds a graph where:
    - Clients connect to routers within CR distance
    - Routers connect to routers within comm_radius distance
    Returns the size of the largest connected component.
    
    Args:
        clients: Array of shape (n, 2) with client positions
        routers: Array of shape (m, 2) with router positions
        CR: Coverage radius (client-router links)
        comm_radius: Communication radius (router-router links)
        
    Returns:
        Tuple of (LCC_size, LCC_ratio) where ratio is normalized by (m+n)
    """
    n = len(clients)
    m = len(routers)
    total_nodes = n + m
    
    if total_nodes == 0:
        return 0, 0.0
    
    # Build adjacency list representation
    # Nodes 0 to n-1 are clients, nodes n to n+m-1 are routers
    adj_list = [[] for _ in range(total_nodes)]
    
    # Client-router edges (if client within CR of router)
    if n > 0 and m > 0:
        client_router_distances = np.sqrt(
            ((clients[:, np.newaxis, :] - routers[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        for i in range(n):
            for j in range(m):
                if client_router_distances[i, j] <= CR:
                    router_idx = n + j
                    adj_list[i].append(router_idx)
                    adj_list[router_idx].append(i)
    
    # Router-router edges (if distance <= comm_radius)
    if m > 1:
        router_distances = np.sqrt(
            ((routers[:, np.newaxis, :] - routers[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        for i in range(m):
            for j in range(i + 1, m):
                if router_distances[i, j] <= comm_radius:
                    router_i_idx = n + i
                    router_j_idx = n + j
                    adj_list[router_i_idx].append(router_j_idx)
                    adj_list[router_j_idx].append(router_i_idx)
    
    # Find largest connected component using BFS
    visited = np.zeros(total_nodes, dtype=bool)
    max_component_size = 0
    
    for start in range(total_nodes):
        if visited[start]:
            continue
        
        # BFS from start
        component_size = 0
        queue = [start]
        visited[start] = True
        
        while queue:
            node = queue.pop(0)
            component_size += 1
            
            # Add unvisited neighbors
            for neighbor in adj_list[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        max_component_size = max(max_component_size, component_size)
    
    lcc_ratio = max_component_size / total_nodes if total_nodes > 0 else 0.0
    return max_component_size, lcc_ratio


def compute_fitness(clients: np.ndarray, routers: np.ndarray, CR: float, 
                    comm_radius: float, lambda_weight: float,
                    use_combined_connectivity: bool = True) -> Tuple[float, float, dict]:
    """
    Compute fitness function: weighted combination of coverage and connectivity
    
    f = lambda * coverage_ratio + (1 - lambda) * connectivity_ratio
    
    Args:
        clients: Array of shape (n, 2) with client positions
        routers: Array of shape (m, 2) with router positions
        CR: Coverage radius
        comm_radius: Communication radius
        lambda_weight: Weight for coverage (0-1)
        use_combined_connectivity: If True, use combined graph; else router-only
        
    Returns:
        Tuple of (fitness, cost, metrics_dict)
        - fitness: value in [0, 1], higher is better
        - cost: 1 - fitness, lower is better (for minimization)
        - metrics_dict: detailed metrics
    """
    # Compute coverage
    coverage_count, coverage_ratio = compute_coverage(clients, routers, CR)
    
    # Compute connectivity
    if use_combined_connectivity:
        connectivity_count, connectivity_ratio = compute_connectivity_combined(
            clients, routers, CR, comm_radius
        )
    else:
        connectivity_count, connectivity_ratio = compute_connectivity_router_only(
            routers, comm_radius
        )
    
    # Compute fitness
    fitness = lambda_weight * coverage_ratio + (1 - lambda_weight) * connectivity_ratio
    cost = 1.0 - fitness
    
    metrics = {
        'coverage_count': coverage_count,
        'coverage_ratio': coverage_ratio,
        'connectivity_count': connectivity_count,
        'connectivity_ratio': connectivity_ratio,
        'fitness': fitness,
        'cost': cost
    }
    
    return fitness, cost, metrics

