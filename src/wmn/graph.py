"""
Graph utilities for WMN (using networkx for visualization/analysis)
"""
import networkx as nx
import numpy as np
from typing import Tuple, Optional


def build_router_graph(routers: np.ndarray, comm_radius: float) -> nx.Graph:
    """
    Build networkx graph of router network
    
    Args:
        routers: Array of shape (m, 2) with router positions
        comm_radius: Communication radius
        
    Returns:
        NetworkX graph with routers as nodes
    """
    G = nx.Graph()
    m = len(routers)
    
    # Add nodes
    for i in range(m):
        G.add_node(i, pos=routers[i])
    
    # Add edges
    distances = np.sqrt(((routers[:, np.newaxis, :] - routers[np.newaxis, :, :]) ** 2).sum(axis=2))
    for i in range(m):
        for j in range(i + 1, m):
            if distances[i, j] <= comm_radius:
                G.add_edge(i, j, weight=distances[i, j])
    
    return G


def build_combined_graph(clients: np.ndarray, routers: np.ndarray, 
                        CR: float, comm_radius: float) -> nx.Graph:
    """
    Build networkx graph of combined network (clients + routers)
    
    Args:
        clients: Array of shape (n, 2) with client positions
        routers: Array of shape (m, 2) with router positions
        CR: Coverage radius (client-router links)
        comm_radius: Communication radius (router-router links)
        
    Returns:
        NetworkX graph
    """
    G = nx.Graph()
    n = len(clients)
    m = len(routers)
    
    # Add client nodes (0 to n-1)
    for i in range(n):
        G.add_node(i, pos=clients[i], type='client')
    
    # Add router nodes (n to n+m-1)
    for i in range(m):
        G.add_node(n + i, pos=routers[i], type='router')
    
    # Add client-router edges
    if n > 0 and m > 0:
        client_router_distances = np.sqrt(
            ((clients[:, np.newaxis, :] - routers[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        for i in range(n):
            for j in range(m):
                if client_router_distances[i, j] <= CR:
                    G.add_edge(i, n + j, weight=client_router_distances[i, j])
    
    # Add router-router edges
    if m > 1:
        router_distances = np.sqrt(
            ((routers[:, np.newaxis, :] - routers[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        for i in range(m):
            for j in range(i + 1, m):
                if router_distances[i, j] <= comm_radius:
                    G.add_edge(n + i, n + j, weight=router_distances[i, j])
    
    return G


def get_largest_connected_component(G: nx.Graph) -> nx.Graph:
    """
    Get the largest connected component of a graph
    
    Args:
        G: NetworkX graph
        
    Returns:
        Subgraph of largest connected component
    """
    if len(G) == 0:
        return G
    
    components = list(nx.connected_components(G))
    if not components:
        return G.subgraph([])
    
    largest_component = max(components, key=len)
    return G.subgraph(largest_component)

