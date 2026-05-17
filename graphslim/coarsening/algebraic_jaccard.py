import copy
import numpy as np
import scipy as sp
import torch
from pygsp import graphs
from torch_geometric.utils import to_dense_adj
from graphslim.coarsening.utils import (
    contract_variation_edges,
    contract_variation_linear,
    get_proximity_measure,
    matching_optimal,
    matching_greedy,
    get_coarsening_matrix,
    coarsen_matrix,
    coarsen_vector,
    zero_diag
)
from graphslim.dataset.convertor import pyg2gsp, csr2ei, ei2csr
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation import *
from graphslim.utils import one_hot, to_tensor
from graphslim.coarsening.coarsening_base import Coarsen


class AlgebraicJC(Coarsen):
    """Implementation of Algebraic Jeopardy Coarsening (AJ) for graph reduction.
    
    This method performs graph coarsening by iteratively contracting nodes based on
    algebraic proximity measures while preserving key structural properties.
    """
    
    def __init__(self, setting, data, args):
        """Initialize the coarsening method.
        
        Args:
            setting: Configuration settings
            data: Input graph data
            args: Additional arguments including reduction parameters
        """
        super().__init__(setting, data, args)
        args.method = "algebraic_jaccard"
        
    def coarsen(self, G):
        """Perform the coarsening operation on graph G.
        
        Args:
            G: Input graph (pygsp graph object)
            
        Returns:
            Tuple containing:
            - C: Coarsening matrix
            - Gc: Coarsened graph
            - Call: List of coarsening matrices at each level
            - Gall: List of graphs at each coarsening level
        """
        # Configuration parameters
        K = 10  # Number of eigenvectors to consider
        max_levels = 10  # Maximum coarsening levels
        max_level_r = 0.99  # Maximum reduction per level
        r = np.clip(self.args.reduction_rate, 0, 0.999)
        method = "algebraic_jaccard"
        algorithm = self.args.coarsen_strategy
        
        # Initialize tracking variables
        G0 = G
        N = G.N
        n, n_target = N, np.ceil(r * N)  # Current and target sizes
        C = sp.sparse.eye(N, format="csc")  # Initial coarsening matrix
        Gc = G
        
        # Storage for all levels
        Call, Gall = [], [G]
        
        # Multi-level coarsening loop
        for level in range(1, max_levels + 1):
            G = Gc
            r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)
            
            # Compute node proximity measures
            weights = get_proximity_measure(G, method, K=K)
            
            # Select matching strategy
            if algorithm == "optimal":
                weights = -weights  # Optimal matching minimizes weight
                if "rss" not in method:
                    weights -= min(weights)
                coarsening_list = matching_optimal(G, weights=weights, r=r_cur)
            elif algorithm == "greedy":
                coarsening_list = matching_greedy(G, weights=weights, r=r_cur)
            
            # Skip if minimal reduction possible
            if len(coarsening_list) < 2:
                break
                
            # Get coarsening matrix and update
            iC = get_coarsening_matrix(G, coarsening_list)
            C = iC.dot(C)
            Call.append(iC)
            
            # Create coarsened graph
            Wc = zero_diag(coarsen_matrix(G.W, iC))  # Remove self-loops
            Wc = (Wc + Wc.T) / 2  # Ensure symmetry
            
            # Handle coordinate information if present
            if not hasattr(G, "coords"):
                Gc = graphs.Graph(Wc)
            else:
                Gc = graphs.Graph(Wc, coords=coarsen_vector(G.coords, iC))
                
            Gall.append(Gc)
            n = Gc.N
            
            # Early termination if target size reached
            if n <= n_target:
                break

        return C, Gc, Call, Gall