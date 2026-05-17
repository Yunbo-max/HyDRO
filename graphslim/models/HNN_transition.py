import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import scipy.sparse as sp
import networkx as nx
import geoopt
import math
import warnings
from itertools import product
from scipy.stats import wasserstein_distance
from torch_geometric.utils import to_networkx, degree
try:
    from torch_geometric.utils import from_scipy_sparse_matrix
except ImportError:
    from torch_geometric.utils import from_scipy_sparse_array as from_scipy_sparse_matrix
from numpy.linalg import norm
from scipy.sparse import coo_matrix

# Suppress warnings and configure deterministic CUDA
warnings.filterwarnings('ignore')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HyperbolicNet(nn.Module):
    """Hyperbolic Graph Neural Network with Poincaré ball embeddings.
    
    This network learns node representations in hyperbolic space and produces
    graph adjacency matrices while preserving spectral properties.
    """
    
    def __init__(self, nfeat, nnodes, nhid=256, 
                 nlayers=3, device=None, args=None, Lx=None):
        super().__init__()
        

        # Hyperbolic manifold setup
        self.curvature = args.curvature
        self.manifold = geoopt.PoincareBall(c=self.curvature)
        
        # Network architecture
        self._build_layers(nfeat, nhid, nlayers)
        
        # Graph structure setup
        self._setup_graph_components(nnodes)
        
        # Training state
        self.iteration = 0
        self.cnt = 0
        self.Lx_inv = None
        self.device = device
    
    def _build_layers(self, nfeat, nhid, nlayers):
        """Construct the hyperbolic neural network layers."""
        self.layers = nn.ModuleList([
            MobiusLinear(nfeat * 2, nhid, curvature=self.curvature)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(nhid)])
        
        # Add hidden layers
        for _ in range(nlayers - 2):
            self.layers.append(MobiusLinear(nhid, nhid, curvature=self.curvature))
            self.bns.append(nn.BatchNorm1d(nhid))
            
        # Output layer
        self.layers.append(MobiusLinear(nhid, 1, curvature=self.curvature))
    
    def _setup_graph_components(self, nnodes):
        """Initialize graph-related components."""
        edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        self.edge_index = edge_index.T
        self.nnodes = nnodes
    
    def forward(self, x, manifold=None, inference=False, Lx=None):
        """Forward pass through the hyperbolic network."""
        manifold = manifold or self.manifold
        
        # Create edge embeddings
        edge_embed = self._create_edge_embeddings(x)
        
        # Process through hyperbolic layers
        adj = self._process_through_layers(edge_embed, manifold)
        
        # Symmetrize and normalize adjacency
        adj = self._postprocess_adjacency(adj)
        
        if inference:
            return adj
            
        # Calculate spectral properties if needed
        if Lx is not None and self.Lx_inv is None:
            self._calculate_spectral_properties(Lx)
            
        return adj, self.opt_loss(adj)
    
    def _create_edge_embeddings(self, x):
        """Create hyperbolic embeddings for all edges."""
        edge_embed = torch.cat([
            x[self.edge_index[0]], 
            x[self.edge_index[1]]
        ], dim=1)
        return self.manifold.expmap0(torch.flatten(edge_embed, start_dim=1))
    
    def _process_through_layers(self, x, manifold):
        """Process input through all hyperbolic layers."""
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x)
                x = hyperbolic_ReLU(x, manifold)
        return x.reshape(self.nnodes, self.nnodes)
    
    def _postprocess_adjacency(self, adj):
        """Symmetrize and normalize adjacency matrix."""
        adj = (adj + adj.T) / 2
        adj = torch.sigmoid(adj)
        return adj - torch.diag(torch.diag(adj))
    
    def _calculate_spectral_properties(self, Lx):
        """Calculate and cache spectral properties of input graph."""
        W2 = calculate_lazy_random_walk_matrix2(Lx, self.device)
        self.Lx_inv = calculate_spectral_gap_W(W2)
    
    def opt_loss(self, adj):
        """Calculate spectral gap preservation loss."""
        W = calculate_lazy_random_walk_matrix(adj)
        spectral_gap_syn = calculate_spectral_gap_W(W)
        return torch.abs(spectral_gap_syn - self.Lx_inv)
    
    @torch.no_grad()
    def inference(self, x):
        """Inference mode forward pass."""
        return self.forward(x, manifold=self.manifold, inference=True)
    
    def reset_parameters(self):
        """Reset all trainable parameters."""
        def weight_reset(m):
            if isinstance(m, MobiusLinear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)





# package.nn.modules.py
def create_ball(ball=None, c=None):
    """
    Helper to create a PoincareBall.

    Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
    In this case you will require same curvature parameters for different layers or end up with nans.

    Parameters
    ----------
    ball : geoopt.PoincareBall
    c : float

    Returns
    -------
    geoopt.PoincareBall
    """
    if ball is None:
        assert c is not None, "curvature of the ball should be explicitly specified"
        ball = geoopt.PoincareBall(c)
    # else trust input
    return ball


class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, curvature=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball = create_ball(ball, curvature)
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.

        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            nonlin=self.nonlin,
            ball=self.ball,
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()


def mobius_linear(input, weight, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    if nonlin is not None:
        output = ball.logmap0(output)
        output = nonlin(output)
        output = ball.expmap0(output)
    return output



# Helper functions
def hyperbolic_ReLU(hyperbolic_input, manifold):
    """Hyperbolic ReLU activation function."""
    euclidean_input = manifold.logmap0(hyperbolic_input)
    euclidean_output = F.relu(euclidean_input)
    return manifold.expmap0(euclidean_output)

def calculate_lazy_random_walk_matrix(adj):
    """Calculate lazy random walk matrix from adjacency."""
    degree_matrix = torch.diag(torch.sum(adj, dim=1))
    d_inv = torch.diag(torch.pow(torch.sum(degree_matrix, dim=1).float(), -1))
    I = torch.eye(adj.shape[0], device=adj.device)
    return 0.5 * (I + torch.matmul(d_inv, adj))

def calculate_lazy_random_walk_matrix2(adj, device):
    """Alternative lazy random walk matrix calculation."""
    I = torch.eye(adj.shape[0], device=device)
    return 0.5 * (I + adj)

def calculate_spectral_gap_W(W):
    """Calculate spectral gap from random walk matrix."""
    eigenvalues = torch.linalg.eigvals(W)
    sorted_eigenvalues = torch.sort(eigenvalues.real, descending=True)[0]
    return 1 - sorted_eigenvalues[-2]

