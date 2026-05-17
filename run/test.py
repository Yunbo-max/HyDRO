import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
from torch_geometric.datasets import Planetoid

import torch
import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid

# Function to normalize the adjacency matrix
def normalize_adj_tensor(adj):
    row_sum = np.sum(adj, axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1  # To avoid division by zero for isolated nodes
    return adj / row_sum

# Function to calculate the lazy random walk matrix
def calculate_lazy_random_walk_matrix(adj):
    degree_matrix = torch.diag(torch.sum(adj, dim=1))
    d_inv = torch.inverse(degree_matrix)
    I = torch.eye(adj.shape[0])
    W = 0.5 * (I + torch.matmul(d_inv, adj))
    return W

def calculate_lazy_random_walk_matrix2(adj):
    # Move the adjacency matrix to the correct device
    print('adj type',type(adj))

    adj = adj
    
    

    # Lazy random walk matrix: W = 1/2 (I + D^(-1) * A)
    I = torch.eye(adj.shape[0])
    W = 0.5 * (I + adj)
    
    return W

# Function to calculate the spectral gap
def calculate_spectral_gap(W):
    eigenvalues = torch.linalg.eigvals(W)
    eigenvalues_real = eigenvalues.real  # Keep only the real parts
    sorted_eigenvalues = torch.sort(eigenvalues_real, descending=True).values
    second_largest_eigenvalue = sorted_eigenvalues[1]  # Second largest eigenvalue
    spectral_gap = 1 - second_largest_eigenvalue
    return spectral_gap.item()

# Load the Cora dataset from PyTorch Geometric
dataset = Planetoid(root='.', name='citeseer')

# Extract the graph (Cora) from the dataset
graph_original = dataset[0]

# Get the edge_index and convert it to an adjacency matrix
edge_index = graph_original.edge_index.numpy()
adj_original = np.zeros((graph_original.num_nodes, graph_original.num_nodes))
for i in range(edge_index.shape[1]):
    adj_original[edge_index[0, i], edge_index[1, i]] = 1
    adj_original[edge_index[1, i], edge_index[0, i]] = 1  # Undirected graph
print(adj_original)
# Step 3: Convert the adjacency matrix to a NetworkX graph
graph_nx = nx.from_numpy_array(adj_original)

# Step 4: Identify connected components
connected_components = list(nx.connected_components(graph_nx))

# Option 1: Select the largest connected component
largest_cc = max(connected_components, key=len)

# Step 5: Extract the adjacency matrix for the largest connected component
subgraph_nx = graph_nx.subgraph(largest_cc).copy()
adj_largest_cc = nx.to_numpy_array(subgraph_nx)


# Step 6: Print or normalize the adjacency matrix of the largest connected component
def normalize_adj_tensor(adj):
    """Normalize an adjacency matrix."""
    degree = np.sum(adj, axis=1)  # Degree matrix
    degree[degree == 0] = 1  # Avoid division by zero
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return np.dot(np.dot(d_inv_sqrt, adj), d_inv_sqrt)

adj_largest_cc = normalize_adj_tensor(adj_largest_cc)

adj_original = adj_largest_cc


# Load the modified graphs from the files
file_path_modified = "checkpoints/reduced_graph/sgdd/adj_pubmed_0.5_1.pt"
file_path_modified2 = "checkpoints/reduced_graph/gdem/adj_pubmed_0.5_1.pt"
file_path_modified3 = "checkpoints/reduced_graph/gcond/adj_pubmed_0.5_1.pt"
file_path_modified4 = "checkpoints/reduced_graph/hydro/adj_pubmed_0.5_14.pt"
adj_modified = torch.load(file_path_modified, map_location=torch.device('cpu')).numpy()
adj_modified2 = torch.load(file_path_modified2, map_location=torch.device('cpu')).numpy()
adj_modified3 = torch.load(file_path_modified3, map_location=torch.device('cpu')).numpy()
adj_modified4 = torch.load(file_path_modified4, map_location=torch.device('cpu')).numpy()

# Normalize the modified adjacency matrices
# adj_modified = normalize_adj_tensor(adj_modified)
# adj_modified2 = normalize_adj_tensor(adj_modified2)

# Convert adjacency matrices to tensors
adj_original_tensor = torch.tensor(adj_original, dtype=torch.float32)
adj_modified_tensor = torch.tensor(adj_modified, dtype=torch.float32)
adj_modified2_tensor = torch.tensor(adj_modified2, dtype=torch.float32)
adj_modified3_tensor = torch.tensor(adj_modified3, dtype=torch.float32)
adj_modified4_tensor = torch.tensor(adj_modified4, dtype=torch.float32)

# Calculate lazy random walk matrices
W_original = calculate_lazy_random_walk_matrix2(adj_original_tensor)
W_modified = calculate_lazy_random_walk_matrix(adj_modified_tensor)
W_modified2 = calculate_lazy_random_walk_matrix(adj_modified2_tensor)
W_modified3 = calculate_lazy_random_walk_matrix(adj_modified3_tensor)
W_modified4 = calculate_lazy_random_walk_matrix(adj_modified4_tensor)


# # Calculate spectral gaps
spectral_gap_original = calculate_spectral_gap(W_original)
spectral_gap_modified = calculate_spectral_gap(W_modified)
spectral_gap_modified2 = calculate_spectral_gap(W_modified2)
spectral_gap_modified3 = calculate_spectral_gap(W_modified3)
spectral_gap_modified4 = calculate_spectral_gap(W_modified4)

# Print results
print(f"Spectral Gap (Original Graph): {spectral_gap_original:.4f}")
print(f"Spectral Gap (Modified Graph 1): {spectral_gap_modified:.4f}")
print(f"Spectral Gap (Modified Graph 2): {spectral_gap_modified2:.4f}")
print(f"Spectral Gap (Modified Graph 3): {spectral_gap_modified3:.4f}")
print(f"Spectral Gap (Modified Graph 4): {spectral_gap_modified4:.4f}")


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.csgraph import dijkstra

# Flow distance calculation using Dijkstra's algorithm
def compute_flow_distance(W):
    """Compute the flow distance matrix using Dijkstra's algorithm for each node."""
    # Convert W into a form suitable for shortest path algorithms (e.g., adjacency matrix)
    flow_distance_matrix = np.zeros((W.shape[0], W.shape[0]))
    
    # Compute shortest path for each pair of nodes
    for i in range(W.shape[0]):
        dist, _ = dijkstra(W, return_predecessors=True, indices=i, directed=False)
        flow_distance_matrix[i] = dist
    
    return flow_distance_matrix

# Compute flow distance matrices for each graph
flow_distance_original = compute_flow_distance(adj_original)
flow_distance_modified = compute_flow_distance(adj_modified)
flow_distance_modified2 = compute_flow_distance(adj_modified2)
flow_distance_modified3 = compute_flow_distance(adj_modified3)
flow_distance_modified4 = compute_flow_distance(adj_modified4)


# def compute_flow_distance(graph):
#     """
#     Compute the flow distance (shortest path distance) for all pairs of nodes in a weighted graph.
    
#     Parameters:
#     - graph: A NetworkX graph with edge weights.

#     Returns:
#     - distance_matrix: A NumPy array representing the flow distance matrix.
#     """
#     # Compute the shortest path length for all node pairs
#     shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
    
#     # Convert the dictionary of shortest path lengths into a matrix
#     nodes = list(graph.nodes)
#     num_nodes = len(nodes)
#     distance_matrix = np.zeros((num_nodes, num_nodes))
    
#     for i, node_i in enumerate(nodes):
#         for j, node_j in enumerate(nodes):
#             distance_matrix[i, j] = shortest_path_lengths[node_i].get(node_j, np.inf)
    
#     return distance_matrix, nodes

# # Updated file path for Windows
# file_path = r"C:\Users\Scail.WINDOWS-4FTU4OO\Desktop\IJCAI\GraphSlim\benchmark\checkpoints\reduced_graph\gdem\adj_cora_0.5_1.pt"

# # Load the adjacency matrix with CPU mapping
# adjacency_matrix = torch.load(file_path, map_location=torch.device('cpu')).numpy()

# # Create a NetworkX graph from the adjacency matrix
# graph = nx.from_numpy_array(adj_original)
# graph1 = nx.from_numpy_array(adj_modified)
# graph2= nx.from_numpy_array(adj_modified2)
# graph3 = nx.from_numpy_array(adj_modified3)
# graph4 = nx.from_numpy_array(adj_modified4)


# # Compute the flow distance matrix
# flow_distance_original, node_labels = compute_flow_distance(graph)
# flow_distance_modified, node_labels = compute_flow_distance(graph1)
# flow_distance_modified2, node_labels = compute_flow_distance(graph2)
# flow_distance_modified3, node_labels = compute_flow_distance(graph3)
# flow_distance_modified4, node_labels = compute_flow_distance(graph4)

# Plot heatmap for the flow distance matrix
# Modify the plot_heatmap function to set the color bar range
def plot_heatmap(flow_distance_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(flow_distance_matrix, cmap="YlGnBu", annot=False, fmt=".2f", square=True,
                vmin=0)  # Set the color bar range here
    plt.title(title)
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.show()



# Normalize the flow distance matrices by the number of nodes
def normalize_flow_distance(flow_distance_matrix):
    num_nodes = flow_distance_matrix.shape[0]
    return flow_distance_matrix / num_nodes

# Normalize flow distances
flow_distance_original_normalized = normalize_flow_distance(flow_distance_original)
flow_distance_modified_normalized = normalize_flow_distance(flow_distance_modified)
flow_distance_modified2_normalized = normalize_flow_distance(flow_distance_modified2)
flow_distance_modified3_normalized = normalize_flow_distance(flow_distance_modified3)
flow_distance_modified4_normalized = normalize_flow_distance(flow_distance_modified4)


# # Plot heatmaps for the normalized flow distance matrices
# plot_heatmap(flow_distance_original, "Normalized Flow Distance: Original Graph")
# plot_heatmap(flow_distance_modified, "Normalized Flow Distance: Modified Graph")
# plot_heatmap(flow_distance_modified2, "Normalized Flow Distance: Modified2 Graph")
# plot_heatmap(flow_distance_modified3, "Normalized Flow Distance: Modified2 Graph")
# plot_heatmap(flow_distance_modified4, "Normalized Flow Distance: Modified2 Graph")


# Plot heatmaps for the normalized flow distance matrices
plot_heatmap(flow_distance_original_normalized, "Normalized Flow Distance: Original Graph")
plot_heatmap(flow_distance_modified_normalized, "Normalized Flow Distance: Modified Graph")
plot_heatmap(flow_distance_modified2_normalized, "Normalized Flow Distance: Modified2 Graph")
plot_heatmap(flow_distance_modified3_normalized, "Normalized Flow Distance: Modified2 Graph")
plot_heatmap(flow_distance_modified4_normalized, "Normalized Flow Distance: Modified2 Graph")