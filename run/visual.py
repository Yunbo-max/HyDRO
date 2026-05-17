import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from graphslim.config import cli
from graphslim.dataset import *
from graphslim.evaluation import Evaluator, PropertyEvaluator
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import pandas as pd
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import os



import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import os

# Ensure the directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to plot and save the graph
def plot_graph(G, save_path='graph_plot.png', figsize=(12, 12), dpi=300):
    print(f"Plotting graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges...")

    # Use a suitable layout for large graphs
    if G.number_of_nodes() > 5000:
        pos = nx.spring_layout(G, seed=42, k=0.1)  # Optimized for large graphs
    else:
        pos = nx.kamada_kawai_layout(G)  # Works better for smaller graphs

    plt.figure(figsize=figsize, dpi=dpi)
    nx.draw(G, pos, node_size=10, edge_color="gray", alpha=0.3, with_labels=False)

    # Save the figure
    plt.title("Graph Visualization", fontsize=16, fontweight='bold')
    plt.axis('off')  # Hide axes for better clarity
    plt.savefig(save_path, format='PNG', bbox_inches='tight', dpi=dpi)
    plt.show()
    print(f"Graph plot saved to {save_path}")


# Main execution
if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)

    graph_dir = './Graph_Drawing'
    ensure_directory_exists(graph_dir)

    if args.eval_whole:
        # Convert original dataset to NetworkX graph
        G = to_networkx(data, to_undirected=True)

        # Add node labels from the original dataset
        for node, label in enumerate(data.y):
            G.nodes[node]['label'] = str(label.item())

        save_path_gexf = f'{graph_dir}/{args.dataset}_whole.gexf'

    else:

        flag = 0
        save_path = f'{args.save_path}/reduced_graph/{args.method}'
        
        feat_syn = torch.load(
            f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
   
    
         

        labels_syn = torch.load(
            f'{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
    



        adj_syn = torch.load(
            f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
 
      
    
        # Create a new PyG Data object using synthetic data
        synthetic_data = Data(x=feat_syn, edge_index=adj_syn.nonzero().t(), y=labels_syn)

        # Convert synthetic PyG data to NetworkX graph
        G = to_networkx(synthetic_data, to_undirected=True)

        # Add synthetic labels to the graph
        for node, label in enumerate(synthetic_data.y):
            G.nodes[node]['label'] = str(label.item())

        save_path_gexf = f'{graph_dir}/{args.dataset}_{args.reduction_rate}.gexf'

    # Save graph in GEXF format
    save_graph(G, save_path_gexf)
