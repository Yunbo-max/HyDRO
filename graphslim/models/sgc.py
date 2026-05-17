# """multiple transformaiton and multiple propagation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse

from graphslim.models.base import BaseGNN
from graphslim.models.layers import MyLinear


# class SGC(BaseGNN):
#     '''
#     multiple transformation layers
#     '''

#     def __init__(self, nfeat, nhid, nclass, args, mode='train'):

#         """nlayers indicates the number of propagations"""
#         super(SGC, self).__init__(nfeat, nhid, nclass, args, mode)

#         if mode in ['eval']:
#             self.ntrans = 1
#         if self.ntrans == 1:
#             self.layers.append(MyLinear(nfeat, nclass))
#         else:
#             self.layers.append(MyLinear(nfeat, nhid))
#             if self.with_bn:
#                 self.bns = torch.nn.ModuleList()
#                 self.bns.append(nn.BatchNorm1d(nhid))
#             for i in range(self.ntrans - 2):
#                 if self.with_bn:
#                     self.bns.append(nn.BatchNorm1d(nhid))
#                 self.layers.append(MyLinear(nhid, nhid))
#             self.layers.append(MyLinear(nhid, nclass))

#     def forward(self, x, adj, output_layer_features=False):
#         for ix, layer in enumerate(self.layers):
#             x = layer(x)
#             if ix != len(self.layers) - 1:
#                 x = self.bns[ix](x) if self.with_bn else x
#                 x = F.relu(x)
#                 x = F.dropout(x, self.dropout, training=self.training)

#         for i in range(self.nlayers):
#             if isinstance(adj, list):
#                 x = torch_sparse.matmul(adj[i], x)
#             elif type(adj) == torch.Tensor:
#                 x = adj @ x
#             else:
#                 x = torch_sparse.matmul(adj, x)

#         x = x.view(-1, x.shape[-1])
#         if output_layer_features:
#             return x, F.log_softmax(x, dim=1)
#         else:
#             return F.log_softmax(x, dim=1)

 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_geometric.utils import negative_sampling
from sklearn.metrics import accuracy_score

class SGCLinkPred(BaseGNN):
    '''
    SGC adapted for link prediction.
    '''

    def __init__(self, nfeat, nhid, nclass, args, mode='train'):
        super(SGCLinkPred, self).__init__(nfeat, nhid, nclass, args, mode)

        # Transformation layers
        if mode in ['eval']:
            self.ntrans = 1
        if self.ntrans == 1:
            self.layers.append(MyLinear(nfeat, nhid))
        else:
            self.layers.append(MyLinear(nfeat, nhid))
            if self.with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(self.ntrans - 2):
                if self.with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
                self.layers.append(MyLinear(nhid, nhid))
            self.layers.append(MyLinear(nhid, nhid))

    def encode(self, x, adj):
        '''
        Encode node features into embeddings.
        '''
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        # Propagation
        for i in range(self.nlayers):
            if isinstance(adj, list):
                x = torch_sparse.matmul(adj[i], x)
            elif type(adj) == torch.Tensor:
                x = adj @ x
            else:
                x = torch_sparse.matmul(adj, x)

        return x

    def decode(self, z, edge_label_index):
        '''
        Decode node embeddings to predict edge probabilities.
        '''
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)  # Dot product for link prediction

    def forward(self, x, adj, edge_label_index):
        '''
        Forward pass for link prediction.
        '''
        z = self.encode(x, adj)
        return self.decode(z, edge_label_index)


def train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=100, verbose=True):
    '''
    Train the link predictor model.
    '''
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Encode node features
        z = model.encode(train_data.x, train_data.edge_index)

        # Sample negative edges
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        # Combine positive and negative edges
        edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        # Decode embeddings to predict edge probabilities
        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        # Validation
        val_f1 = eval_link_predictor(model, val_data)

        if epoch % 10 == 0 and verbose:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val F1: {val_f1:.3f}")

    return model


@torch.no_grad()
def eval_link_predictor(model, data):
    '''
    Evaluate the link predictor model.
    '''
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    pred = (out > 0.5).float().cpu().numpy()
    return accuracy_score(data.edge_label.cpu().numpy(), pred)


def link_pred(nfeat, train_data, val_data, test_data, nruns):
    '''
    Run link prediction experiments.
    '''
    res = []
    for _ in range(nruns):
        model = SGCLinkPred(nfeat, 128, 64, args).to(args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        model = train_link_predictor(model, train_data, val_data, optimizer, criterion)
        test_auc = eval_link_predictor(model, test_data)
        res.append(test_auc)
    print(f"Test mean: {np.array(res).mean(axis=0):.4f}, std: {np.array(res).std(axis=0):.4f}")