
from collections import Counter

import torch.nn as nn

from graphslim.condensation.gcond_base import GCondBase
from graphslim.condensation.utils import match_loss
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation import *

from graphslim.evaluation.utils import *
from graphslim.models.HNN_transition import *
from graphslim.utils import *
from tqdm import trange

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_sparse import matmul

from torch_geometric.utils import dense_to_sparse
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.evaluation.utils import *
from graphslim.models import *
from torch_sparse import SparseTensor
from graphslim.dataset.convertor import ei2csr
from graphslim.utils import accuracy, seed_everything, normalize_adj_tensor, to_tensor, is_sparse_tensor, is_identity, \
    f1_macro


class HyDRO(GCondBase):


    def __init__(self, setting, data, args, **kwargs):
        super(HyDRO, self).__init__(setting, data, args, **kwargs)
        self.data = data
        self.args = args
        print('feat_train',data.feat_train.shape)
        self.data.labels_syn = self.generate_labels_syn(data)
        n = self.data.labels_syn.shape[0]
        d = data.feat_train.shape[1]


        self.hyp = HyperbolicNet(nfeat=d, nnodes=n, device=self.device,args=args).to(self.device)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = geoopt.optim.RiemannianSGD(list(self.hyp.parameters()), lr=args.lr_adj, momentum=args.momentum,weight_decay=args.weight_decay_hyper)
        
        print('adj_syn:', (self.nnodes_syn, self.nnodes_syn), 'feat_syn:', self.feat_syn.shape)


    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args

     


        feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        feat_init = self.init()
        self.feat_syn.data.copy_(feat_init)

        adj = normalize_adj_tensor(adj, sparse=True)

        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        best_val = 0

        model = eval(args.condense_model)(feat_syn.shape[1], args.hidden,
                                          data.nclass, args).to(self.device)
        for it in trange(args.epochs):

            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()

            for ol in range(outer_loop):

                if adj.size(0) > 5000:
                    random_nodes = np.random.choice(list(range(adj.size(0))), 5000, replace=False)
                    adj_syn, opt_loss = self.hyp(self.feat_syn, Lx=adj[random_nodes].to_dense()[:, random_nodes])
                else:
                    adj_syn, opt_loss = self.hyp(self.feat_syn, Lx=adj)


                self.adj_syn = normalize_adj_tensor(adj_syn, sparse=False)
                model = self.check_bn(model)
                loss = self.train_class(model, adj, features, labels, labels_syn, args)
                loss_opt = opt_loss
           

                
                loss_avg += loss.item()

                if args.beta > 0:
                    loss_reg = args.beta * regularization(adj_syn, tensor2onehot(labels_syn.to('cpu')).to(adj_syn.device), args)
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg

                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()

                if it % 50 < 10:
                    loss = loss + loss_opt
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()
                feat_syn_inner = self.feat_syn.detach()
                adj_syn_inner = self.hyp.inference(feat_syn_inner)
                adj_syn_inner_norm = normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()

            loss_avg /= (data.nclass * outer_loop)

            if it in args.checkpoints:
                self.adj_syn = adj_syn_inner
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
