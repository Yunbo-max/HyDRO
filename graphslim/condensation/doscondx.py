from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.models import *
from graphslim.utils import *
import torch.nn.functional as F


class DosCondX(GCondBase):
    """
    A structure-free variant of DosCond. "Condensing Graphs via One-Step Gradient Matching" https://arxiv.org/abs/2206.07746
    """
    def __init__(self, setting, data, args, **kwargs):
        super(DosCondX, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):

        args = self.args
        feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train,
                                              device=self.device)

        # initialization the features
        feat_init = self.init(with_adj=False)
        self.feat_syn.data.copy_(feat_init)

        self.adj_syn = torch.eye(feat_init.shape[0], device=self.device)

        adj = normalize_adj_tensor(adj, sparse=True)

        outer_loop, inner_loop = self.get_loops(args)
        loss_avg = 0
        best_val = 0

        model = eval(args.condense_model)(feat_syn.shape[1], args.hidden, data.nclass, args).to(self.device)
        for it in trange(args.epochs):
            # seed_everything(args.seed + it)

            model.initialize()
            model.train()

            for ol in range(outer_loop):
                model = self.check_bn(model)
                loss = self.train_class(model, adj, features, labels, labels_syn, args)

                loss_avg += loss.item()

                self.optimizer_feat.zero_grad()
                loss.backward()
                self.optimizer_feat.step()

            loss_avg /= (data.nclass * outer_loop)
            if it in args.checkpoints:
                data.adj_syn, data.feat_syn, data.labels_syn = torch.eye(
                    self.feat_syn.shape[0]).detach(), self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
