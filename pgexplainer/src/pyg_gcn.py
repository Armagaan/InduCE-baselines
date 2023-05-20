import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNN(nn.Module):
    """3-layer GCN used in GNN Explainer synthetic tasks"""
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GNN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.gc3 = GCNConv(nhid, nout)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # convert edge index to dense_adj
        x1 = F.relu(self.gc1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, edge_index))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, edge_index)
        x = self.lin(torch.cat((x1, x2, x3), dim=1))
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
