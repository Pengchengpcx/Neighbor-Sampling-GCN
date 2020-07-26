import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
from utils import sub_graph


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sample):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.sample = sample

    def forward(self, x, adj):
        adj1 = sub_graph(adj, self.sample[0])
        x = F.relu(self.gc1(x, adj1))
        x = F.dropout(x, self.dropout, training=self.training)
        adj2 = sub_graph(adj,self.sample[1])
        x = self.gc2(x, adj2)
        return F.log_softmax(x, dim=1)
