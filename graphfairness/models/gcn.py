# import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import List

class GCN(nn.Module):
    def __init__(self, nfeat: int, nhid: List[int]=[16], 
                 nclass: int=1, dropout: float=0.5, **model_args):
        super(GCN, self).__init__()

        in_channel = nfeat
        conv = []
        for hid in nhid:
            conv.append(GCNConv(in_channel, hid))
            in_channel = hid
        self.conv = nn.ModuleList(conv)
        self.fc = nn.Linear(nhid[-1], nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for conv in self.conv:
            h = conv(h, edge_index, edge_weight)
        output = self.fc(h)
        return output
    
    def get_embs_and_outs(self, x, edge_index, edge_weight=None):
        h = x
        for conv in self.conv:
            h = conv(h, edge_index, edge_weight)
        output = self.fc(h)
        return h, output
