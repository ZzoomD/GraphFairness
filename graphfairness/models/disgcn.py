import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.dense.linear import Linear as PyGLinear


class NeiborAssigner(nn.Module):
    """
    Identify latent factors causing the connection between nodes.
    """
    def __init__(self, nfeat, channels):
        super(NeiborAssigner, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * nfeat, out_features=channels),
            nn.Linear(in_features=channels, out_features=channels)
        )
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features_pair):
        alpha_score = self.layers(features_pair)
        return torch.softmax(alpha_score, dim=1)

class DisenLayer(MessagePassing):
    """
    Multi-channel graph convolution for disentangled representations.
    """
    def __init__(self, in_dim, out_dim, channels, reduce=True):
        super(DisenLayer, self).__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.per_channel_dim = out_dim // channels
        self.reduce = reduce
        
        self.lin_layers = nn.ModuleList([nn.Linear(in_dim, self.per_channel_dim) if reduce else None for _ in range(channels)])
        self.conv_layers = nn.ModuleList([PyGLinear(self.per_channel_dim if reduce else in_dim, self.per_channel_dim, bias=False, 
                                                    weight_initializer='glorot') for _ in range(channels)])
        self.bias_list = nn.ParameterList([nn.Parameter(torch.empty(size=(1, self.per_channel_dim), dtype=torch.float), requires_grad=True) for _ in range(channels)])

    def forward(self, x, edge_index, edge_weight):
        c_feats = []
        for k in range(self.channels):
            z_k = self.lin_layers[k](x) if self.reduce else x
            h_k = self.conv_layers[k](z_k)
            
            edge_index_k = edge_index.clone()
            if not edge_index_k.has_value():
                edge_index_k = edge_index_k.fill_value(1.0)
            edge_index_k.storage.set_value_(edge_index_k.storage.value() * edge_weight[:, k])
            
            out = self.propagate(edge_index_k, x=h_k)
            out = out + self.bias_list[k]
            c_feats.append(F.normalize(out, p=2, dim=1))
            
        return torch.cat(c_feats, dim=1)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class DisGCN(nn.Module):
    """
    Backbone for FairSAD using multi-channel disentangled graph convolution.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, **model_args):
        super(DisGCN, self).__init__()
        self.channels = model_args.get('channels', 4)
        self.nfeat = nfeat
        self.nclass = nclass
        self.nhid_list = [nhid] if not isinstance(nhid, list) else nhid
        
        self.assigner = NeiborAssigner(nfeat, self.channels)
        self.disenlayers = nn.ModuleList()
        for i, nhid in enumerate(self.nhid_list):
            in_d = self.nfeat if i == 0 else nhid
            self.disenlayers.append(DisenLayer(in_d, nhid, self.channels, reduce=True))
        
        self.dropout = nn.Dropout(dropout)
        self.init_parameters()

    def init_parameters(self):
        """
        Initialize all parameters with normal distribution as per original code.
        """
        for item in self.parameters():
            torch.nn.init.normal_(item, mean=0, std=1)
    
    def init_edge_weight(self):
        for m in self.assigner.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        row, col, _ = edge_index.coo()
        feats_pair = torch.cat([x[col], x[row]], dim=1)
        edge_weight = self.assigner(feats_pair)
        
        for layer in self.disenlayers:
            x = layer(x, edge_index, edge_weight)
            x = self.dropout(x)
        return x