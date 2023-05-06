import torch.optim as optim
from .gcn import GCN
from .gat import GAT
from .sage import SAGE
from .gin import GIN
from .jk import JK
from .rogcn import RobustGCN
from .prognn import ProGNN
from .fairgnn import FairGNN
from .infomax import Encoder_DGI, Encoder_CLS, GraphInfoMax
from .ssf import SSF, Encoder, Classifier, drop_feature
from .mlp import MLP
from .projector import Projector
from .pre_model import GcnTopo, Mlp, CombineModel
from .fairgkd import SynTeacher
from .mlp_gcn import MLPGCN, GCNORI


class BuildModel:
    def __init__(self, args, device):
        super(BuildModel, self).__init__()
        self.args = args
        self.device = device

    def build(self):
        if self.args.model == 'gcn':
            model = GCN(nfeat=self.args.nfeat,
                        nhid=self.args.hidden,
                        nclass=self.args.nclass,
                        dropout=self.args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model = model.to(self.device)
        elif self.args.model == 'sage':
            model = SAGE(nfeat=self.args.nfeat,
                         nhid=self.args.hidden,
                         nclass=self.args.nclass,
                         dropout=self.args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model = model.to(self.device)
        elif self.args.model == 'gin':
            model = GIN(nfeat=self.args.nfeat,
                        nhid=self.args.hidden,
                        nclass=self.args.nclass,
                        dropout=self.args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=args.weight_decay)
            model = model.to(self.device)
        elif self.args.model == 'mlpgcn':
            model = MLPGCN(nfeat=self.args.nfeat,
                           nhid=self.args.hidden,
                           nclass=self.args.nclass,
                           dropout=self.args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model = model.to(self.device)
        elif self.args.model == 'gcnori':
            model = GCNORI(nfeat=self.args.nfeat,
                           nhid=self.args.hidden,
                           nclass=self.args.nclass,
                           dropout=self.args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            model = model.to(self.device)
        else:
            raise RuntimeError("Invalid model name")
        return model, optimizer
