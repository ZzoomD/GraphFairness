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


def build_model(args):
    if args.model == 'gcn':
        model = GCN(nfeat=args.nfeat,
                    nhid=args.hidden,
                    nclass=args.nclass,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 'sage':
        model = SAGE(nfeat=args.nfeat,
                     nhid=args.hidden,
                     nclass=args.nclass,
                     dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 'gin':
        model = GIN(nfeat=args.nfeat,
                    nhid=args.hidden,
                    nclass=args.nclass,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 'mlpgcn':
        model = MLPGCN(nfeat=args.nfeat,
                       nhid=args.hidden,
                       nclass=args.nclass,
                       dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model == 'gcnori':
        model = GCNORI(nfeat=args.nfeat,
                       nhid=args.hidden,
                       nclass=args.nclass,
                       dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("Invalid model name")
        return
    return model, optimizer