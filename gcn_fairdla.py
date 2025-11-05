#%%
import argparse
import numpy as np

import torch

import random

import warnings
warnings.filterwarnings('ignore')

from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models import *
from graphfairness.methods.inprocess.fairdla import FairDLA
from graphfairness.train import *
from graphfairness.evaluation import *
from graphfairness.utils import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhid', type=str, default='16', help='Number of hidden units. split by comma .')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument("--num_heads", type=int, default=1, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--channels", type=int, default=2, help="number of channels")
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'RobustGCN', 'mlpgcn', 'gcnori', 'disengnn',
                                'disengcn', 'pcagcn', 'adagcn', 'adagcn_new'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--tem', type=float, default=0.5, help='the temperature of contrastive learning loss '
                                                               '(mutual information maximize)')
    parser.add_argument('--alpha', type=float, default=0.25, help='weight coefficient for disentanglement.')
    parser.add_argument('--beta', type=float, default=0.25, help='weight coefficient for channel masker.')
    parser.add_argument('--lr_w', type=float, default=1,
                        help='the learning rate of the adaptive weight coefficient')
    parser.add_argument('--model_type', type=str, default='gnn', choices=['gnn', 'mlp', 'other'])
    parser.add_argument('--weight_path', type=str, default='./Weights/model_weight.pt')
    parser.add_argument('--save_results', type=bool, default=False)
    parser.add_argument('--pre_seed', type=int, default=1)
    parser.add_argument('--pre_train', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--avgy', type=str2bool, default=False)
    parser.add_argument('--per', type=float, default=0.3)
    parser.add_argument('--rs', type=int, default=10)
    parser.add_argument('--copy', type=int, default=0)
    parser.add_argument('--adv', type=int, default=0)

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    if args.cuda:
        torch.cuda.set_device(args.device)
        args.device = torch.device(f'cuda:{args.device}')
    else:
        args.device = torch.device('cpu')
    
    args.nhid = [int(x.strip()) for x in args.nhid.split(',')]

    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run(args):
    """
    Load data
    """
    root = '/home/yczhu/public_data/pyg_data/FairData'
    dataset = FairDataset(root=root, name=args.dataset)
    fair_dataset = dataset.data
    fair_dataset = fair_dataset.to(args.device)

    args.nfeat = fair_dataset.features.shape[1]
    args.nclass = 1

    """
    Build model and optimizer
    """
    # Initialize FairDLA model
    model = FairDLA(nfeat=args.nfeat,
                    nhid=args.nhid[0],  # Using the first hidden size
                    nclass=args.nclass,
                    channels=args.channels,
                    dropout=args.dropout,
                    lr=args.lr,
                    weight_decay=args.weight_decay)

    """
    Train model
    """
    # Move model components to device
    model.encoder = model.encoder.to(args.device)
    model.classifier = model.classifier.to(args.device)
    model.channel_cls = model.channel_cls.to(args.device)
    
    # Train the model
    auc, f1, acc, parity, equality = model.train_fit(fair_dataset, args.epochs, alpha=args.alpha, device=args.device)

    return auc, f1, acc, parity, equality


if __name__ == '__main__':
    # Training settings
    args = args_parser()

    model_num = 1
    results = Results(args.seed_num, model_num)

    print(f"=============Train FairDLA START=============")
    for seed in range(args.seed_num):
        # set seeds
        set_seed(seed)

        # running train
        results.auc[seed, 0], results.f1[seed, 0], results.acc[seed, 0], results.parity[seed, 0], \
        results.equality[seed, 0] = run(args)
        print(f"finish seed {seed}!")

    # reporting results
    print(f"=============Train FairDLA END=============")
    results.report_results()
    if args.save_results:
        results.save_results(args)