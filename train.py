#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from Datasets import *
from Models import *
from Methods import *
from Train import *
from Evaluation import *
from Utils import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
import torch.nn as nn
from torch_sparse import SparseTensor


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument("--num_heads", type=int, default=1, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'RobustGCN'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--tem', type=float, default=0.5, help='the temperature of contrastive learning loss '
                                                               '(mutual information maximize)')
    parser.add_argument('--alpha', type=float, default=0.25, help='empower coefficient')
    parser.add_argument('--lr_w', type=float, default=1,
                        help='the learning rate of the adaptive weight coefficient')
    parser.add_argument('--model_type', type=str, default='gnn', choices=['gnn', 'mlp', 'other'])
    parser.add_argument('--weight_path', type=str, default='./Weights/model_weight.pt')
    parser.add_argument('--save_results', type=bool, default=False)

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    fair_dataset = FairDataset(args.dataset, args.device)
    fair_dataset.load_data()

    # The number of classier
    # num_class = labels.unique().shape[0] - 1
    num_class = 1
    args.nfeat = fair_dataset.features.shape[1]
    args.nclass = num_class

    """
    Build model and optimizer
    """
    # constructing model and optimizer
    model_builder = BuildModel(args, args.device)
    model, optimizer = model_builder.build()

    """
    Train model (Teacher model and Student model)
    """
    # training vanilla model (for synthetic teacher)
    weight_path = f'./Weights/{args.model}_vanilla.pt'
    criterion = torch.nn.BCEWithLogitsLoss()
    trainer = Trainer(model, optimizer, criterion)
    trainer.train(fair_dataset, args.epochs, model_type=args.model_type, weight_path=weight_path)

    """
    evaluation
    """
    # auc_roc_test, f1_s, acc, parity, equality = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
    auc_roc_test, f1_s, acc, parity, equality = evaluate(model=model, weight_path=weight_path,
                                                         data=fair_dataset, model_type=args.model_type)

    return auc_roc_test, f1_s, acc, parity, equality


if __name__ == '__main__':
    # Training settings
    args = args_parser()

    model_num = 1
    results = Results(args.seed_num, model_num)

    for seed in range(args.seed_num):
        # set seeds
        set_seed(seed)

        # running train
        results.auc[seed, 0], results.f1[seed, 0], results.acc[seed, 0], results.parity[seed, 0], \
        results.equality[seed, 0] = run(args)
        print(f"========finish seed {seed}========")

    # reporting results
    results.report_results()
    if args.save_results:
        results.save_results(args)
