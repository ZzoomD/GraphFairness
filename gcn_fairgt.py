#%%
import argparse
import numpy as np
import torch
import random
import warnings
warnings.filterwarnings('ignore')

from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models import *
from graphfairness.methods import *
from graphfairness.train import *
from graphfairness.evaluation import *
from graphfairness.utils import *


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhid', type=str, default='64', help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument('--nhead', type=int, default=2, help='Number of attention heads.')
    parser.add_argument('--nlayer', type=int, default=1, help='Number of transformer layers.')
    parser.add_argument('--hops', type=int, default=2, help='Number of hop neighbors to aggregate.')
    parser.add_argument('--save_results', type=bool, default=False)

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.nhid = [int(x.strip()) for x in args.nhid.split(',')]

    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.allow_tf32 = False


def run(args):
    """
    Load data
    """
    root = './FairData'
    dataset = FairDataset(root=root, name=args.dataset)
    fair_dataset = dataset.data

    # Convert SparseTensor to edge_index if necessary
    try:
        from torch_sparse import SparseTensor
        if isinstance(fair_dataset.edge_index, SparseTensor):
            row, col, _ = fair_dataset.edge_index.coo()
            fair_dataset.edge_index = torch.stack([row, col], dim=0)
    except ImportError:
        pass

    fair_dataset = fair_dataset.to(args.device)

    args.nfeat = fair_dataset.features.shape[1]
    args.nclass = 1
    args.num_nodes = fair_dataset.features.shape[0]

    """
    Build model
    """
    model = GraphTransformer(nfeat=args.nfeat, 
                            nhid=args.nhid, 
                            nclass=args.nclass,
                            nhead=args.nhead,
                            nlayer=args.nlayer,
                            dropout=args.dropout).to(args.device)

    """
    Train model
    """
    fairgt = FairGT(model, 
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    hops=args.hops,
                    num_nodes=args.num_nodes)
    fairgt.train(fair_dataset, args.epochs)

    """
    Evaluation
    """
    results = fairgt.evaluate(fair_dataset)

    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']


if __name__ == '__main__':
    # Training settings
    args = args_parser()

    model_num = 1
    results = Results(args.seed_num, model_num)

    print(f"=============FairGT Train START=============")
    print(f"Dataset: {args.dataset}")
    print(f"Hidden dim: {args.nhid}, Heads: {args.nhead}, Layers: {args.nlayer}, Hops: {args.hops}")
    print(f"LR: {args.lr}, Dropout: {args.dropout}")
    print("="*50)
    
    for seed in range(args.seed_num):
        # set seeds
        set_seed(seed)

        # running train
        results.auc[seed, 0], results.f1[seed, 0], results.acc[seed, 0], results.parity[seed, 0], \
        results.equality[seed, 0] = run(args)
        print(f"Finished seed {seed}!")

    # reporting results
    print(f"=============FairGT Train END=============")
    results.report_results()
    if args.save_results:
        results.save_results(args)
