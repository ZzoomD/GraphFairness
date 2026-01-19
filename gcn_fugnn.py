#%%
import argparse
import numpy as np
import torch
import random
import warnings
warnings.filterwarnings('ignore')

from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models.fugnn import FUGNN as FUGNNModel
from graphfairness.methods.inprocess.fugnn import FairFUGNN 
from graphfairness.utils import Results

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhid', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--dataset', type=str, default='credit',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument('--nhead', type=int, default=1, help='Number of attention heads.')
    parser.add_argument('--nlayer', type=int, default=1, help='Number of layers.')
    parser.add_argument('--k', type=int, default=10, help='Number of eigenvalues/eigenvectors.')
    parser.add_argument('--norm', type=str, default='none', choices=['none', 'layer', 'batch'])
    parser.add_argument('--save_results', type=bool, default=False)

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    args.nclass = 2 # FUGNN uses CrossEntropy with 2 outputs for binary labels

    """
    Build model
    """
    model = FUGNNModel(nclass=args.nclass, 
                       nfeat=args.nfeat, 
                       nlayer=args.nlayer, 
                       hidden_dim=args.nhid, 
                       nheads=args.nhead,
                       tran_dropout=args.dropout, 
                       feat_dropout=args.dropout, 
                       prop_dropout=args.dropout, 
                       norm=args.norm).to(args.device)

    """
    Train model
    """
    trainer = FairFUGNN(model, 
                       lr=args.lr,
                       weight_decay=args.weight_decay,
                       k=args.k)
                       
    trainer.train(fair_dataset, args.epochs)

    """
    Evaluation
    """
    results = trainer.evaluate(fair_dataset)

    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']

if __name__ == '__main__':
    # Training settings
    args = args_parser()

    model_num = 1
    results = Results(args.seed_num, model_num)

    print(f"=============FUGNN Train START=============")
    print(f"Dataset: {args.dataset}")
    print(f"Hidden dim: {args.nhid}, Heads: {args.nhead}, Layers: {args.nlayer}, K: {args.k}")
    print(f"LR: {args.lr}, Dropout: {args.dropout}, Norm: {args.norm}")
    print("="*50)
    
    for seed in range(args.seed_num):
        # set seeds
        set_seed(seed)

        # running train
        results.auc[seed, 0], results.f1[seed, 0], results.acc[seed, 0], results.parity[seed, 0], \
        results.equality[seed, 0] = run(args)
        print(f"Finished seed {seed}!")

    # reporting results
    print(f"=============FUGNN Train END=============")
    results.report_results()
    if args.save_results:
        results.save_results(args)
