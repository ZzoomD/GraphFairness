import argparse
import numpy as np
import torch
import random
import warnings
import os

warnings.filterwarnings('ignore')

from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models import ModelBuilder
from graphfairness.methods import FairINV 
from graphfairness.utils import BunchDict, Results
from graphfairness.evaluation import fair_metric

def args_parser():
    parser = argparse.ArgumentParser(description="FairINV Example")
    
    # General training settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train for SIL stage.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for the main model.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhid', type=str, default='16', help='Number of hidden units. split by comma.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'],
                        help='Dataset name.')
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gin'],
                        help='Backbone model name.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Whether to save the results.')
    
    # FairINV specific hyperparameters
    parser.add_argument('--env_num', type=int, default=2, 
                        help='Number of environments for partition (t in paper).')
    parser.add_argument('--partition_times', type=int, default=3, 
                        help='Number of times to perform environment partition (k in paper).')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Balance coefficient for mean loss in SIL stage.')
    parser.add_argument('--lr_sp', type=float, default=0.01, 
                        help='Learning rate for the SAP (Sensitive Attribute Partition) module.')

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    args.nhid = [int(x.strip()) for x in args.nhid.split(',')]
    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
    model_builder = ModelBuilder(args.device)
    model = model_builder.build(model_name=args.model,
                                nfeat=args.nfeat,
                                nclass=args.nclass,
                                nhid=args.nhid,
                                dropout=args.dropout)

    """
    Train model
    """
    fair_inv = FairINV(model=model,
                       nfeat=args.nfeat,
                       nclass=args.nclass,
                       nhid=args.nhid,
                       lr=args.lr,
                       weight_decay=args.weight_decay,
                       env_num=args.env_num,
                       partition_times=args.partition_times,
                       alpha=args.alpha,
                       lr_sp=args.lr_sp,
                      )
    fair_inv.train(fair_dataset, args.epochs, validation=True)

    """
    Evaluation
    """
    results = fair_inv.evaluate(fair_dataset)
    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']

if __name__ == '__main__':
    # Parse command line arguments
    args = args_parser()
    
    # Results collector
    model_num = 1
    results = Results(args.seed_num, model_num)
    
    print(f"============= FairINV Train START (Dataset: {args.dataset}) =============")
    for seed in range(args.seed_num):
        # Set random seeds for reproducibility
        set_seed(seed)
        
        # Run training and evaluation for the current seed
        results.auc[seed, 0], results.f1[seed, 0], results.acc[seed, 0], results.parity[seed, 0], \
        results.equality[seed, 0] = run(args)
        
        print(f"finish seed {seed}!")
        
    # Final reporting
    print(f"============= Train END =============")
    results.report_results()
    
    if args.save_results:
        results.save_results(args)