#%%
import argparse
import numpy as np
import torch
import random
import warnings
import os
warnings.filterwarnings('ignore')
from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models import *
from graphfairness.methods.inprocess.fairsad import FairSAD
from graphfairness.train import *
from graphfairness.evaluation import *
from graphfairness.utils import *

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhid', type=str, default='16', help='Number of hidden units. split by comma .')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument('--model', type=str, default='fairsad',
                        choices=['gcn', 'sage', 'gin', 'fairsad'])
    parser.add_argument('--save_results', type=bool, default=False)
    
    # FairSAD specificed hyperpararmeters
    parser.add_argument('--alpha', type=float, default=0.1, help='empower coefficient for disentanglement.')
    parser.add_argument('--beta', type=float, default=0.1, help='empower coefficient for decorrelation.')
    parser.add_argument('--channels', type=int, default=4, help='Number of disentangled channels.')
    
    
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # set device
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
    # FairSAD relies on the DisGCN backbone to identify latent factors
    model_builder = ModelBuilder(device=args.device)
    model = model_builder.build(model_name=args.model, 
                                nfeat=args.nfeat, 
                                nclass=args.nclass,
                                nhid=args.nhid,
                                dropout=args.dropout,
                                channels=args.channels # FairSAD model args
                                )
    
    """
    Train model
    """
    fairsad = FairSAD(model, nfeat=args.nfeat, nhid=args.nhid, nclass=args.nclass, 
                              dropout=args.dropout, channels=args.channels)
    fairsad.train(fair_dataset, args.epochs, validation=True, 
                          alpha=args.alpha, beta=args.beta)
    
    """
    Evaluation
    """
    results = fairsad.evaluate(fair_dataset)
    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']

if __name__ == '__main__':
    # Training settings
    args = args_parser()
    model_num = 1
    results = Results(args.seed_num, model_num)
    
    print(f"=============Train START=============")
    for seed in range(args.seed_num):
        # set seeds
        set_seed(seed)
        # running train
        results.auc[seed, 0], results.f1[seed, 0], results.acc[seed, 0], results.parity[seed, 0], \
        results.equality[seed, 0] = run(args)
        print(f"finish seed {seed}!")
        
    # reporting results
    print(f"=============Train END=============")
    results.report_results()
    
    if args.save_results:
        results.save_results(args)