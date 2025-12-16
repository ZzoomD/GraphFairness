import argparse
import numpy as np
import torch
import random
import warnings
from torch_sparse import SparseTensor
warnings.filterwarnings('ignore')

from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models import *
from graphfairness.methods import *
from graphfairness.utils import *

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')  
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--nhid', type=str, default='32', help='Number of hidden units.') 
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate.')  
    parser.add_argument('--dataset', type=str, default='pokec_n',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gin'])
    
    # FairSIN specific arguments 
    parser.add_argument('--delta', type=float, default=0.8, help='Strength of feature neutralization.')
    parser.add_argument('--beta', type=float, default=0.1, help='Adversarial training coefficient.')
    parser.add_argument('--m_epochs', type=int, default=300, help='Epochs for estimator pre-training.')
    parser.add_argument('--m_hidden', type=int, default=32, help='Hidden dimension for estimator.')
    parser.add_argument('--m_lr', type=float, default=0.005, help='Learning rate for estimator pre-training.')
    
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False


def run(args):
    """
    Load data
    """
    root = '/home/zzxie/public_data/pyg_data/FairData' 
    dataset = FairDataset(root=root, name=args.dataset)
    fair_dataset = dataset.data
    fair_dataset = fair_dataset.to(args.device)
    

    args.nfeat = fair_dataset.features.shape[1]
    args.nclass = 1  # Binary classification for fairness datasets typically

    """
    Build model and optimizer
    """
    model_builder = ModelBuilder(device=args.device)
    model = model_builder.build(model_name=args.model, 
                                nfeat=args.nfeat, 
                                nhid=args.nhid,
                                nclass=args.nclass,
                                dropout=args.dropout,
                                )

    """
    Train model
    """
    # Create FairSIN instance
    fairsin = FairSIN(model, 
                      nfeat=args.nfeat, 
                      nhid=args.nhid, 
                      lr=args.lr, 
                      weight_decay=args.weight_decay, 
                      m_hidden=args.m_hidden,
                      m_lr=args.m_lr)
    
    
    # Train
    fairsin.train(fair_dataset, 
                  epochs=args.epochs, 
                  m_epochs=args.m_epochs,
                  delta=args.delta, 
                  beta=args.beta)


    results = fairsin.evaluate(fair_dataset, delta=args.delta)

    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']

if __name__ == '__main__':
    # Training settings
    args = args_parser()

    model_num = 1
    results = Results(args.seed_num, model_num)

    print(f"=============Train START ({args.dataset} - {args.model})=============")
    for seed in range(args.seed_num):
        set_seed(seed)
        
        auc, f1, acc, dp, eo = run(args)
        
        results.auc[seed, 0] = auc
        results.f1[seed, 0] = f1
        results.acc[seed, 0] = acc
        results.parity[seed, 0] = dp
        results.equality[seed, 0] = eo
        
        print(f"Seed {seed}: AUC={auc:.4f}, F1={f1:.4f}, ACC={acc:.4f}, DP={dp:.4f}, EO={eo:.4f}")

    # reporting results
    print(f"=============Train END=============")
    results.report_results()
    if args.save_results:
        results.save_results(args)