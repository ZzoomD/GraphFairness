import argparse
import numpy as np
import torch
import random
import warnings

warnings.filterwarnings('ignore')

from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models import * 
from graphfairness.methods.preprocess.Graphair.graphair import Graphair 
from graphfairness.train import *
from graphfairness.evaluation import *
from graphfairness.utils import *
from graphfairness.methods.preprocess.Graphair.graphair_components import aug_module, GCN_Body

def args_parser():
    parser = argparse.ArgumentParser(description="Run Graphair Experiment")

    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed runs.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save results to a file.')
    parser.add_argument('--dataset', type=str, default='nba', choices=['nba', 'pokec_z', 'pokec_n', 'credit', 'german', 'bail'])
    
    
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs for Graphair.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for Graphair main training.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Graphair main training.')
    
    
    parser.add_argument('--model_nhid', type=int, nargs='+', default=[64, 64, 64], help='Hidden layer dims for f_encoder GCN (e.g., 64 64 for 2 layers).')
    parser.add_argument('--adv_nhid', type=int, nargs='+', default=[64, 64], help='Hidden layer dims for adversary GCN.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for GCNs in Graphair.')
    
    
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for adversarial loss.')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for contrastive loss.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Weight for reconstruction loss.')
    parser.add_argument('--lam', type=float, default=10.0, help='Weight for feature reconstruction within reconstruction loss.')
    parser.add_argument('--proj_hidden_dim', type=int, default=64, help='Hidden dimension of the projection head.')

    
    parser.add_argument('--test_epochs', type=int, default=500, help='Number of epochs for downstream classifier.')
    parser.add_argument('--classifier_hidden_dim', type=int, default=128, help='Hidden dimension of the downstream classifier.')
    parser.add_argument('--classifier_lr', type=float, default=1e-3, help='Learning rate for the downstream classifier.')
    parser.add_argument('--classifier_wd', type=float, default=1e-5, help='Weight decay for the downstream classifier.')
    parser.add_argument('--warmup', type=int, default=50, help='Number of warmup epochs for augmentation module.')

    
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    return args

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False   

def run(args):
    """
    Load data
    """
   
    root = '/home/zmzhang/public_data/pyg_data/FairData' 
    dataset = FairDataset(root=root, name=args.dataset)
    data = dataset.data
    data = data.to(args.device)

    args.nfeat = data.features.shape[1]
    args.nclass = 1 

    """
    Build model
    """
   
    
    """
    Train model
    """
    
    
    
  
    cfg = {
        'nfeat': args.nfeat,
        'model_nhid': args.model_nhid,
        'adv_nhid': args.adv_nhid,
        'proj_hidden_dim': args.proj_hidden_dim,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'classifier_lr': args.classifier_lr,
        'classifier_wd': args.classifier_wd,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'device': args.device,
        'dataset': args.dataset
    }
    
    
    method = Graphair(model = None,**cfg)
    
    
    results = method.train(
        data, 
        args.epochs, 
        
        alpha=args.alpha, 
        beta=args.beta, 
        gamma=args.gamma, 
        lam=args.lam,
        test_epochs=args.test_epochs,
        warmup=args.warmup
    )

    """
    evaluation
    """

    
    #print("\n--- Results for this run ---")
    #print(f"AUC: {results['auc_mean']:.4f}, F1: {results['f1_mean']:.4f}, ACC: {results['acc_mean']:.4f}, DP: {results['dp_mean']:.4f}, EO: {results['eo_mean']:.4f}")

    return results['auc_mean'], results['f1_mean'], results['acc_mean'], results['dp_mean'], results['eo_mean']


if __name__ == '__main__':
    
    args = args_parser()
    print("Running with the following arguments:")
    print(args)

    
    results = Results(args.seed_num, 1) 

    print(f"\n============= Starting Training on {args.dataset.upper()} for {args.seed_num} seeds =============")
    for seed in range(args.seed_num):
        print(f"\n--- Running Seed {seed+1}/{args.seed_num} ---")
        
        set_seed(seed*10, args.cuda)

        
        auc, f1, acc, dp, eo = run(args)
        
        
        results.auc[seed, 0] = auc
        results.f1[seed, 0] = f1
        results.acc[seed, 0] = acc
        results.parity[seed, 0] = dp
        results.equality[seed, 0] = eo

    
    print(f"\n============= Training Finished =============")
    results.report_results()

    
    if args.save_results:
        
        results.save_results(args)