import argparse
import numpy as np
import torch
import random
import warnings
import os
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
    parser.add_argument('--nhid', type=str, default='16', help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gin'])
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use.')

    # FairGB specific arguments
    parser.add_argument('--eta', type=float, default=0.5, help='Hyperparameter for mixup ratio.')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Trade-off parameter.')
    
    parser.add_argument('--save_results', type=bool, default=False)

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    if args.cuda:
        if args.gpu >= 0 and args.gpu < torch.cuda.device_count():
            args.device = torch.device(f'cuda:{args.gpu}')
        else:
            args.device = torch.device('cuda:0')
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
    torch.backends.cudnn.allow_tf32 = False

def run(args):
    # Load data
    root = '/home/zzxie/public_data/pyg_data/FairData' 
    dataset = FairDataset(root=root, name=args.dataset)
    fair_dataset = dataset.data.to(args.device)

    #Feature Standardization for German, Credit, NBA datasets
    if args.dataset in ['german', 'credit', 'nba']:
        print(f"Standardizing features for {args.dataset}...")
        features = fair_dataset.features.float()
        # Use training stats only to prevent leakage
        train_mask = fair_dataset.idx_train
        mean = features[train_mask].mean(dim=0)
        std = features[train_mask].std(dim=0)
        std[std == 0] = 1.0 
        fair_dataset.features = (features - mean) / std

    args.nfeat = fair_dataset.features.shape[1]
    args.nclass = 1 

    # Build model
    model_builder = ModelBuilder(device=args.device)
    model = model_builder.build(model_name=args.model, 
                                nfeat=args.nfeat, 
                                nhid=args.nhid,
                                nclass=args.nclass,
                                dropout=args.dropout)

    # Train
    fairgb = FairGB(model, 
                    lr=args.lr, 
                    weight_decay=args.weight_decay,
                    eta=args.eta,
                    warmup=args.warmup,
                    alpha=args.alpha)
    
    fairgb.train(fair_dataset, 
                 epochs=args.epochs,
                 eta=args.eta,
                 warmup=args.warmup)

    # Evaluation
    results = fairgb.evaluate(fair_dataset, split='test')

    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']

if __name__ == '__main__':
    # Training settings
    args = args_parser()
    model_num = 1
    results = Results(args.seed_num, model_num)

    print(f"=============Train START ({args.dataset} - {args.model})=============")
    for seed in range(args.seed_num):
        # set seeds
        set_seed(seed)
        auc, f1, acc, dp, eo = run(args)
        
        results.auc[seed, 0] = auc
        results.f1[seed, 0] = f1
        results.acc[seed, 0] = acc
        results.parity[seed, 0] = dp
        results.equality[seed, 0] = eo
        
        print(f"Seed {seed}: AUC={auc:.4f}, F1={f1:.4f}, ACC={acc:.4f}, DP={dp:.4f}, EO={eo:.4f}")

    print(f"=============Train END=============")
    results.report_results()
    if args.save_results:
        results.save_results(args)