#%%
import argparse
import numpy as np
import torch
import random
import warnings
import os

warnings.filterwarnings('ignore')

# 算法库标准组件导入
from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models import *
from graphfairness.methods.inprocess.fairadj import FairAdj, FairAdjVAE
from graphfairness.train import *
from graphfairness.evaluation import *
from graphfairness.utils import *

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    
   
    parser.add_argument('--outer_epochs', type=int, default=4, help='Number of outer co-adaptation epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    
   
    parser.add_argument('--eta', type=float, default=0.2, help='Learning rate for adjacency matrix.')
    parser.add_argument('--T1', type=int, default=50, help='Inner utility epochs.')
    parser.add_argument('--T2', type=int, default=5, help='Inner fairness epochs.')
    parser.add_argument('--hidden1', type=int, default=32, help='VAE hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='VAE hidden layer 2 (Embedding size).')
    
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument('--save_results', type=bool, default=False)
 

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
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
    fair_dataset = dataset.data
    

    mean = fair_dataset.features.mean(dim=0)
    std = fair_dataset.features.std(dim=0) + 1e-6
    fair_dataset.features = (fair_dataset.features - mean) / std
        
    fair_dataset = fair_dataset.to(args.device)

    args.nfeat = fair_dataset.features.shape[1]
    args.nclass = 1

    """
    Build model
    """

    model = FairAdjVAE(
        nfeat=args.nfeat, 
        nhid1=args.hidden1, 
        nhid2=args.hidden2, 
        dropout=args.dropout, 
        nclass=args.nclass
    ).to(args.device)

    """
    Train model
    """

    fairadj_trainer = FairAdj(
        model, 
        lr=args.lr, 
        eta=args.eta, 
        
        T1=args.T1, 
        T2=args.T2, 
        outer_epochs=args.outer_epochs,
        device=args.device,
        weight_path=f'./weights/best_fairadj_{args.dataset}_seed.pt'
    )
    

    fairadj_trainer.train(fair_dataset, epochs=None, validation=True)

    """
    evaluation
    """

    results = fairadj_trainer.evaluate(fair_dataset)

    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']

if __name__ == '__main__':
   
    args = args_parser()

    model_num = 1 
    results = Results(args.seed_num, model_num)

    print(f"============= FairAdj Train START =============")
    for seed in range(args.seed_num):
        
        set_seed(seed)
        
        
        auc, f1, acc, dp, eo = run(args)
        
        
        results.auc[seed, 0] = auc
        results.f1[seed, 0] = f1
        results.acc[seed, 0] = acc
        results.parity[seed, 0] = dp
        results.equality[seed, 0] = eo
        
        print(f"Finish seed {seed} | AUC: {auc:.4f} | F1: {f1:.4f} | ACC: {acc:.4f} | DP: {dp:.4f} | EO: {eo:.4f}")

  
    print(f"============= FairAdj Train END =============")
    results.report_results()
    
    if args.save_results:
        results.save_results(args)