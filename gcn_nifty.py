import argparse
import numpy as np

import torch
import torch.nn as nn 
from torch.nn.utils import spectral_norm 
import random

import warnings
warnings.filterwarnings('ignore')

from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models import *
from graphfairness.methods import *
from graphfairness.train import *
from graphfairness.evaluation import *
from graphfairness.utils import *
from torch_geometric.nn import GCNConv


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhid', type=str, default='16', help='Number of hidden units. split by comma .')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'RobustGCN'])
    parser.add_argument('--sim_coeff', type=float, default=0.5, help='Coefficient for similarity loss.')
    parser.add_argument('--proj_hidden', type=int, default=16, help='Hidden dimension for projection head.')
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.001, help='Edge drop rate for first augmentation.')
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.001, help='Edge drop rate for second augmentation.')
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.1, help='Feature drop rate for first augmentation.')
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.1, help='Feature drop rate for second augmentation.')
    parser.add_argument('--sens_idx', type=int, default=0, help='Index of sensitive attribute in feature vector.')
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

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# Recursively applies Spectral Normalization to relevant layers in a model
def apply_spectral_norm_recursively(model):
    for name, module in model.named_children():
        # Apply to standard linear layers
        if isinstance(module, (nn.Linear)):
            if not any(isinstance(hook, torch.nn.utils.spectral_norm.SpectralNorm) for hook in module._forward_pre_hooks.values()):
                 setattr(model, name, spectral_norm(module))
        
        # Apply to GCNConv layers
        elif isinstance(module, GCNConv):
            if not hasattr(module, 'lin') or not any(isinstance(hook, torch.nn.utils.spectral_norm.SpectralNorm) for hook in module.lin._forward_pre_hooks.values()):
                if hasattr(module, 'lin') and isinstance(module.lin, nn.Linear):
                    setattr(module, 'lin', spectral_norm(module.lin))
                elif hasattr(module, 'weight'): 
                     pass 
        
        # Recurse into ModuleList containers
        elif isinstance(module, nn.ModuleList):
            for i, sub_module in enumerate(module):
                apply_spectral_norm_recursively(sub_module)
        
        # Recurse into other submodules
        else:
            apply_spectral_norm_recursively(module)


def run(args):
    """
    Load data
    """
    root = '/home/zzxie/public_data/pyg_data/FairData'
    dataset = FairDataset(root=root, name=args.dataset)
    fair_dataset = dataset.data
    fair_dataset = fair_dataset.to(args.device)

    args.nfeat = fair_dataset.features.shape[1]
    args.nclass = 1  

    """
    Build model and optimizer
    """
    model_builder = ModelBuilder(device=args.device)
    gnn_embedding_dim = args.nhid[-1] if isinstance(args.nhid, list) else args.nhid
    model = model_builder.build(model_name=args.model, 
                                nfeat=args.nfeat, 
                                nclass=gnn_embedding_dim) 

    print(f"Applying Lipschitz-based Spectral Normalization to {args.model} layers...")
    apply_spectral_norm_recursively(model)

    """
    Train model
    """
    nifty = NIFTY(model, 
                  nfeat=args.nfeat, 
                  nclass=args.nclass, 
                  nhid=args.nhid, 
                  dropout=args.dropout,
                  sim_coeff=args.sim_coeff,
                  proj_hidden=args.proj_hidden,
                  drop_edge_rate_1=args.drop_edge_rate_1,
                  drop_edge_rate_2=args.drop_edge_rate_2,
                  drop_feature_rate_1=args.drop_feature_rate_1,
                  drop_feature_rate_2=args.drop_feature_rate_2)
    
    nifty.train(fair_dataset, args.epochs, validation=True, sens_idx=args.sens_idx)

    """
    evaluation
    """
    results = nifty.evaluate(fair_dataset)

    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']


if __name__ == '__main__':
    # Training settings
    args = args_parser()

    model_num = 1
    results = Results(args.seed_num, model_num)

    print(f"=============Train START=============")
    print(f"Sensitive attribute index: {args.sens_idx}")
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