
import argparse
import numpy as np
import torch
import random
import warnings


from graphfairness.datasets import FairDataset
from graphfairness.models import ModelBuilder
from graphfairness.methods.preprocess import EDITS
from graphfairness.evaluation import *
from graphfairness.utils import *

warnings.filterwarnings('ignore')

def args_parser():
    parser = argparse.ArgumentParser(description="Run EDITS")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=1, help='The number of random seeds.')
    parser.add_argument('--dataset', type=str, default='german', choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'], help='Dataset')
    
    # GCN parameters
    parser.add_argument('--gcn_epochs', type=int, default=1000, help='Number of epochs to train the downstream GCN.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for GCN.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for GCN.')
    parser.add_argument('--nhid', type=str, default='16', help='Number of hidden units for GCN.')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate for GCN.')
    parser.add_argument('--model', type=str, default='gcn', help='Downstream classifier model.')
    
    # EDITS parameters
    parser.add_argument('--edits_epochs', type=int, default=500, help='Number of epochs for EDITS pre-processing.')
    parser.add_argument('--lr_edits', type=float, default=0.003, help='Learning rate for EDITS.')
    parser.add_argument('--wd_edits', type=float, default=1e-7, help='Weight decay for EDITS.')                                            
    parser.add_argument('--adj_lambda', type=float, default=1e-1, help='Adjacency matrix regularization coefficient for EDITS.')
    parser.add_argument('--layer_threshold', type=int, default=2, help='Graph propagation layers for EDITS.')
    parser.add_argument('--threshold', type=float, default=0.29, help='Threshold for sparsifying the learned adjacency matrix.')
    
    parser.add_argument('--save_results', action='store_true', default=False, help='Save the results.')
    parser.add_argument('--pruning_method', type=str, default='absolute', choices=['absolute', 'relative'], 
                    help="Pruning method for the learned adj matrix. 'absolute' uses a fixed threshold, 'relative' uses the original binarize logic.")#新加的参数，效果不好就去掉
    parser.add_argument('--threshold_prop', type=float, default=0.015, help="Threshold proportion for 'relative' pruning (e.g., bail: 0.015, german: 0.29).")#新加的参数，效果不好就去掉
    parser.add_argument('--recon_weight', type=float, help='Weight for feature reconstruction loss.')
    parser.add_argument('--adv_weight_feat', type=float, help='Weight for feature adversarial loss.')
    parser.add_argument('--fro_weight', type=float, default=1.0, help='Weight for structural similarity loss (Frobenius norm).')
    parser.add_argument('--adv_weight_adj', type=float, default=15.0, help='Weight for structural adversarial loss.')
    
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
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
    root = '/home/zmzhang/public_data/pyg_data/FairData' 
    dataset = FairDataset(root=root, name=args.dataset)
    data = dataset.data
    data = data.to(args.device)

    args.nfeat = data.features.shape[1]
    args.nclass = 1
    args.node_num = data.features.shape[0]
    args.dropout_edits = 0.2  # EDITS dropout
    """
    Build model
    """
    model_builder = ModelBuilder(device=args.device)
    model = model_builder.build(model_name=args.model,
                                nfeat=args.nfeat,
                                nclass=args.nclass,
                                nhid=args.nhid,
                                dropout=args.dropout)

    """
    Train model using EDITS method
    """
    cfg_dict = vars(args).copy()
    if 'model' in cfg_dict:
        del cfg_dict['model']
    
    trainer = EDITS(model, **cfg_dict)
    trainer.train(data, epochs=args.gcn_epochs, edits_epochs=args.edits_epochs)

    """
    Evaluation
    """
    results = trainer.evaluate(data)
    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']


if __name__ == '__main__':
    args = args_parser()
    
    model_num = 1
    results = Results(args.seed_num, model_num)

    print("============= Train START =============")
    for seed in range(args.seed_num):
        set_seed(seed)
        
        results.auc[seed, 0], results.f1[seed, 0], results.acc[seed, 0], results.parity[seed, 0], \
        results.equality[seed, 0] = run(args)
        print(f"Finish seed {seed}!")
    
    print("============= Train END =============")
    results.report_results()
    if args.save_results:
        results.save_results(args)