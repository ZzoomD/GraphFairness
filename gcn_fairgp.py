#%%
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import random
import warnings
warnings.filterwarnings('ignore')

from graphfairness.datasets.fair_datasets import FairDataset
from graphfairness.models.fairgp import FairGP as FairGPModel
from graphfairness.methods.inprocess.fairgp import FairGP as FairGPMethod
from graphfairness.utils import Results

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhid', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='credit',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--nlayer', type=int, default=2, help='Number of transformer layers.')
    
    # FairGP specific args
    parser.add_argument('--pe_method', type=str, default='adj', choices=['adj','lap','none'])
    parser.add_argument('--pe_dim', type=int, default=2, help='positional encoding dim')
    parser.add_argument('--patch_method', type=str, default='metis', choices=['metis', 'louvain', 'random', 'leiden'])
    parser.add_argument('--n_patch', type=int, default=100, help='number of patches')
    
    parser.add_argument('--save_results', type=bool, default=False)

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # FairGP adds a virtual node, so we need to account for it + handle dynamic adding in Trainer/Utils.
    # But here we pass original nodes num to Model Init, wait.
    # In FairGP original code:
    # model init uses num_nodes (original)
    # But Trainer logic adds virtual node.
    # Actually, FairGP model `ICABlock` has: `patch_mask = (patch != self.num_nodes - 1)`
    # The `partition_patch` returns `num_nodes` increased by 1.
    # If we pass original `num_nodes` to `FairGPModel`, then `self.num_nodes - 1` would be `ori - 1`.
    # But the virtual node index is `ori`.
    # So we should pass `ori + 1` to `FairGPModel`?
    # Let's check original logic in run_models.py:
    # patch, ..., num_nodes = partition_patch(..., num_nodes) -> num_nodes increases by 1
    # model = FairGP(num_nodes=num_nodes, ...) -> Uses increased num_nodes
    
    # So we need to calculate final num_nodes before creating model?
    # OR we let Trainer handle everything. But Model needs to be created before Trainer usually.
    # AND Trainer needs model instance.
    
    # Solution: We can anticipate the +1.
    num_nodes_final = fair_dataset.features.shape[0] + 1
    nfeat_final = args.nfeat + (args.pe_dim if args.pe_method != 'none' else 0)

    """
    Build model
    """
    model = FairGPModel(num_nodes=num_nodes_final, 
                        in_channels=nfeat_final, 
                        hidden_channels=args.nhid, 
                        out_channels=2, 
                        activation=F.relu,
                        n_head=args.nhead,
                        layers=args.nlayer,
                        dropout1=args.dropout).to(args.device)

    """
    Train model
    """
    fairgp = FairGPMethod(model, 
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    pe_method=args.pe_method,
                    pe_dim=args.pe_dim,
                    patch_method=args.patch_method,
                    n_patch=args.n_patch,
                    num_nodes=fair_dataset.features.shape[0]) # Pass original num nodes just in case
                    
    fairgp.train(fair_dataset, args.epochs)

    """
    Evaluation
    """
    results = fairgp.evaluate(fair_dataset)

    return results['auc'], results['f1'], results['acc'], results['dp'], results['eo']


if __name__ == '__main__':
    # Training settings
    args = args_parser()

    model_num = 1
    results = Results(args.seed_num, model_num)

    print(f"=============FairGP Train START=============")
    print(f"Dataset: {args.dataset}")
    print(f"PE Method: {args.pe_method}, Dim: {args.pe_dim}")
    print(f"Patch: {args.patch_method}, Num: {args.n_patch}")
    print("="*50)
    
    for seed in range(args.seed_num):
        # set seeds
        set_seed(seed)

        # running train
        results.auc[seed, 0], results.f1[seed, 0], results.acc[seed, 0], results.parity[seed, 0], \
        results.equality[seed, 0] = run(args)
        print(f"Finished seed {seed}!")

    # reporting results
    print(f"=============FairGP Train END=============")
    results.report_results()
    if args.save_results:
        results.save_results(args)
