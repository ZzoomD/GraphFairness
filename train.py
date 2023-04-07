#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from Datasets import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
import torch.nn as nn
from torch_sparse import SparseTensor


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


def run(args):
    """
    Load data
    """
    fair_dataset = FairDataset(args.dataset)
    fair_dataset.load_data()


    # The number of classier
    # num_class = labels.unique().shape[0] - 1
    num_class = 1

    # """
    # Build model and optimizer
    # """
    # # Synthetic teacher model and optimizer
    # syn_t = SynTeacher(in_dim=features.shape[1], hid_dim=args.hidden, out_dim=num_class)
    # # optimizer_t = optim.Adam(syn_t.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer_mlp = optim.Adam(syn_t.para_mlp, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer_gnn = optim.Adam(syn_t.para_gnn, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer_proj = optim.Adam(syn_t.para_proj, lr=args.lr, weight_decay=args.weight_decay)
    # syn_t = syn_t.to(args.device)
    #
    # # Student model and optimizer
    # if args.model == 'gcn':
    #     # student model
    #     model = GCN(nfeat=features.shape[1],
    #                 nhid=args.hidden,
    #                 nclass=num_class,
    #                 dropout=args.dropout)
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     model = model.to(args.device)
    #
    #     # vanilla model (for synthetic teacher)
    #     model_van = GCN(nfeat=features.shape[1],
    #                     nhid=args.hidden,
    #                     nclass=num_class,
    #                     dropout=args.dropout)
    #     optimizer_van = optim.Adam(model_van.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     model_van = model_van.to(args.device)
    #
    #     gnn_van = GCN(nfeat=features.shape[1],
    #                   nhid=args.hidden,
    #                   nclass=num_class,
    #                   dropout=args.dropout)
    #     optimizer_gvan = optim.Adam(gnn_van.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     gnn_van = gnn_van.to(args.device)
    #
    # elif args.model == 'sage':
    #     model = SAGE(nfeat=features.shape[1],
    #                  nhid=args.hidden,
    #                  nclass=num_class,
    #                  dropout=args.dropout)
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     model = model.to(args.device)
    #
    #     # vanilla model (for synthetic teacher)
    #     model_van = SAGE(nfeat=features.shape[1],
    #                      nhid=args.hidden,
    #                      nclass=num_class,
    #                      dropout=args.dropout)
    #     optimizer_van = optim.Adam(model_van.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     model_van = model_van.to(args.device)
    #
    #     gnn_van = SAGE(nfeat=features.shape[1],
    #                    nhid=args.hidden,
    #                    nclass=num_class,
    #                    dropout=args.dropout)
    #     optimizer_gvan = optim.Adam(gnn_van.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     gnn_van = gnn_van.to(args.device)
    #
    # elif args.model == 'gin':
    #     model = GIN(nfeat=features.shape[1],
    #                 nhid=args.hidden,
    #                 nclass=num_class,
    #                 dropout=args.dropout)
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     model = model.to(args.device)
    #
    #     # vanilla model (for synthetic teacher)
    #     model_van = GIN(nfeat=features.shape[1],
    #                     nhid=args.hidden,
    #                     nclass=num_class,
    #                     dropout=args.dropout)
    #     optimizer_van = optim.Adam(model_van.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     model_van = model_van.to(args.device)
    #
    #     gnn_van = GIN(nfeat=features.shape[1],
    #                   nhid=args.hidden,
    #                   nclass=num_class,
    #                   dropout=args.dropout)
    #     optimizer_gvan = optim.Adam(gnn_van.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     gnn_van = gnn_van.to(args.device)
    #
    # """
    # Train model (Teacher model and Student model)
    # """
    # t_total = time.time()
    #
    # # training vanilla model (for synthetic teacher)
    # criterion_van = torch.nn.BCEWithLogitsLoss()
    # train_teacher(model_van, optimizer_van, criterion_van, args.epochs, data, save_name=f'{args.model}_vanilla.pt',
    #               model_type='gnn')
    #
    # # obtain nodes embedding from gnn vanilla
    # model_van.load_state_dict(torch.load(f'{args.model}_vanilla.pt'))
    # model_van.eval()
    # with torch.no_grad():
    #     h_van, output_van = model_van(features, edge_index.to(args.device))
    # # output_soft_van = torch.softmax(output_van, dim=1)
    # # criterion_soft = torch.nn.MSELoss(reduction='mean')
    #
    # # train synthetic teacher model
    # # criterion_t = CoLoss(tem=args.tem)
    # # criterion_extra = torch.nn.BCEWithLogitsLoss()
    # # train_teacher(syn_t, optimizer_t, criterion_t, epochs=1000, data=data, save_name='synthetic_teacher.pt',
    # #               model_type='syn', h_labels=h_van, criterion_extra=criterion_extra, criterion_soft=criterion_soft,
    # #               soft_labels=output_soft_van)
    #
    # # train two experts
    # criterion = torch.nn.BCEWithLogitsLoss()
    # h_fair_mlp = syn_t.train_expert_mlp(optimizer=optimizer_mlp, criterion=criterion, epochs=args.epochs, data=data)
    # h_fair_gnn = syn_t.train_expert_gnn(optimizer=optimizer_gnn, criterion=criterion, epochs=args.epochs, data=data)
    #
    # # train projector
    # projecter_input = torch.cat((h_fair_mlp, h_fair_gnn), 1)
    # criterion_proj = CoLoss(tem=args.tem)
    # h_fair = syn_t.train_projector(optimizer=optimizer_proj, criterion=criterion_proj, epochs=args.epochs, data=data,
    #                                input=projecter_input, label=h_van)
    #
    # # # distill knowledge from synthetic teacher
    # # syn_t.load_state_dict(torch.load('synthetic_teacher.pt'))
    # # syn_t.eval()
    # # with torch.no_grad():
    # #     h_fair = syn_t.distill(features, edge_index, features_ones)
    #
    # # training student model
    # criterion_bce = torch.nn.BCEWithLogitsLoss()
    # criterion_kd = KDLoss(loss_type='col', tem=args.tem)
    # train_student(model, optimizer, criterion_bce, criterion_kd, args, data,
    #               save_name=f'{args.model}_student.pt', soft_target_mlp=h_fair,
    #               soft_target_gnn=h_fair)
    #
    # # train gnn vanilla
    # criterion_gvan = torch.nn.BCEWithLogitsLoss()
    # train_teacher(gnn_van, optimizer_gvan, criterion_gvan, args.epochs, data, save_name=f'{args.model}_original.pt',
    #               model_type='gnn')
    #
    # """
    # evaluation
    # """
    # auc_roc_test, f1_s, acc, parity, equality = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
    # # before teaching
    # auc_roc_test[0], f1_s[0], acc[0], parity[0], equality[0] = evaluation(gnn_van, f'{args.model}_original.pt', data,
    #                                                                       model_type='gnn')
    # # after teaching
    # auc_roc_test[1], f1_s[1], acc[1], parity[1], equality[1] = evaluation(model, f'{args.model}_student.pt', data,
    #                                                                       model_type='gnn')
    #
    # return auc_roc_test, f1_s, acc, parity, equality
    return 0, 0, 0, 0, 0


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument("--num_heads", type=int, default=1, help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn'])
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--tem', type=float, default=0.5, help='the temperature of contrastive learning loss '
                                                               '(mutual information maximize)')
    parser.add_argument('--alpha', type=float, default=0.25, help='empower coefficient')
    parser.add_argument('--lr_w', type=float, default=1,
                        help='the learning rate of the adaptive weight coefficient')

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    auc, f1, acc, parity, equality = np.zeros(shape=(args.seed_num, 2)), np.zeros(shape=(args.seed_num, 2)), \
                                     np.zeros(shape=(args.seed_num, 2)), np.zeros(shape=(args.seed_num, 2)), \
                                     np.zeros(shape=(args.seed_num, 2))

    for seed in range(args.seed_num):
        # set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        # torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        auc[seed, :], f1[seed, :], acc[seed, :], parity[seed, :], equality[seed, :] = run(args)

        print(f"========finish seed {seed}========")

    # print report
    print("=================START=================")
    print(f"Parameter:τ={args.tem}, γ={args.alpha}, lr_w={args.lr_w}")
    print(f"============" + "before teaching" + "============")
    print(f"AUCROC: {np.around(np.mean(auc[:, 0]) * 100, 2)} ± {np.around(np.std(auc[:, 0]) * 100, 2)}")
    print(f'F1-score: {np.around(np.mean(f1[:, 0]) * 100, 2)} ± {np.around(np.std(f1[:, 0]) * 100, 2)}')
    print(f'ACC: {np.around(np.mean(acc[:, 0]) * 100, 2)} ± {np.around(np.std(acc[:, 0]) * 100, 2)}')
    print(f'Parity: {np.around(np.mean(parity[:, 0]) * 100, 2)} ± {np.around(np.std(parity[:, 0]) * 100, 2)}')
    print(f'Equality: {np.around(np.mean(equality[:, 0]) * 100, 2)} ± {np.around(np.std(equality[:, 0]) * 100, 2)}')

    print(f"============" + "after teaching" + "============")
    print(f"AUCROC: {np.around(np.mean(auc[:, 1]) * 100, 2)} ± {np.around(np.std(auc[:, 1]) * 100, 2)}")
    print(f'F1-score: {np.around(np.mean(f1[:, 1]) * 100, 2)} ± {np.around(np.std(f1[:, 1]) * 100, 2)}')
    print(f'ACC: {np.around(np.mean(acc[:, 1]) * 100, 2)} ± {np.around(np.std(acc[:, 1]) * 100, 2)}')
    print(f'Parity: {np.around(np.mean(parity[:, 1]) * 100, 2)} ± {np.around(np.std(parity[:, 1]) * 100, 2)}')
    print(f'Equality: {np.around(np.mean(equality[:, 1]) * 100, 2)} ± {np.around(np.std(equality[:, 1]) * 100, 2)}')

    print("=================END=================")
    with open(f"{args.dataset}_tao{args.tem}.txt", 'a') as f:
        f.write(f"τ={args.tem}, γ={args.alpha}, lr_w={args.lr_w}\n")
        f.write(f"AUCROC: {np.around(np.mean(auc[:, 1]) * 100, 2)} ± {np.around(np.std(auc[:, 1]) * 100, 2)}\n")
        f.write(f'F1-score: {np.around(np.mean(f1[:, 1]) * 100, 2)} ± {np.around(np.std(f1[:, 1]) * 100, 2)}\n')
        f.write(f'ACC: {np.around(np.mean(acc[:, 1]) * 100, 2)} ± {np.around(np.std(acc[:, 1]) * 100, 2)}\n')
        f.write(f'Parity: {np.around(np.mean(parity[:, 1]) * 100, 2)} ± {np.around(np.std(parity[:, 1]) * 100, 2)}\n')
        f.write(f'Equality: {np.around(np.mean(equality[:, 1]) * 100, 2)} ± {np.around(np.std(equality[:, 1]) * 100, 2)}\n')
    f.close()

