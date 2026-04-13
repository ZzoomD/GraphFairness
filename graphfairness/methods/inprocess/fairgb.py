import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add
from tqdm import tqdm
import os

from graphfairness.train import Trainer
from graphfairness.utils import BunchDict
from graphfairness.evaluation.metrics import fair_metric
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Tiny constant for numerical stability
EPS = 1e-6

class FairGB(Trainer):
    r"""Implementation of `FairGB` from the paper entitled `"Rethinking Fair Graph Neural Networks from Re-balancing"<https://arxiv.org/pdf/2407.11624>`.

    FairGB addresses unfairness in Graph Neural Networks through group re-balancing with two key components:
    1. Counterfactual Node Mixup (CNM): Mixes ego-networks across demographic groups to create balanced augmented graphs
    2. Contribution Alignment Loss (CAL): Re-weights groups based on gradient contributions to achieve balanced learning

    The method is based on the insight that group imbalance is a primary source of unfairness in GNNs.
    CNM generates synthetic nodes by interpolating both features and neighbor distributions of counterfactual pairs
    (inter-domain: same label, different sensitive attribute; inter-class: different label, same sensitive attribute).
    CAL then balances the contribution of each demographic group by weighting loss terms according to gradient magnitudes.

    Parameters
    ----------
    model : nn.Module
        The GNN backbone model for classification
    **cfg : dict
        Configuration parameters including:
        - lr : float, optional
            Learning rate for optimization, by default 1e-3
        - weight_decay : float, optional
            Weight decay for regularization, by default 1e-5
        - eta : float, optional
            Hyperparameter controlling mixup ratio between inter-domain and inter-class mixup, by default 0.5
        - warmup : int, optional
            Number of warmup epochs before applying FairGB, by default 5
        - alpha : float, optional
            Trade-off parameter for validation metric: utility - alpha * fairness, by default 1.0

    Example
    -------
    .. code-block:: python

        from graphfairness.methods.fairgb import FairGB
        from graphfairness.models import GCN
        from graphfairness.datasets.fair_datasets import FairDataset
        
        # Load data
        dataset = FairDataset(root='./data', name='german')
        data = dataset.data
        
        # Initialize GNN backbone
        gnn_model = GCN(nfeat=data.features.shape[1], nhid=[16], nclass=1, dropout=0.5)
        
        # Create FairGB instance
        fair_model = FairGB(gnn_model, lr=0.001, weight_decay=1e-5, eta=0.5, warmup=5, alpha=1.0)
        
        # Train the model
        fair_model.train(data, epochs=1000, validation=True, eta=0.5, warmup=5)
        
        # Evaluate the model
        metrics = fair_model.evaluate(data)
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"Demographic Parity: {metrics['dp']:.4f}")
        print(f"Equal Opportunity: {metrics['eo']:.4f}")

    Note
    ----
    * FairGB requires binary classification tasks with binary sensitive attributes
    * The method includes a warmup period where standard training is performed before applying mixup
    * Only one additional hyperparameter (eta) is needed beyond standard GNN training
    """


    def __init__(self, model, **cfg):
        super().__init__(model)
        self.model = model
        self.cfg = BunchDict(cfg)
        
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)
        
        self.eta = self.cfg.get('eta', 0.5)
        self.warmup = self.cfg.get('warmup', 5)
        self.alpha = self.cfg.get('alpha', 1.0) 

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, data, epochs, validation=True, **train_args):
        """Train FairGB model with mixup and gradient re-weighting."""
        eta = train_args.get('eta', self.eta)
        warmup = train_args.get('warmup', self.warmup)
        
        device = data.features.device
        self.model.to(device)

        # Pre-compute neighbor distributions
        print("Pre-computing neighbor distributions for FairGB...")
        
        if hasattr(data.edge_index, 'coo'):
            row, col, _ = data.edge_index.coo()
            edge_index = torch.stack([row, col]).to(dtype=torch.long, device=device)
        elif hasattr(data.edge_index, 'indices'):
            edge_index = torch.stack(data.edge_index.indices()).to(device)
        else:
            edge_index = data.edge_index.to(device)
            
        # Get neighbor distribution for all nodes
        neighbor_dist_list = self.get_ins_neighbor_dist(data.features.size(0), edge_index, device)
        
        # Pre-compute group indices
        n_cls = data.labels.max().int().item() + 1
        n_sen = data.sens.max().int().item() + 1
        index_list = torch.arange(len(data.labels)).to(device)
        group_num_list, idx_info = [], []
        
        train_mask = torch.zeros(data.features.shape[0], dtype=torch.bool, device=device)
        train_mask[data.idx_train] = True
        
        for i in range(n_cls):
            for j in range(n_sen):
                mask = ((data.labels == i) & (data.sens == j) & train_mask)
                data_num = mask.sum()
                group_num_list.append(int(data_num.item()))
                idx_info.append(index_list[mask])

        best_val_tradeoff = -float('inf')
        
        tpbar = tqdm(total=epochs, desc=f"Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")

        for epoch in range(epochs):
            loss = self.train_step(data, epoch, warmup, eta, neighbor_dist_list, group_num_list, idx_info, n_cls, n_sen, train_mask)
            
            if validation:
                ret_val = self.evaluate(data, split='val')
                
                # Trade-off selection: (AUC + F1 + ACC) - alpha * (DP + EO)
                val_utility = ret_val['auc'] + ret_val['f1'] + ret_val['acc']
                val_fairness = ret_val['dp'] + ret_val['eo']
                current_tradeoff = val_utility - self.alpha * val_fairness

                if current_tradeoff > best_val_tradeoff:
                    best_val_tradeoff = current_tradeoff
                    if hasattr(self, 'weight_path'):
                        os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                        torch.save(self.model.state_dict(), self.weight_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss': "{:.4f}".format(loss)})
                tpbar.update(1)
        
        if tpbar is not None:
            tpbar.close()

    def train_step(self, data, epoch, warmup, eta, neighbor_dist_list, group_num_list, idx_info, n_cls, n_sen, train_mask):
        """Single training step with warmup and FairGB logic."""
        self.model.train()
        self.optimizer.zero_grad()
        device = data.features.device

        if hasattr(data.edge_index, 'coo'):
            row, col, _ = data.edge_index.coo()
            edge_index = torch.stack([row, col]).to(dtype=torch.long, device=device)
        elif hasattr(data.edge_index, 'indices'):
            edge_index = torch.stack(data.edge_index.indices()).to(device)
        else:
            edge_index = data.edge_index.to(device)

        if epoch >= warmup:
            # FairGB training with mixup and CAL
            sampling_src_idx, sampling_dst_idx = self.sampling_idx_individual_dst(
                group_num_list, idx_info, eta)
            
            # Beta Distribution for Lambda
            beta = torch.distributions.beta.Beta(2, 2)
            lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1).to(device)
            
            # Mixup
            new_edge_index = self.neighbor_sampling(data.features.size(0), edge_index, sampling_src_idx, neighbor_dist_list)
            new_x = self.saliency_mixup(data.features, sampling_src_idx, sampling_dst_idx, lam)
            
            output = self.model(new_x, new_edge_index) 
            
            num_original = data.features.shape[0]
            add_num = output.shape[0] - num_original
            new_train_mask = torch.zeros(output.shape[0], dtype=torch.bool, device=device)
            new_train_mask[num_original:] = True
            
            # Ensure output is [N, 1] for BCE
            output_mixed = output[new_train_mask].view(-1, 1)
            y_src = data.labels[sampling_src_idx].float().view(-1, 1).to(device)
            y_dst = data.labels[sampling_dst_idx].float().view(-1, 1).to(device)
            
            # Calculate UNREDUCED loss
            loss_src = F.binary_cross_entropy_with_logits(output_mixed, y_src, reduction='none')
            loss_dst = F.binary_cross_entropy_with_logits(output_mixed, y_dst, reduction='none')
            
            # Gradient-based Reweighting 
            # pos_grad formula: (1 - exp(-loss)) * lam
            pos_grad_src = (1. - torch.exp(-loss_src.detach())) * lam
            pos_grad_dst = (1. - torch.exp(-loss_dst.detach())) * (1 - lam)
            
            grad_count = []
            # Calculate sum of gradients per group
            for i in range(n_cls):
                for j in range(n_sen):
                    # Indices in the SAMPLING batch corresponding to this group
                    mask_src = (data.labels[sampling_src_idx] == i) & (data.sens[sampling_src_idx] == j)
                    mask_dst = (data.labels[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                    
                    # Sum gradients for this group from both src and dst contributions
                    g_sum = pos_grad_src[mask_src].sum().item() + pos_grad_dst[mask_dst].sum().item()
                    grad_count.append(g_sum)
            
            # Compute Group Weights 
            grad_array = np.array(grad_count)
            min_grad = np.min(grad_array) if len(grad_array) > 0 else 0
            if min_grad == 0 and np.sum(grad_array) > 0:
                 min_grad = np.min(grad_array[grad_array > 0]) # Find smallest non-zero

            group_weight_list = [float(min_grad) / (float(num) + EPS) for num in grad_count]
            
            # Apply Weights to Loss
            final_loss_src = loss_src.clone()
            final_loss_dst = loss_dst.clone()
            
            for i in range(n_cls):
                for j in range(n_sen):
                    
                    # Robust Logic:
                    weight_index = i * n_sen + j
                    if weight_index < len(group_weight_list):
                        w = group_weight_list[weight_index]
                        
                        # Apply to src loss where sample is in group (i,j)
                        m_src = (data.labels[sampling_src_idx] == i) & (data.sens[sampling_src_idx] == j)
                        final_loss_src[m_src] *= w
                        
                        # Apply to dst loss where sample is in group (i,j)
                        m_dst = (data.labels[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        final_loss_dst[m_dst] *= w

            loss = lam * final_loss_src + (1 - lam) * final_loss_dst
            loss = loss.mean()

        else:
            # Warmup: Standard Training 
            output = self.model(data.features, edge_index)
            loss = self.criterion(output[train_mask].view(-1, 1), data.labels[train_mask].float().view(-1, 1))

        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, split='test'):
        self.model.eval()
        
        if split == 'test' and hasattr(self, 'weight_path') and os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path))

        if hasattr(data.edge_index, 'coo'):
            row, col, _ = data.edge_index.coo()
            edge_index = torch.stack([row, col]).to(dtype=torch.long, device=data.features.device)
        elif hasattr(data.edge_index, 'indices'):
            edge_index = torch.stack(data.edge_index.indices()).to(data.features.device)
        else:
            edge_index = data.edge_index.to(data.features.device)

        output = self.model(data.features, edge_index)
        
        if split == 'val':
            idx = data.idx_val
        elif split == 'test':
            idx = data.idx_test
        else:
            idx = data.idx_train

        preds = (output.squeeze() > 0).type_as(data.labels)
        
        labels_eval = data.labels[idx].cpu().numpy()
        output_eval = output.detach().cpu().numpy()[idx.cpu()]
        preds_eval = preds[idx].cpu().numpy()
        sens_eval = data.sens[idx].cpu().numpy()

        auc = roc_auc_score(labels_eval, output_eval)
        f1 = f1_score(labels_eval, preds_eval)
        acc = accuracy_score(labels_eval, preds_eval)
        dp, eo = fair_metric(preds_eval, labels_eval, sens_eval)

        return {'auc': auc, 'f1': f1, 'acc': acc, 'dp': dp, 'eo': eo}

    # =================================================================
    # Static Methods for Mixup and Neighbor Sampling
    # =================================================================

    @staticmethod
    @torch.no_grad()
    def get_ins_neighbor_dist(num_nodes, edge_index, device):
        """
        Compute adjacent node distribution.
        """
        edge_index = edge_index.clone().to(device)
        row, col = edge_index[0], edge_index[1]
        
        neighbor_dist_list = []
        for j in range(num_nodes): 
            neighbor_dist = torch.zeros(num_nodes, dtype=torch.float32, device=device)
            idx = row[(col == j)]
            neighbor_dist[idx] = neighbor_dist[idx] + 1
            neighbor_dist_list.append(neighbor_dist)

        neighbor_dist_list = torch.stack(neighbor_dist_list, dim=0)
        neighbor_dist_list = F.normalize(neighbor_dist_list, dim=1, p=1)
        return neighbor_dist_list

    @staticmethod
    @torch.no_grad()
    def sampling_idx_individual_dst(group_num_list, idx_info, eta=0.5):
        """Sample counterfactual pairs for mixup."""
        n_cls, n_grp = 2, 2
        sampling_src_idx = torch.cat(idx_info)
        
        if np.random.rand() < eta:
            inter = True
        else:
            inter = False
            
        sampling_dst_idx = []
        for i in range(n_cls):
            for j in range(n_grp):
                if inter:
                    target_group_id = 2 * (1 - i) + j
                else:
                    target_group_id = 2 * i + (1 - j)
                
                if target_group_id < len(group_num_list) and group_num_list[target_group_id] > 0:
                    prob = torch.ones(group_num_list[target_group_id]) / group_num_list[target_group_id]
                    num_samples = group_num_list[i * 2 + j] 
                    if num_samples > 0:
                        sampled_idx = torch.multinomial(prob, num_samples, replacement=True)
                        sampled_idx = idx_info[target_group_id][sampled_idx]
                        sampling_dst_idx.append(sampled_idx)
                else:
                    num_samples = group_num_list[i * 2 + j]
                    if num_samples > 0:
                        sampling_dst_idx.append(idx_info[i*2+j])

        if len(sampling_dst_idx) > 0:
            sampling_dst_idx = torch.cat(sampling_dst_idx)
        else:
            sampling_dst_idx = sampling_src_idx # Fallback

        sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
        sampling_dst_idx = sampling_dst_idx[sorted_idx]

        return sampling_src_idx, sampling_dst_idx

    @staticmethod
    def saliency_mixup(x, sampling_src_idx, sampling_dst_idx, lam):
        """Mix node features of source and destination pairs."""
        new_src = x[sampling_src_idx.to(x.device), :].clone()
        new_dst = x[sampling_dst_idx.to(x.device), :].clone()
        lam = lam.to(x.device)

        mixed_node = lam * new_src + (1 - lam) * new_dst
        new_x = torch.cat([x, mixed_node], dim=0)
        return new_x

    @staticmethod
    @torch.no_grad()
    def neighbor_sampling(total_node, edge_index, sampling_src_idx, neighbor_dist_list):
        """Sample neighbors for mixed nodes to construct augmented graph."""
        device = edge_index.device
        sampling_src_idx = sampling_src_idx.clone().to(device)

        mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

        row, col = edge_index[0], edge_index[1]
        
        degree = scatter_add(torch.ones_like(col), col, dim_size=total_node)
        
        if len(degree) < total_node:
            degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))], dim=0)
            
        train_node_mask = torch.ones_like(degree, dtype=torch.bool)
        
        max_deg = int(degree.max().item())
        degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask], dim_size=max_deg+1).to(device).type(torch.float32)

        prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx), 1)
        aug_degree = torch.multinomial(prob + 1e-12, 1).to(device).squeeze(dim=1)
        
        aug_degree = torch.min(aug_degree, degree[sampling_src_idx])
        
        max_degree = int(degree.max().item()) + 1
        
        current_max_deg = int(aug_degree.max().item())
        if current_max_deg == 0: 
            return edge_index
            
        new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, current_max_deg, replacement=True)
        
        tgt_index = torch.arange(current_max_deg).unsqueeze(dim=0).to(device)
        new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
        
        new_row = (torch.arange(len(sampling_src_idx)).to(device) + total_node)
        new_row = new_row.repeat_interleave(aug_degree)
        
        inv_edge_index = torch.stack([new_col, new_row], dim=0)
        new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

        return new_edge_index