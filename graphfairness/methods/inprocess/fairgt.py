"""
FairGT: Fair Graph Transformer
Implements the FairGT method with random walk sampling and hop neighbor aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from graphfairness.train import Trainer
from graphfairness.utils import BunchDict
from graphfairness.evaluation.metrics import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
import os
import numpy as np


class FairGT(Trainer):
    r"""Implementation of FairGT (Fair Graph Transformer).

    FairGT uses Graph Transformer architecture with random walk sampling and 
    multi-hop neighbor aggregation to achieve fair node classification.

    Parameters
    ----------
    model : nn.Module
        The Graph Transformer backbone model
    **cfg : dict
        Additional configuration parameters
        - lr : float, optional
            Learning rate for optimization, by default 1e-3
        - weight_decay : float, optional
            Weight decay for regularization, by default 1e-5
        - hops : int, optional
            Number of hop neighbors to aggregate, by default 2
        - num_nodes : int
            Total number of nodes in the graph

    Example
    -------
    .. code-block:: python

        from graphfairness.methods.inprocess.fairgt import FairGT
        from graphfairness.models import GraphTransformer
        
        # Load data
        dataset = FairDataset(root='./', name='german')
        n_feat = dataset.data.features.shape[1]

        # Initialize the GraphTransformer backbone
        gt_model = GraphTransformer(nfeat=n_feat, nhid=[64], nclass=1, 
                                    nhead=2, nlayer=1, dropout=0.3)
        
        # Create FairGT instance
        fair_model = FairGT(gt_model, hops=2, num_nodes=dataset.data.features.shape[0])
        
        # Train the model
        fair_model.train(data, epochs=1000)
        
        # Evaluate
        metrics = fair_model.evaluate(data)
    """
    def __init__(self, model, **cfg):
        super().__init__(model)
        self.model = model
        
        self.cfg = BunchDict(cfg)
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)
        self.hops = self.cfg.get('hops', 2)
        self.num_nodes = self.cfg.get('num_nodes', 1000)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Learnable transition probabilities for random walk
        self.transition_probs = nn.Parameter(torch.rand(self.num_nodes, self.num_nodes))
        self.attn_layer = nn.Linear(self.model.hidden_dim, 1)
        
        # Move to device if model is on a device
        if next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
            self.transition_probs = self.transition_probs.to(device)
            self.attn_layer = self.attn_layer.to(device)

    def preprocess_features(self, data):
        """
        Preprocess features to include hop neighbors
        
        Args:
            data: Fair dataset object
            
        Returns:
            batched_features: [N, seq_len, nfeat] where seq_len = 1 + hops + walk_len
        """
        features = data.features
        edge_index = data.edge_index
        num_nodes = features.shape[0]
        nfeat = features.shape[1]
        
        # Initialize: self + hop neighbors + random walk
        seq_len = 1 + 2 * self.hops  # self + hops + walk_len
        batched_features = torch.zeros(num_nodes, seq_len, nfeat, device=features.device)
        
        # Add self features
        batched_features[:, 0, :] = features
        
        # Sample hop neighbors
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)
        
        # Multi-hop neighbor sampling
        for node_idx in range(num_nodes):
            neighbors_1hop = adj_list[node_idx]
            
            # 1-hop neighbors
            if len(neighbors_1hop) > 0:
                sampled_neighbors = np.random.choice(neighbors_1hop, 
                                                     size=min(self.hops, len(neighbors_1hop)), 
                                                     replace=False)
                for i, neighbor in enumerate(sampled_neighbors):
                    batched_features[node_idx, 1 + i, :] = features[neighbor]
        
        return batched_features

    def random_walk_sampling(self, batched_features):
        """
        Random walk sampling based on learned transition probabilities
        
        Args:
            batched_features: [N, seq_len, nfeat]
            
        Returns:
            walk_features: [N, walk_len, nfeat]
        """
        num_nodes = batched_features.shape[0]
        nfeat = batched_features.shape[2]
        walk_len = self.hops
        
        walk_features = torch.zeros(num_nodes, walk_len, nfeat, device=batched_features.device)
        
        for node_idx in range(num_nodes):
            current_node = node_idx
            for step in range(walk_len):
                # Sample next node based on learned transition probabilities
                transition_prob = F.softmax(self.transition_probs[current_node], dim=0)
                next_node = torch.multinomial(transition_prob, 1).item()
                walk_features[node_idx, step, :] = batched_features[next_node, 0, :]
                current_node = next_node
        
        return walk_features

    def forward_with_hop(self, data):
        """
        Forward pass with hop neighbor and random walk features
        
        Args:
            data: Fair dataset object
            
        Returns:
            output: [N, nclass] Classification logits
        """
        # Preprocess to get hop neighbor features
        batched_features = self.preprocess_features(data)  # [N, 1+hops, nfeat]
        
        # Add random walk features
        walk_features = self.random_walk_sampling(batched_features)  # [N, hops, nfeat]
        
        # Combine: self + hop + walk
        combined_features = torch.cat([batched_features, walk_features], dim=1)  # [N, 1+2*hops, nfeat]
        
        # Pass through Graph Transformer
        h = self.model.att_embeddings(combined_features)  # [N, seq_len, hidden_dim]
        
        # Transform through encoder layers
        for layer in self.model.layers:
            h = layer(h)
        
        h = self.model.final_ln(h)  # [N, seq_len, hidden_dim]
        
        # Attention-based aggregation
        self_repr = h[:, 0, :].unsqueeze(1)  # [N, 1, hidden_dim]
        neighbor_repr = h[:, 1:, :]  # [N, seq_len-1, hidden_dim]
        
        att_weights = F.softmax(self.attn_layer(h), dim=1)  # [N, seq_len, 1]
        aggregated = (neighbor_repr * att_weights[:, 1:, :]).sum(dim=1)  # [N, hidden_dim]
        
        # Final representation: self + aggregated neighbors
        final_repr = self_repr.squeeze(1) + aggregated
        
        # Classification
        output = self.model.fc(torch.relu(self.model.out_proj(final_repr)))
        
        return output

    def train(self, data, epochs, validation=True, **train_kwargs):
        """
        Train the FairGT model
        
        Args:
            data: Fair dataset object
            epochs: Number of training epochs
            validation: Whether to use validation
        """
        best_auc_val = 0.0
        
        tpbar = tqdm(total=epochs, desc=f"Training FairGT", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        
        for epoch in range(epochs):
            self.model.train()
            self.attn_layer.train()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.forward_with_hop(data)
            
            # Compute loss on training set
            loss = self.criterion(output[data.idx_train].squeeze(), 
                                 data.labels[data.idx_train].float())
            
            loss.backward()
            self.optimizer.step()
            
            if validation:
                ret_val = self.evaluate_step(data)
                
                if ret_val["auc_val"] > best_auc_val:
                    best_auc_val = ret_val["auc_val"]
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'attn_layer_state_dict': self.attn_layer.state_dict(),
                        'transition_probs': self.transition_probs.data
                    }, self.weight_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss': "{:.4f}".format(loss.item()),
                                  'auc_val': "{:.4f}".format(ret_val.get("auc_val", 0.0))})
                tpbar.update(1)
        
        if tpbar is not None:
            tpbar.close()

    def evaluate_step(self, data):
        """Evaluate on validation set"""
        self.model.eval()
        self.attn_layer.eval()
        
        with torch.no_grad():
            output = self.forward_with_hop(data)
        
        # Validation metrics
        output_val = output[data.idx_val].detach().cpu().numpy()
        labels_val = data.labels[data.idx_val].cpu().numpy()
        sens_val = data.sens[data.idx_val].cpu().numpy()
        
        auc_val = roc_auc_score(labels_val, output_val)
        preds_val = (output_val > 0).astype(int)
        
        acc_val = accuracy_score(labels_val, preds_val)
        f1_val = f1_score(labels_val, preds_val)
        dp_val, eo_val = fair_metric(preds_val, labels_val, sens_val)
        
        return {
            "auc_val": auc_val,
            "acc_val": acc_val,
            "f1_val": f1_val,
            "dp_val": abs(dp_val),
            "eo_val": abs(eo_val)
        }

    def evaluate(self, data):
        """Evaluate on test set"""
        # Load best model
        if os.path.exists(self.weight_path):
            checkpoint = torch.load(self.weight_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.attn_layer.load_state_dict(checkpoint['attn_layer_state_dict'])
            self.transition_probs.data = checkpoint['transition_probs']
        
        self.model.eval()
        self.attn_layer.eval()
        
        with torch.no_grad():
            output = self.forward_with_hop(data)
        
        # Test metrics
        output_test = output[data.idx_test].detach().cpu().numpy()
        labels_test = data.labels[data.idx_test].cpu().numpy()
        sens_test = data.sens[data.idx_test].cpu().numpy()
        
        auc_test = roc_auc_score(labels_test, output_test)
        preds_test = (output_test > 0).astype(int)
        
        acc_test = accuracy_score(labels_test, preds_test)
        f1_test = f1_score(labels_test, preds_test)
        dp_test, eo_test = fair_metric(preds_test, labels_test, sens_test)
        
        return {
            "auc": auc_test,
            "acc": acc_test,
            "f1": f1_test,
            "dp": abs(dp_test),
            "eo": abs(eo_test)
        }
