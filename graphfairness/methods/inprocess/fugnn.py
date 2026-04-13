import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from graphfairness.train import Trainer
from graphfairness.utils import BunchDict
from graphfairness.evaluation.metrics import fair_metric
from graphfairness.utils.fairgp_utils import edge_index_2_sparse_mx, adjacency_positional_encoding

class FairFUGNN(Trainer):
    r"""Implementation of FUGNN (Fairness-aware Undirected Graph Neural Network).

    FUGNN uses spectral decomposition and neural networks to achieve fair node classification.

    Parameters
    ----------
    model : nn.Module
        The FUGNN backbone model
    **cfg : dict
        Additional configuration parameters
        - lr : float, optional
            Learning rate for optimization, by default 1e-3
        - weight_decay : float, optional
            Weight decay for regularization, by default 1e-5
        - k : int, optional
            Number of eigenvalues/eigenvectors, by default 10
    """
    def __init__(self, model, **cfg):
        super().__init__(model)
        self.model = model
        self.cfg = BunchDict(cfg)
        
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)
        self.k = self.cfg.get('k', 10)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        # Spectral info cache
        self.eignvalue = None
        self.eignvector = None

    def preprocess_spectral(self, data):
        """Perform spectral decomposition if not already done."""
        if self.eignvalue is not None and self.eignvector is not None:
            return self.eignvalue, self.eignvector
            
        edge_index = data.edge_index
        num_nodes = data.features.shape[0]
        
        # Sparse Matrix conversion
        # We use symmetric normalized or just raw adj? 
        # FUGNN original uses adj = adj + adj.T ... (symmetric)
        # We follow the same logic.
        sp_adj = edge_index_2_sparse_mx(edge_index, num_nodes=num_nodes)
        # Symmetrize and add self loops if needed is usually handled in edge_index_2_sparse_mx or before
        
        eignvalue, eignvector = adjacency_positional_encoding(sp_adj, self.k)
        
        self.eignvalue = eignvalue.to(data.features.device)
        self.eignvector = eignvector.to(data.features.device)
        
        return self.eignvalue, self.eignvector

    def train(self, data, epochs, validation=True, **train_kwargs):
        """Train the FUGNN model."""
        best_auc_val = 0.0
        
        # Ensure spectral data is ready
        ev, evc = self.preprocess_spectral(data)
        
        tpbar = tqdm(total=epochs, desc=f"Training FUGNN", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(ev, evc, data.features)
            
            # Compute loss
            loss = self.criterion(output[data.idx_train], data.labels[data.idx_train].long())
            
            loss.backward()
            self.optimizer.step()
            
            if validation:
                ret_val = self.evaluate_step(data)
                if ret_val["auc_val"] > best_auc_val:
                    best_auc_val = ret_val["auc_val"]
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.weight_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss': "{:.4f}".format(loss.item()), 
                                   'auc_val': "{:.4f}".format(ret_val.get("auc_val", 0.0))})
                tpbar.update(1)
        
        if tpbar is not None:
            tpbar.close()

    def evaluate_step(self, data):
        """Evaluate on validation set."""
        self.model.eval()
        
        ev, evc = self.preprocess_spectral(data)
        
        with torch.no_grad():
            output = self.model(ev, evc, data.features)
        
        output_val = output[data.idx_val].detach()
        labels_val = data.labels[data.idx_val].cpu().numpy()
        sens_val = data.sens[data.idx_val].cpu().numpy()
        
        # FUGNN uses CrossEntropy, output is logits [N, 2]
        probs_val = F.softmax(output_val, dim=1)[:, 1].cpu().numpy()
        preds_val = torch.argmax(output_val, dim=1).cpu().numpy()
        
        auc_val = roc_auc_score(labels_val, probs_val)
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
        """Evaluate on test set."""
        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path))
            
        self.model.eval()
        ev, evc = self.preprocess_spectral(data)
        
        with torch.no_grad():
            output = self.model(ev, evc, data.features)
            
        output_test = output[data.idx_test].detach()
        labels_test = data.labels[data.idx_test].cpu().numpy()
        sens_test = data.sens[data.idx_test].cpu().numpy()
        
        probs_test = F.softmax(output_test, dim=1)[:, 1].cpu().numpy()
        preds_test = torch.argmax(output_test, dim=1).cpu().numpy()
        
        auc_test = roc_auc_score(labels_test, probs_test)
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
