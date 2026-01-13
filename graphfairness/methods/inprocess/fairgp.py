import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from graphfairness.train import Trainer
from graphfairness.utils import BunchDict
from graphfairness.evaluation.metrics import fair_metric
from graphfairness.utils.fairgp_utils import partition_patch, adjacency_positional_encoding, laplacian_positional_encoding, edge_index_2_sparse_mx

class FairGP(Trainer):
    """
    FairGP: Graph Fairness via Graph Patching
    """
    def __init__(self, model, **cfg):
        super().__init__(model)
        self.model = model
        self.cfg = BunchDict(cfg)
        
        lr = self.cfg.get('lr', 0.01)
        weight_decay = self.cfg.get('weight_decay', 1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.pe_method = self.cfg.get('pe_method', 'adj')
        self.pe_dim = self.cfg.get('pe_dim', 2)
        self.patch_method = self.cfg.get('patch_method', 'metis')
        self.n_patch = self.cfg.get('n_patch', 100)
        self.num_hops = self.cfg.get('num_hops', 1)
        self.num_nodes_ori = self.cfg.get('num_nodes', None)

    def preprocess(self, data):
        """
        Preprocess data for FairGP: Positional Encoding + Patch Partition
        """
        feature = data.features
        edge_index = data.edge_index
        labels = data.labels
        
        # 1. Positional Encoding
        if self.pe_method == 'adj':
            sp_adj = edge_index_2_sparse_mx(edge_index, num_nodes=feature.shape[0])
            eignvalue, eignvector = adjacency_positional_encoding(sp_adj, self.pe_dim)
            if feature.is_cuda:
                eignvector = eignvector.to(feature.device)
            feature = torch.cat((feature, eignvector), dim=1)
        elif self.pe_method == 'lap':
            sp_adj = edge_index_2_sparse_mx(edge_index, num_nodes=feature.shape[0])
            eignvalue, eignvector = laplacian_positional_encoding(sp_adj, self.pe_dim)
            if feature.is_cuda:
                eignvector = eignvector.to(feature.device)
            feature = torch.cat((feature, eignvector), dim=1)
            
        # 2. Patch Partition
        # Note: partition_patch pads feature and labels with a virtual node
        patch, feature, labels, num_nodes_new = partition_patch(
            feature, edge_index, labels, 
            n_patches=self.n_patch, 
            num_nodes=feature.shape[0]-1 if self.model.num_nodes > feature.shape[0] else feature.shape[0], # handle if called multiple times or match
            method=self.patch_method
        )
        
        if feature.is_cuda or next(self.model.parameters()).is_cuda:
            device = next(self.model.parameters()).device
            patch = patch.to(device)
            feature = feature.to(device)
            labels = labels.to(device)
            
        return feature, labels, patch

    def train(self, data, epochs, validation=True, **train_kwargs):
        # Preprocess data once
        feature, labels, patch = self.preprocess(data)
        
        # Update model if feature dimension changed (PE added)
        # Note: In FairGP design, the model needs to be initialized with correct input dim.
        # Here we assume the user/caller handled initialization correctly OR we rebuild parts if needed.
        # But usually in this framework, model is passed in. 
        # CAUTION: If pe_dim > 0, the input model must accept nfeat + pe_dim. 
        # The calling script should handle this.
        
        best_auc_val = 0.0
        
        tpbar = tqdm(total=epochs, desc=f"Training FairGP", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward
            # FairGP model takes (x, patch, edge_index)
            # Typically edge_index is not used in FairGP forward if using ICABlock only, but the signature has it.
            logits = self.model(feature, patch, data.edge_index)
            
            loss = F.cross_entropy(logits[data.idx_train], labels[data.idx_train])
            
            loss.backward()
            self.optimizer.step()
            
            if validation:
                ret_val = self.evaluate_internal(logits, labels, data.sens, data.idx_val)
                
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
            
    def evaluate_internal(self, logits, labels, sens, idx):
        self.model.eval()
        with torch.no_grad():
            output = logits[idx]
            y = labels[idx]
            s = sens[idx]
            
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            auc = roc_auc_score(y.cpu().numpy(), probs[:, 1].cpu().numpy())
            acc = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
            f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy())
            
            dp, eo = fair_metric(preds.cpu().numpy(), y.cpu().numpy(), s.cpu().numpy())
            return {
                "auc_val": auc,
                "acc_val": acc,
                "f1_val": f1,
                "dp_val": dp,
                "eo_val": eo
            }

    def evaluate(self, data):
        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path))
        
        feature, labels, patch = self.preprocess(data)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(feature, patch, data.edge_index)
            
            res = self.evaluate_internal(logits, labels, data.sens, data.idx_test)
            
            return {
                "auc": res["auc_val"],
                "acc": res["acc_val"],
                "f1": res["f1_val"],
                "dp": res["dp_val"],
                "eo": res["eo_val"]
            }

