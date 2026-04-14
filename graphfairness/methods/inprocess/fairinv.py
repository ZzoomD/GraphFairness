import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from graphfairness.train import Trainer
from graphfairness.utils import BunchDict
from graphfairness.evaluation.metrics import fair_metric
import copy
from torch.nn.modules.loss import _Loss

class SAP(nn.Module):
    """
    Sensitive Attribute Partition (SAP) Module for unsupervised environment inference.
    """
    def __init__(self, hid_dim, out_dim, sens_infer_backbone):
        super(SAP, self).__init__()
        self.variant_infer = nn.Sequential(
            nn.Linear(in_features=2*hid_dim, out_features=1),
            nn.Sigmoid()
        )
        self.sens_infer_backbone = sens_infer_backbone
        self.sens_infer_classifier = nn.Sequential(
            nn.Linear(in_features=hid_dim+1, out_features=out_dim),
            nn.Softmax(dim=1)
        )
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, emb_cat, features, edge_index, labels):
        # Calculate variant scores for edges
        edge_weight_variant = self.variant_infer(emb_cat).squeeze()
        # Infer sensitive attribute partition based on variant scores
        edge_index = edge_index.fill_value(1., dtype=None)
        edge_index.storage.set_value_(edge_index.storage.value() * edge_weight_variant.to(edge_index.device()))
        h, output = self.sens_infer_backbone.get_embs_and_outs(features, edge_index) 
        sens_attr_partition = self.sens_infer_classifier(torch.cat([h, labels.unsqueeze(1)], dim=1))
        return sens_attr_partition, 1 - edge_weight_variant


class FairINV(Trainer,nn.Module):
    r"""Implementation of `FairINV` from the paper entitled `“One Fits All: Learning Fair 
    Graph Neural Networks for Various Sensitive Attributes” <https://arxiv.org/pdf/2406.13544v3>`.
    
    FairINV formulates the graph fairness problem from an invariant learning perspective. 
    It aims to learn invariant representations across different sensitive attribute environments 
    automatically inferred via a sensitive attribute partition module. It consists of two main 
    stages: Sensitive Attribute Partition (SAP) to unsupervisedly identify diverse environments, 
    and Sensitive Invariant Learning (SIL) to eliminate spurious correlations using a 
    variance-based loss.

    Parameters
    ----------
    model : nn.Module
        The GNN backbone model used for classification
    **cfg : dict
        Additional configuration parameters
        - lr : float, optional
            Learning rate for optimization, by default 1e-3
        - weight_decay : float, optional
            Weight decay for regularization, by default 1e-5
        - nfeat : int
            Number of input features
        - nhid : list
            Hidden layer dimensions (e.g., [16])
        - nclass : int
            Number of output classes
        - env_num : int
            Number of environments for partition, default is 2
        - partition_times : int
            Number of times to perform environment partition, default is 3
        - alpha : float
            Balance coefficient for mean loss in SIL, default is 10.0

    Example
    -------
    .. code-block:: python
        from graphfairness.methods.inprocess.fairgkd import FairGKD
        from graphfairness.data import FairDataset
        from graphfairness.models import ModelBuilder
        import torch

        # Load data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = FairDataset(root='./', name='german')
        fair_dataset = dataset.data.to(device)
        n_feat = fair_dataset.features.shape[1]
        
        # Build student model using ModelBuilder
        model_builder = ModelBuilder(device)
        model = model_builder.build(model_name='gcn',
                                    nfeat=n_feat,
                                    nclass=1,
                                    nhid=[16],
                                    dropout=0.5)
        
        # Create FairINV instance
        fair_model = FairINV(model, nfeat=n_feat, nhid=[16], nclass=1)
        
        # Train the model
        fair_model.train(fair_dataset, epochs=200, validation=True)
        
        # Evaluate the model
        metrics = fair_model.evaluate(fair_dataset)
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Demographic Parity: {metrics['dp']:.4f}")
        print(f"Equal Opportunity: {metrics['eo']:.4f}")

    Note
    ----
    * FairINV does not require access to sensitive attributes during training.
    * The stage-1 SAP training is independent of the stage-2 SIL training.
    """
    def __init__(self, model, **cfg):
        super().__init__(model, **cfg)
        self.model = model
        
        # Extract configuration
        self.cfg = BunchDict(cfg)
        self.nfeat = self.cfg.get('nfeat')
        self.nhid = self.cfg.get('nhid')[0] if isinstance(self.cfg.get('nhid'), list) else self.cfg.get('nhid')
        self.nclass = self.cfg.get('nclass', 1)
        self.env_num = self.cfg.get('env_num', 2)
        self.partition_times = self.cfg.get('partition_times', 3)
        self.alpha = self.cfg.get('alpha', 0.5)
        self.lr_sp = self.cfg.get('lr_sp', 0.01)
        self.device = next(model.parameters()).device
        
        # Loss functions
        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.criterion_env = nn.BCEWithLogitsLoss(reduction='none')

        # Optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.cfg.get('lr', 1e-3), 
            weight_decay=self.cfg.get('weight_decay', 1e-5)
        )
        
        for m in self.modules():
            self._weights_init(m)
        
    def _weights_init(self, m):        
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def train(self, data, epochs, validation=True):
        # Stage 1: Sensitive Attributes Partition (SAP)
        self.part_mat_list, self.edge_weight_inv_list = [], []
        
        for _ in range(self.partition_times):
            part_mat, edge_weight_inv = self._sens_partition(data)
            self.part_mat_list.append(part_mat[data.idx_train])
            self.edge_weight_inv_list.append(edge_weight_inv)
        
        # Stage 2: Sensitive Invariant Learning (SIL)
        tpbar = tqdm(total=epochs, desc=f"Training FairINV", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        
        for epoch in range(epochs):
            loss_results = self.train_step(data)
            
            if validation:
                ret_val = self.evaluate_step(data)
                
                # Model selection strategy based on dataset characteristics
                if data.dataset in ['pokec_z', 'pokec_n']:
                    # best validation loss for model selection
                    if ret_val['loss'] < getattr(self, 'best_loss', 100):
                        self.best_loss = ret_val['loss']
                        os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                        torch.save(self.model.state_dict(), self.weight_path)
                else:
                    # Utility-Fairness trade-off metric for model selection
                    res = ret_val['auc'] - ret_val['dp'] - ret_val['eo']
                    if res > getattr(self, 'best_res', 0):
                        self.best_res = res
                        os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                        torch.save(self.model.state_dict(), self.weight_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss': "{:.4f}".format(loss_results['loss'])})
                tpbar.update(1)
        
        if tpbar is not None:
            tpbar.close()

    def train_step(self, data) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        
        loss_log_list = []
        
        # Compute losses across multiple inferred environments
        for i, edge_weight_inv in enumerate(self.edge_weight_inv_list):
            edge_weight_copy = data.edge_index.clone()
            edge_weight_inv = edge_weight_inv.squeeze()
            edge_weight_copy = edge_weight_copy.fill_value(1., dtype=None)
            edge_weight_copy.storage.set_value_(edge_weight_copy.storage.value() * edge_weight_inv.to(edge_weight_copy.device()))
            output = self.model(data.features, edge_weight_copy) 
            
            # Split into groups based on SAP partition
            group_assign = self.part_mat_list[i].argmax(dim=1)
            for j in range(self.part_mat_list[i].shape[-1]):
                select_idx = torch.where(group_assign == j)[0]
                if len(select_idx) == 0: continue
                
                sub_logits = output[data.idx_train][select_idx]
                sub_labels = data.labels[data.idx_train][select_idx]
                
                loss_log = self.criterion_cls(sub_logits, sub_labels.unsqueeze(1).float())
                loss_log_list.append(loss_log.view(-1))
        
        # Variance-based loss to encourage invariant predictions [1]
        loss_log_cat = torch.cat(loss_log_list, dim=0)
        var, mean = torch.var_mean(loss_log_cat)
        loss_train = var + self.alpha * mean
        
        loss_train.backward()
        self.optimizer.step()
        
        return dict(loss=loss_train.item())

    @torch.no_grad()
    def evaluate_step(self, data, is_predict=False):
        self.model.eval()
        output = self.model(data.features, data.edge_index)
        
        if is_predict:
            return output
        else:
            loss_val = self.criterion_cls(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())
            preds = (output.squeeze() > 0).type_as(data.labels)
            
            # Calculate standard and fairness metrics
            auc_val = roc_auc_score(data.labels[data.idx_val].cpu(), output[data.idx_val].cpu())
            f1_val = f1_score(data.labels[data.idx_val].cpu(), preds[data.idx_val].cpu())
            acc_val = accuracy_score(data.labels[data.idx_val].cpu(), preds[data.idx_val].cpu())
            parity, equality = fair_metric(
                preds[data.idx_val].cpu().numpy(), 
                data.labels[data.idx_val].cpu().numpy(),
                data.sens[data.idx_val].cpu().numpy()
            )
            
            return dict(
                loss=loss_val.item(),
                auc=auc_val,
                f1=f1_val,
                acc=acc_val,
                dp=parity,
                eo=equality
            )

    def _sens_partition(self, data):
        """
        Performs sensitive attribute partition using unsupervised environment inference.
        """
        # Create helper modules
        sens_infer_backbone = copy.deepcopy(self.model).to(self.device)
        sens_infer_backbone.apply(self._weights_init) 
         
        partition_module = SAP(
            self.nhid, self.env_num, sens_infer_backbone
        ).to(self.device)
        optimizer_sp = torch.optim.Adam(
            partition_module.parameters(), 
            lr=self.lr_sp, 
            weight_decay=1e-5
        )
        
        # Train a reference ERM model to obtain initial embeddings
        ref_backbone = self._train_ref_model(data)
        
        ref_backbone.eval()
        with torch.no_grad():
            h, output = ref_backbone.get_embs_and_outs(data.features, data.edge_index)
        # Setup scale parameter for IRM penalty
        scale = torch.tensor(1.).to(self.device).requires_grad_()
        error = self.criterion_env(output[data.idx_train] * scale, data.labels[data.idx_train].unsqueeze(1).float())
        
        row, col = data.edge_index.storage.row(), data.edge_index.storage.col()
        emb_cat = torch.cat([h[row], h[col]], dim=1)
    
        # Optimize partition module to maximize IRM penalty (inferring worst-case environments)
        for _ in range(500):
            partition_module.train()
            optimizer_sp.zero_grad()
            
            part_mat, _ = partition_module(emb_cat.detach(), data.features, data.edge_index, data.labels)
            
            loss_penalty_list = []
            for env_idx in range(self.env_num):
                loss_weight = part_mat[:, env_idx]
                # Gradient-based IRM penalty calculation
                penalty_grad = grad(
                    (error.squeeze(1) * loss_weight[data.idx_train]).mean(), 
                    [scale], 
                    create_graph=True
                )[0].pow(2).mean()
                loss_penalty_list.append(penalty_grad)
            
            # Minimize negative penalty to maximize environment variability
            risk_final = -torch.stack(loss_penalty_list).sum()
            risk_final.backward(retain_graph=True)
            optimizer_sp.step()
        
        partition_module.eval()
        with torch.no_grad():
            soft_split, edge_weight_inv = partition_module(emb_cat.detach(), data.features, data.edge_index, data.labels)
        
        return soft_split, edge_weight_inv

    def _train_ref_model(self, data):
        """Trains a temporary reference GNN model via standard ERM."""
        ref_backbone = copy.deepcopy(self.model).to(self.device)
        ref_backbone.apply(self._weights_init) 
        
        optimizer_ref = torch.optim.Adam(
            ref_backbone.parameters(),
            lr=self.cfg.get('lr', 1e-3),
            weight_decay=self.cfg.get('weight_decay', 1e-5)
        )
        
        for _ in range(500):
            ref_backbone.train()
            optimizer_ref.zero_grad()
            
            output = ref_backbone(data.features, data.edge_index)
            loss = self.criterion_cls(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
            loss.backward()
            optimizer_ref.step()
        
        return ref_backbone