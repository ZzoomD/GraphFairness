import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm 
from torch_geometric.utils import dropout_adj
from graphfairness.train import *
from graphfairness.models import *
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score
from graphfairness.evaluation.metrics import *
from graphfairness.utils import BunchDict
from tqdm import tqdm
import os


class NIFTY(Trainer):
    r"""Implementation of `NIFTY` from the paper entitled `"Towards a Unified Framework for Fair and Stable Graph Representation Learning" <https://arxiv.org/abs/2102.13186>`.

    NIFTY incorporates both fairness and stability in graph representation learning by:
    1. Using a triplet-based objective function that maximizes agreement between original and augmented views
    2. Applying Lipschitz normalization in GNN architecture to enhance stability
    3. Generating augmented views through attribute perturbation, sensitive attribute flipping, and edge dropping

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
        - nhid : list or int
            Number of hidden units in each layer
        - nclass : int
            Number of output classes
        - dropout : float
            Dropout probability
        - sim_coeff : float, optional
            Coefficient for similarity loss, by default 0.5
        - proj_hidden : int, optional
            Hidden dimension for projection head, by default 16
        - drop_edge_rate_1 : float, optional
            Edge drop rate for first augmentation, by default 0.1
        - drop_edge_rate_2 : float, optional
            Edge drop rate for second augmentation, by default 0.1
        - drop_feature_rate_1 : float, optional
            Feature drop rate for first augmentation, by default 0.1
        - drop_feature_rate_2 : float, optional
            Feature drop rate for second augmentation, by default 0.1

    Example
    -------
    .. code-block:: python

        from graphfairness.methods.inprocess.nifty import NIFTY
        from graphfairness.models import GCN
        import torch_geometric as pyg

        # load data
        dataset = FairDataset(root='./', name='german')
        n_feat = dataset.data.features.shape[1]

        # Initialize the GNN backbone (must output `nhid` dimension, e.g. 16)
        gnn_model = GCN(nfeat=n_feat, nhid=[16], nclass=1, dropout=0.5)
        
        # Create NIFTY instance
        nifty_model = NIFTY(gnn_model, nfeat=n_feat, nclass=1, nhid=[16], dropout=0.5, 
                           sim_coeff=0.5, proj_hidden=16)
        
        # Train the model
        nifty_model.train(data, epochs=200, validation=True, sens_idx=0)
        
        # Evaluate the model
        metrics = nifty_model.evaluate(data)
        print(f"Accuracy: {metrics['acc_val']:.4f}")
        print(f"AUC: {metrics['auc_val']:.4f}")
        print(f"Demographic Parity: {metrics['dp_val']:.4f}")
        print(f"Equal Opportunity: {metrics['eo_val']:.4f}")

    Note
    ----
    * NIFTY simultaneously improves both fairness and stability of graph representations.
    * The method uses self-supervised learning with augmented views to achieve invariance to sensitive attributes and perturbations.
    * Hyperparameter sim_coeff controls the trade-off between contrastive learning and classification tasks.
    * The sens_idx parameter should be specified during training to indicate the index of the sensitive attribute.
    * The underlying GNN model (passed as `model`) should ideally incorporate **Spectral Normalization** for Lipschitz constraint, as suggested by the paper.
    """

    def __init__(self, model, **cfg):
        super().__init__(model)
        self.model = model
        
        self.cfg = BunchDict(cfg)
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)
        self.sim_coeff = self.cfg.get('sim_coeff', 0.5)
        self.proj_hidden = self.cfg.get('proj_hidden', 16)
        self.drop_edge_rate_1 = self.cfg.get('drop_edge_rate_1', 0.001)
        self.drop_edge_rate_2 = self.cfg.get('drop_edge_rate_2', 0.001)
        self.drop_feature_rate_1 = self.cfg.get('drop_feature_rate_1', 0.1)
        self.drop_feature_rate_2 = self.cfg.get('drop_feature_rate_2', 0.1)

        self.gnn_output_dim = self.cfg.nhid[-1] if isinstance(self.cfg.nhid, list) else self.cfg.nhid
        
        # Projection head
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(self.gnn_output_dim, self.proj_hidden)),
            nn.BatchNorm1d(self.proj_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(self.proj_hidden, self.gnn_output_dim)),
            nn.BatchNorm1d(self.gnn_output_dim)
        )

        # Prediction head
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(self.gnn_output_dim, self.gnn_output_dim)),
            nn.BatchNorm1d(self.gnn_output_dim),
            nn.ReLU(inplace=True)
        )
        self.fc4 = spectral_norm(nn.Linear(self.gnn_output_dim, self.gnn_output_dim))

        # Classifier
        self.classifier = spectral_norm(nn.Linear(self.gnn_output_dim, self.cfg.nclass))

        # Optimizers
        self.optimizer_1 = torch.optim.Adam(
            list(self.model.parameters()) + 
            list(self.fc1.parameters()) + 
            list(self.fc2.parameters()) + 
            list(self.fc3.parameters()) + 
            list(self.fc4.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        self.optimizer_2 = torch.optim.Adam(
            list(self.classifier.parameters()) + list(self.model.parameters()),
            lr=lr, weight_decay=weight_decay
        )

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def to(self, device):
        """Override to method to ensure all components are on the same device"""
        self.model = self.model.to(device)
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        self.fc3 = self.fc3.to(device)
        self.fc4 = self.fc4.to(device)
        self.classifier = self.classifier.to(device)
        return self

    def projection(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        return z

    def prediction(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        return z

    def D(self, x1, x2):
        """Negative cosine similarity"""
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    def drop_feature(self, x, drop_prob, sens_idx, sens_flag=True):
        """Drop features and optionally flip sensitive attribute"""
        drop_mask = torch.empty(
            (x.size(1), ),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob

        x = x.clone()
        drop_mask[sens_idx] = False

        # Add noise to dropped features
        x[:, drop_mask] += torch.randn_like(x[:, drop_mask]) * 0.1

        # Flip sensitive attribute
        if sens_flag:
            x[:, sens_idx] = 1 - x[:, sens_idx]

        return x

    def train(self, data, epochs, validation=True, **train_wargs):
        # Get sensitive attribute index from train_wargs
        sens_idx = train_wargs.get('sens_idx', 0)
        print(f"Using sensitive attribute index: {sens_idx}")

        # Ensure all components are on the same device as the data
        self.to(data.features.device)

        # Convert edge index from SparseTensor to standard [2, E] tensor format if needed
        if 'SparseTensor' in str(type(data.edge_index)):
            adj = data.edge_index
            row, col, _ = adj.coo()
            data.edge_index = torch.stack([row, col], dim=0)

        # Prepare validation augmentations - use our custom dropout_adj
        val_edge_index_1 = dropout_adj(data.edge_index, p=self.drop_edge_rate_1)[0]
        val_edge_index_2 = dropout_adj(data.edge_index, p=self.drop_edge_rate_2)[0]
        val_x_1 = self.drop_feature(data.features, self.drop_feature_rate_1, sens_idx, sens_flag=False)
        val_x_2 = self.drop_feature(data.features, self.drop_feature_rate_2, sens_idx)

        best_loss = float('inf')
        best_auc_val = 0.0
        
        tpbar = tqdm(total=epochs, desc=f"Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        
        for epoch in range(epochs):
            loss_train = self.train_step(data, sens_idx)

            if validation:
                val_s_loss, val_c_loss = self.validation_step(
                    data, val_x_1, val_edge_index_1, val_x_2, val_edge_index_2
                )
                val_loss = val_s_loss + val_c_loss

                # Get validation metrics
                self.model.eval()
                emb = self.model(data.features, data.edge_index)
                output = self.classifier(emb)
                preds = (output.squeeze() > 0).type_as(data.labels)
                auc_val = roc_auc_score(data.labels.cpu().numpy()[data.idx_val.cpu()], 
                                       output.detach().cpu().numpy()[data.idx_val.cpu()])

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_auc_val = auc_val
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.weight_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss_train': "{:.4f}".format(loss_train)})
                tpbar.update(1)

        if tpbar is not None:
            tpbar.close()

    def train_step(self, data, sens_idx) -> float:
        self.model.train()
        self.fc1.train()
        self.fc2.train()
        self.fc3.train()
        self.fc4.train()
        self.classifier.train()
        
        # Convert edge index from SparseTensor to standard [2, E] tensor format if needed
        if 'SparseTensor' in str(type(data.edge_index)):
            adj = data.edge_index
            row, col, _ = adj.coo()
            data.edge_index = torch.stack([row, col], dim=0)

        # Generate two augmented views using our custom dropout_adj
        edge_index_1 = dropout_adj(data.edge_index, p=self.drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(data.edge_index, p=self.drop_edge_rate_2)[0]
        x_1 = self.drop_feature(data.features, self.drop_feature_rate_1, sens_idx, sens_flag=False)
        x_2 = self.drop_feature(data.features, self.drop_feature_rate_2, sens_idx)

        # Get embeddings for both views
        z1 = self.model(x_1, edge_index_1)
        z2 = self.model(x_2, edge_index_2)

        # Step 1: Update encoder and projection heads (contrastive learning)
        self.optimizer_1.zero_grad()
        
        # Projections
        p1 = self.projection(z1)
        p2 = self.projection(z2)
        
        # Predictions
        h1 = self.prediction(p1)
        h2 = self.prediction(p2)
        
        # Contrastive loss
        l1 = self.D(h1[data.idx_train], p2[data.idx_train]) / 2
        l2 = self.D(h2[data.idx_train], p1[data.idx_train]) / 2
        sim_loss = self.sim_coeff * (l1 + l2)
        
        sim_loss.backward()
        self.optimizer_1.step()

        # Step 2: Update classifier
        self.optimizer_2.zero_grad()
        
        z1 = self.model(x_1, edge_index_1)
        z2 = self.model(x_2, edge_index_2)
        c1 = self.classifier(z1)
        c2 = self.classifier(z2)
        
        # Classification loss
        l3 = self.criterion(c1[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float()) / 2
        l4 = self.criterion(c2[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float()) / 2
        cl_loss = (1 - self.sim_coeff) * (l3 + l4)
        
        cl_loss.backward()
        self.optimizer_2.step()

        total_loss = sim_loss.item() + cl_loss.item()
        return total_loss

    def validation_step(self, data, x_1, edge_index_1, x_2, edge_index_2):
        self.model.eval()
        self.fc1.eval()
        self.fc2.eval()
        self.fc3.eval()
        self.fc4.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            z1 = self.model(x_1, edge_index_1)
            z2 = self.model(x_2, edge_index_2)

            # Projections
            p1 = self.projection(z1)
            p2 = self.projection(z2)
            
            # Predictions
            h1 = self.prediction(p1)
            h2 = self.prediction(p2)
            
            # Contrastive loss
            l1 = self.D(h1[data.idx_val], p2[data.idx_val]) / 2
            l2 = self.D(h2[data.idx_val], p1[data.idx_val]) / 2
            sim_loss = self.sim_coeff * (l1 + l2)

            # Classification loss
            c1 = self.classifier(z1)
            c2 = self.classifier(z2)
            l3 = self.criterion(c1[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float()) / 2
            l4 = self.criterion(c2[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float()) / 2
            cl_loss = (1 - self.sim_coeff) * (l3 + l4)

        return sim_loss.item(), cl_loss.item()

    @torch.no_grad()
    def evaluate_step(self, data, is_predict=False):
        """
        shared by self.evaluate and self.predict
        """
        self.model.eval()
        self.fc1.eval()
        self.fc2.eval()
        self.fc3.eval()
        self.fc4.eval()
        self.classifier.eval()
        
        emb = self.model(data.features, data.edge_index)
        output = self.classifier(emb)
        output = output.detach()
        preds = (output.squeeze() > 0).type_as(data.labels)
        
        if is_predict:
            return output
        else:
            auc_val = roc_auc_score(data.labels.cpu().numpy()[data.idx_val.cpu()],
                                    output.detach().cpu().numpy()[data.idx_val.cpu()])
            f1_val = f1_score(data.labels[data.idx_val].cpu().numpy(), preds[data.idx_val].cpu().numpy())
            acc_val = accuracy_score(data.labels[data.idx_val].cpu().numpy(), preds[data.idx_val].cpu().numpy())
            parity_val, equality_val = fair_metric(preds[data.idx_val].cpu().numpy(), 
                                                  data.labels[data.idx_val].cpu().numpy(),
                                                  data.sens[data.idx_val].cpu().numpy())
            return dict(auc_val=auc_val,
                        f1_val=f1_val,
                        acc_val=acc_val,
                        dp_val=parity_val,
                        eo_val=equality_val)