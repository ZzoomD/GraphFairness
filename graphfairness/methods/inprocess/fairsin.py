import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from graphfairness.train import Trainer
from graphfairness.models import *
from graphfairness.utils import BunchDict
from graphfairness.evaluation.metrics import fair_metric
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
import os

class FairSIN(Trainer):
    r"""Implementation of `FairSIN` from the paper entitled `“FairSIN: Achieving Fairness in 
    Graph Neural Networks through Sensitive Information Neutralization”<https://arxiv.org/pdf/2403.12474>`.

    FairSIN introduces a neutralization-based paradigm where additional Fairness-facilitating Features (F3) 
    are incorporated into node features. It trains an estimator to predict the features of heterogeneous neighbors 
    and uses adversarial training to further reduce sensitive bias.

    Parameters
    ----------
    model : nn.Module
        The GNN backbone model used for classification
    **cfg : dict
        Additional configuration parameters
        - lr : float
            Learning rate
        - weight_decay : float
            Weight decay
        - delta : float
            Trade-off parameter for feature neutralization strength
        - m_hidden : int
            Hidden dimension for the estimator MLP
        - beta : float
            Coefficient for adversarial training

    Example
    -------
    .. code-block:: python

        from graphfairness.methods.fairsin import FairSIN
        from graphfairness.models import GCN

        # Initialize GNN
        gnn_model = GCN(nfeat=n_feat, nhid=[16], nclass=1, dropout=0.5)
        
        # Initialize FairSIN
        method = FairSIN(gnn_model, nfeat=n_feat, nhid=[16], nclass=1, delta=1.0)
        
        # Train
        method.train(data, epochs=500, m_epochs=200)

    Note
    ----
    * FairSIN operates by neutralizing sensitive information rather than filtering it out
    * Requires heterogeneous neighbors for effective neutralization; uses MLP to generalize knowledge
    """

    def __init__(self, model,** cfg):
        super().__init__(model)
        self.model = model
        self.cfg = BunchDict(cfg)
        
        lr = self.cfg.get('lr', 1e-3)
        self.weight_decay = self.cfg.get('weight_decay', 1e-5)
        m_hidden = self.cfg.get('m_hidden', 16)
        nfeat = self.cfg.get('nfeat')
        self.m_lr = cfg.get("m_lr", 0.001)

        # Estimator MLP for predicting heterogeneous neighbor features
        self.estimator = MLP(input_dim=nfeat, hidden_dim=m_hidden, output_dim=nfeat)
        
        # Discriminator for adversarial fairness constraint
        disc_input_dim = self.cfg.nhid[-1] if isinstance(self.cfg.nhid, list) else self.cfg.nhid
        self.discriminator = nn.Sequential(
            nn.Linear(disc_input_dim, disc_input_dim // 2),
            nn.ReLU(),
            nn.Linear(disc_input_dim // 2, 1)
        )

        # Optimizers
        self.optimizer_est = torch.optim.Adam(self.estimator.parameters(), lr=self.m_lr, weight_decay=self.weight_decay)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=self.weight_decay)
        
        # Combined optimizer for GNN and estimator
        self.optimizer_g = torch.optim.Adam(
            list(self.model.parameters()) + list(self.estimator.parameters()),
            lr=lr, weight_decay=self.weight_decay
        )

        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.criterion_est = nn.MSELoss()
        self.criterion_adv = nn.BCEWithLogitsLoss()
        
        self.best_model_path = os.path.join(os.getcwd(), "best_model.pth")
        self.best_val_auc = -1

    def train(self, data, epochs, validation=True, **train_args):
        # Parsing training arguments
        alpha = train_args.get('alpha', 0) 
        beta = train_args.get('beta', 0.01) # Adversarial weight
        delta = train_args.get('delta', 1.0) # Neutralization strength
        m_epochs = train_args.get('m_epochs', 200) # Epochs for estimator pre-training

        # Move models to device
        self.model = self.model.to(data.features.device)
        self.estimator = self.estimator.to(data.features.device)
        self.discriminator = self.discriminator.to(data.features.device)

        # Calculate ground-truth heterogeneous neighbor features
        print("Calculating heterogeneous neighbor features...")
        h_X, mask = self.get_hetero_features(data)
        h_X = h_X.to(data.features.device)
        
        # Pre-train the estimator MLP
        print("Pre-training Estimator...")
        self.train_estimator(data.features, h_X, mask, m_epochs)

        # Main Training Loop
        best_auc_val = 0.0
        best_fair_val = 100.0
        
        tpbar = tqdm(total=epochs, desc=f"Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        
        for epoch in range(epochs):
            loss_dict = self.train_step(data, delta, beta)

            if validation and (epoch + 1) % 10 == 0:  
                ret_val = self.evaluate_step(data, delta, split='val')
                
                if ret_val["auc"] > best_auc_val:
                    best_auc_val = ret_val["auc"]
                    best_fair_val = ret_val["dp"] + ret_val["eo"]
                    os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'estimator_state_dict': self.estimator.state_dict()
                    }, self.best_model_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss': "{:.2f}".format(loss_dict["loss"])})
                tpbar.update(1)

        if tpbar is not None:
            tpbar.close()

        # Load best model after training
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.estimator.load_state_dict(checkpoint['estimator_state_dict'])

    def train_estimator(self, features, h_X, mask, m_epochs):
        """
        Pre-trains the sensitive information estimator (A_hat) to predict h_X.
        Only nodes with heterogeneous neighbors (mask=True) are used for training.
        """
        print("Pre-training Estimator...")
        
        # Skip if no nodes have heterogeneous neighbors
        if not torch.any(mask):
            print("Warning: No nodes have heterogeneous neighbors (mask is all False). Skipping estimator pre-training.")
            return

        # Use only nodes with heterogeneous neighbors for training
        features_masked = features[mask]
        h_X_masked = h_X[mask] 
        
        # Early stopping parameters
        best_loss = float('inf')
        patience = 20
        counter = 0

        for epoch in tqdm(range(m_epochs)):
            self.estimator.train()
            self.optimizer_est.zero_grad()
            
            output = self.estimator(features_masked) 
            
            loss = self.criterion_est(output, h_X_masked) 
            
            loss.backward()
            self.optimizer_est.step()
            
            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"Estimator pre-training finished. Best loss: {best_loss:.4f}")
    
    def train_step(self, data, delta, beta) -> dict:
        self.model.train()
        self.estimator.train()
        self.discriminator.train()

        # --- Update GNN + Estimator (Generator) ---
        self.discriminator.requires_grad_(False)
        self.optimizer_g.zero_grad()

        # Feature Neutralization: X_tilde = X + delta * Estimator(X)
        x_neutral = data.features + delta * self.estimator(data.features)
        
        # Forward pass through GNN
        if hasattr(self.model, 'get_embs_and_outs'):
            embs, y_output = self.model.get_embs_and_outs(x_neutral, data.edge_index)
        else:
            y_output = self.model(x_neutral, data.edge_index)
            embs = y_output 

        # Classification Loss 
        loss_cls = self.criterion_cls(y_output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        
        # Adversarial loss for fairness (generator tries to fool discriminator)
        loss_adv = torch.tensor(0.0)
        if beta > 0:
            s_pred = self.discriminator(embs[data.idx_train])
            s_target = data.sens[data.idx_train].unsqueeze(1).float()
            loss_adv = self.criterion_adv(s_pred, s_target)
            loss_g = loss_cls - beta * loss_adv  # Minimize classification loss, maximize adversarial loss
        else:
            loss_g = loss_cls

        loss_g.backward()
        self.optimizer_g.step()

        # --- Update Discriminator ---
        loss_d = torch.tensor(0.0)
        if beta > 0:
            self.discriminator.requires_grad_(True)
            self.optimizer_disc.zero_grad()
            
            # Discriminator tries to predict sensitive attributes from embeddings
            s_pred_d = self.discriminator(embs[data.idx_train].detach())
            loss_d = self.criterion_adv(s_pred_d, data.sens[data.idx_train].unsqueeze(1).float())
            
            loss_d.backward()
            self.optimizer_disc.step()

        return dict(loss=loss_g.item(), 
                    loss_cls=loss_cls.item(), 
                    loss_adv=loss_adv.item(),
                    loss_disc=loss_d.item())

    @torch.no_grad()
    def evaluate_step(self, data, delta=1.0, split='val', is_predict=False):
        self.model.eval()
        self.estimator.eval()
        
        # Use neutralized features for evaluation
        x_neutral = data.features + delta * self.estimator(data.features)
        
        output = self.model(x_neutral, data.edge_index)
        output = output.detach()
        preds = (output.squeeze() > 0).type_as(data.labels)

        if is_predict:
            return output
        else:
            # Select indices based on split
            if split == 'val':
                idx = data.idx_val
            elif split == 'test':
                idx = data.idx_test
            else:  # 'train'
                idx = data.idx_train
                
            # Calculate evaluation metrics
            auc = roc_auc_score(
                data.labels.cpu().numpy()[idx.cpu()],
                output.detach().cpu().numpy()[idx.cpu()]
            )
            f1 = f1_score(
                data.labels[idx].cpu().numpy(), 
                preds[idx].cpu().numpy()
            )
            acc = accuracy_score(
                data.labels[idx].cpu().numpy(), 
                preds[idx].cpu().numpy()
            )
            
            parity, equality = fair_metric(
                preds[idx].cpu().numpy(), 
                data.labels[idx].cpu().numpy(),
                data.sens[idx].cpu().numpy()
            )
            return dict(
                auc=auc,
                f1=f1,
                acc=acc,
                dp=parity,
                eo=equality
            )

    def get_hetero_features(self, data):
        """
        Calculates average features of heterogeneous neighbors for each node.
        
        Args:
            data: Graph data containing edge_index, features, and sensitive attributes
            
        Returns:
            h_X: Tensor of shape [N, F] - average features of heterogeneous neighbors
            mask: Boolean tensor of shape [N] - True if node has heterogeneous neighbors
        """
        device = data.features.device
        
        num_nodes = data.features.shape[0]

        row, col, _ = data.edge_index.coo()
        row = row.to(device)
        col = col.to(device)
        
        sens = data.sens
        if hasattr(sens, 'to_dense'):
            sens = sens.to_dense() 
        sens = sens.to(device)
        if sens.ndim > 1:
            sens = sens.squeeze()
        if sens.dtype == torch.bool:
             sens = sens.long()
             
        # Identify heterogeneous edges (different sensitive attributes)
        hetero_edge_mask = (sens[row] != sens[col])
        hetero_row = row[hetero_edge_mask]
        hetero_col = col[hetero_edge_mask]
        
        num_hetero_edges = hetero_row.shape[0]

        # Build heterogeneous adjacency matrix
        if num_hetero_edges == 0:
            h_X = torch.zeros_like(data.features)
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=device) 
            return h_X, mask
        
        # Create bidirectional edges
        hetero_edges = torch.cat([
            torch.stack([hetero_row, hetero_col], dim=0),
            torch.stack([hetero_col, hetero_row], dim=0)
        ], dim=1)
        
        # Remove duplicate edges
        hetero_edges = torch.unique(hetero_edges, dim=1)
        
        # Create sparse matrix for heterogeneous edges
        hetero_adj_values = torch.ones(hetero_edges.shape[1]).to(device)
        hetero_adj = torch.sparse_coo_tensor(hetero_edges, 
                                             hetero_adj_values, 
                                             (num_nodes, num_nodes)).to(device) 
        
        h_sum = torch.spmm(hetero_adj, data.features)
        ones_vector = torch.ones(num_nodes, 1).to(device) 
        h_degree = torch.spmm(hetero_adj, ones_vector).squeeze()
        
        # Avoid division by zero
        mask = h_degree > 0
        h_degree_safe = h_degree.clone()
        h_degree_safe[~mask] = 1.0
        
        h_X = h_sum / h_degree_safe.unsqueeze(1)
        
        # Return full h_X and the mask
        return h_X, mask
    

    def evaluate(self, data, delta=1.0):
        val_results = self.evaluate_step(data, delta, split='val')
        test_results = self.evaluate_step(data, delta, split='test')
        return {**test_results, 'val_auc': val_results['auc']}
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3) 
        
    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x