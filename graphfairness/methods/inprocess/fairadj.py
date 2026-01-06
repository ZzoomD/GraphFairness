import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from graphfairness.train import Trainer
from graphfairness.utils import BunchDict
from graphfairness.evaluation.metrics import fair_metric


def grl_hook():
    def fun1(grad):
        return -1 * grad.clone()
    return fun1

class FairAdj(Trainer):
    r"""Implementation of `FairAdj` from the paper entitled `“On Dyadic Fairness: 
    Exploring and Mitigating Bias in Graph Connections” <https://arxiv.org/abs/2102.00845>`.

    FairAdj is a framework designed to achieve dyadic fairness in graph-structured data by 
    empirically learning a fair adjacency matrix. The core idea is that a predictive 
    relationship between two instances should be independent of their sensitive attributes. 
    It reveals that regulating weights on existing edges in a graph contributes to 
    fairness conditionally.

    The algorithm employs a dual-optimization strategy:
    1. Utility Optimization (T1): Optimizes GNN parameters to preserve predictive accuracy 
       (reconstruction of graph structure and node classification).
    2. Fairness Optimization (T2): Learns a fair adjacency matrix by updating the normalized 
       weights under structural constraints and a right stochastic constraint using 
       projected gradient descent.

    Parameters
    ----------
    model : nn.Module
        The FairAdjVAE model consisting of an encoder, decoder, and classification head.
    **cfg : dict
        Additional configuration parameters:
        - lr : float, optional
            Learning rate for model parameter optimization, by default 1e-2.
        - eta : float, optional
            Learning rate for the fair adjacency matrix optimization, by default 0.1.
        - outer_epochs : int, optional
            Number of outer co-adaptation cycles, by default 4.
        - T1 : int, optional
            Number of iterations for utility optimization in each cycle, by default 50.
        - T2 : int, optional
            Number of iterations for fairness optimization in each cycle, by default 20.
        - weight_path : str, optional
            Path to save the best model weights.

    Example
    -------
    .. code-block:: python

        from graphfairness.methods.inprocess.fairadj import FairAdj, FairAdjVAE
        from graphfairness.datasets import FairDataset

        # Load data
        dataset = FairDataset(root='./data', name='german')
        data = dataset.data

        # Initialize FairAdjVAE model
        nfeat = data.features.shape[1]
        model = FairAdjVAE(nfeat=nfeat, nhid1=32, nhid2=16, dropout=0.1, nclass=1)

        # Initialize FairAdj trainer
        trainer = FairAdj(model, lr=0.01, eta=0.1, T1=50, T2=20, outer_epochs=4)
        
        # Train and evaluate
        trainer.train(data, epochs=None, validation=True)
        metrics = trainer.evaluate(data)

    Note
    ----
    * FairAdj preserves the original graph structure (it only adapts weights of existing edges).
    * The adjacency matrix is constrained to be a right stochastic matrix to ensure 
      numerical stability and valid message passing.
    * This implementation adapts the original link-prediction-focused FairAdj to node 
      classification tasks while maintaining structural fairness.
    """
    def __init__(self, model, **cfg):
        super().__init__(model)
        self.cfg = BunchDict(cfg)
        self.lr = self.cfg.get('lr', 0.01)
        self.eta = self.cfg.get('eta', 0.1)
        self.T1 = self.cfg.get('T1', 50)
        self.T2 = self.cfg.get('T2', 20)
        self.outer_epochs = self.cfg.get('outer_epochs', 4)
        self.device = self.cfg.get('device', 'cpu')
        self.weight_path = self.cfg.get('weight_path', './weights/best_fairadj.pt')
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, data, epochs=None, validation=True, **train_wargs):
        
        adj_sparse = data.edge_index.to_scipy(layout='coo')
        n_nodes = data.features.shape[0]
        
        
        adj_with_eye = adj_sparse + sp.eye(n_nodes)
        self.adj_label = torch.FloatTensor(adj_with_eye.toarray()).to(self.device)
        adj_sum = adj_sparse.sum()
        self.pos_weight = torch.Tensor([(n_nodes**2 - adj_sum) / adj_sum]).to(self.device)
        self.norm = n_nodes**2 / float((n_nodes**2 - adj_sum) * 2)

        
        self.adj_norm = self._preprocess_graph(adj_sparse).to(self.device)
        self.intra_pos, self.inter_pos = self._find_link_indices(adj_sparse, data.sens)

        self.model.to(self.device)
        best_auc_val = 0.0
        
        
        for out_epoch in range(self.outer_epochs):
            
            t1_pbar = tqdm(total=self.T1, desc=f"Outer {out_epoch+1} T1", bar_format="{l_bar}{bar:20}{r_bar}")
            for _ in range(self.T1):
                loss_val = self._train_step_t1(data)
                t1_pbar.set_postfix(loss=f"{loss_val:.4f}")
                t1_pbar.update(1)
            t1_pbar.close()

            
            t2_pbar = tqdm(total=self.T2, desc=f"Outer {out_epoch+1} T2", bar_format="{l_bar}{bar:20}{r_bar}")
            for _ in range(self.T2):
                loss_f = self._train_step_t2(data)
                t2_pbar.set_postfix(fair_loss=f"{loss_f:.6f}")
                t2_pbar.update(1)
            t2_pbar.close()

          
            if validation:
                ret = self.evaluate_step(data)
                if ret["auc_val"] > best_auc_val:
                    best_auc_val = ret["auc_val"]
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.weight_path)

    def _train_step_t1(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        recovered, z, mu, logvar, cls_out = self.model(data.features, self.adj_norm)
        cost = self.norm * F.binary_cross_entropy_with_logits(recovered, self.adj_label, pos_weight=self.pos_weight)
        kld = -0.5 / data.features.shape[0] * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - torch.exp(logvar).pow(2), 1))
        loss_cls = self.criterion(cls_out[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())

        total_loss = cost + kld + loss_cls
        # total_loss = cost + kld 
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return total_loss.item()

    def _train_step_t2(self, data):
        
        self.model.eval()
      
        adj_dense = self.adj_norm.to_dense().detach().requires_grad_(True)
        
        
        out = self.model(data.features, adj_dense)
        recovered = out[0]
        
        intra_score = recovered[self.intra_pos[:, 0], self.intra_pos[:, 1]].mean()
        inter_score = recovered[self.inter_pos[:, 0], self.inter_pos[:, 1]].mean()
        
        loss_dyadic = F.mse_loss(intra_score, inter_score)
        loss_dyadic.backward()
        
        with torch.no_grad():
            
            updated_adj = adj_dense.add(adj_dense.grad.mul(-self.eta))
            
            for i in range(updated_adj.shape[0]):
                updated_adj[i] = self._project_simplex(updated_adj[i])
            self.adj_norm = updated_adj.to_sparse().detach()
            
        return loss_dyadic.item()

    @torch.no_grad()
    def evaluate_step(self, data, is_predict=False):
      
        self.model.eval()
        out = self.model(data.features, self.adj_norm)
        cls_out = out[4] 
        
       
        output = cls_out.detach()
        
        if is_predict:
            return output
        
        preds = (output.squeeze() > 0).type_as(data.labels)
        idx_test = data.idx_test.cpu() 
        
       
        y_true = data.labels.cpu().numpy()[idx_test]
        y_score = output.cpu().numpy()[idx_test]
        y_pred = preds.cpu().numpy()[idx_test]
        s_test = data.sens.cpu().numpy()[idx_test]

        auc_test = roc_auc_score(y_true, y_score)
        f1_test = f1_score(y_true, y_pred)
        acc_test = accuracy_score(y_true, y_pred)
        dp_test, eo_test = fair_metric(y_pred, y_true, s_test)
        
        return dict(auc_val=auc_test, f1_val=f1_test, acc_val=acc_test, 
                    dp_val=dp_test, eo_val=eo_test)

   

    def _project_simplex(self, y):
        
        des, _ = torch.sort(y, descending=True)
        cumsum = torch.cumsum(des, dim=0)
        pos = torch.ones(des.shape[0]).to(y.device) / torch.arange(1, des.shape[0] + 1).to(y.device)
        rho = des + pos * (1.0 - cumsum)
        rho_val = (rho > 0).sum()
        lambda_ = (1. / rho_val.float()) * (1. - cumsum[rho_val - 1])
        x = y + lambda_
        x[x < 0.] = 0.
        return x

    def _preprocess_graph(self, adj):
        
        adj_ = sp.coo_matrix(adj) + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        d_inv = sp.diags(np.power(rowsum, -1.0).flatten())
        adj_norm = d_inv.dot(adj_).tocoo()
        
        indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
        values = torch.from_numpy(adj_norm.data.astype(np.float32))
        return torch.sparse.FloatTensor(indices, values, torch.Size(adj_norm.shape))

    def _find_link_indices(self, adj, sensitive):
      
        sens = sensitive.cpu().numpy()
        s_mat = (sens[:, None] == sens[None, :])
        intra = np.argwhere(s_mat == True)
        inter = np.argwhere(s_mat == False)
        return torch.LongTensor(intra).to(self.device), torch.LongTensor(inter).to(self.device)



class FairAdjVAE(nn.Module):
    r"""Variational Graph Auto-Encoder (VGAE) architecture customized for `FairAdj`.

    This model serves as the backbone for FairAdj, integrating graph representation 
    learning with classification and structural fairness. It consists of a 
    GCN-based encoder to generate latent embeddings (mu and log-variance), an 
    inner-product decoder for link reconstruction, and a classification head 
    for downstream node classification tasks.

    Parameters
    ----------
    nfeat : int
        Dimension of the input node features.
    nhid1 : int
        Dimension of the first hidden GCN layer.
    nhid2 : int
        Dimension of the latent space (latent embedding size).
    dropout : float
        Dropout probability for regularization.
    nclass : int, optional
        Number of output classes for node classification, by default 1.

    Attributes
    ----------
    gc1 : nn.Linear
        Weights for the first GCN transformation.
    gc2_mu : nn.Linear
        Weights for the mean latent vector generation.
    gc2_logvar : nn.Linear
        Weights for the log-variance latent vector generation.
    classifier : nn.Sequential
        A single-layer linear classification head that maps latent embeddings to node classes.
    """
    def __init__(self, nfeat, nhid1, nhid2, dropout, nclass=1):
        super().__init__()
       
        self.gc1 = nn.Linear(nfeat, nhid1)
        self.gc2_mu = nn.Linear(nhid1, nhid2)
        self.gc2_logvar = nn.Linear(nhid1, nhid2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(nhid2, nclass)
        
       

    def forward(self, x, adj):
        def gcn_op(x_, a_):
            if a_.is_sparse: return torch.spmm(a_, x_)
            return torch.matmul(a_, x_)

        h = F.relu(gcn_op(self.dropout(x), adj))
        h = self.gc1(h)
        
        mu = self.gc2_mu(h)
        logvar = self.gc2_logvar(h)
        logvar = torch.clamp(logvar, -10, 10)

        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu) if self.training else mu
        recovered = torch.matmul(z, z.t())
        cls_out = self.classifier(z)
        return recovered, z, mu, logvar, cls_out