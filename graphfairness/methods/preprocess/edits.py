# /graphfairness/methods/preprocess/edits.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
import warnings
import copy

from graphfairness.train import Trainer
from graphfairness.utils import BunchDict
from graphfairness.models import GCN
from torch_sparse import SparseTensor

warnings.filterwarnings('ignore')






class EDITS(Trainer):
    r"""Implementation of `EDITS` from the paper entitled `“EDITS: Modeling and Mitigating Data Bias 
    for Graph Neural Networks” <https://arxiv.org/abs/2108.05233>`.

    EDITS is a **model-agnostic pre-processing framework** that aims to debias the input attributed network 
    itself, so that any downstream GNN trained on the processed data can achieve fairer results. 
    It mitigates bias from two data modalities: node attributes and network structure. The core idea 
    is to use adversarial training to learn a debiased feature representation and a debiased graph 
    structure simultaneously, such that the processed data contains minimal information about sensitive 
    attributes. 

    Parameters
    ----------
    model : nn.Module
        The downstream GNN model to be trained on the debiased graph. This wrapper will
        rebuild the model to match the final feature dimension after pre-processing.
    **cfg : dict
        Configuration parameters passed from the command line. Key parameters include:
        - lr : float
            Learning rate for the downstream GNN optimizer.
        - weight_decay : float
            Weight decay for the downstream GNN optimizer.
        - nfeat : int
            Number of input features of the original graph.
        - node_num : int
            Number of nodes in the graph.
        - nhid : list
            A list of hidden dimensions for the GCN backbone.
        - dropout : float
            Dropout rate for the downstream GNN.
        - lr_edits : float
            Learning rate for the EDITS pre-processing stage.
        - edits_epochs : int
            Number of epochs for the EDITS pre-processing stage.
        - recon_weight : float
            Hyperparameter controlling the weight of the feature reconstruction loss. Corresponds to `μ₁` in the paper.
        - adv_weight_feat : float
            Hyperparameter controlling the weight of the adversarial loss on features.
        - fro_weight : float
            Hyperparameter controlling the weight of the structural similarity loss. Corresponds to `μ₃` in the paper.
        - adv_weight_adj : float
            Hyperparameter controlling the weight of the adversarial loss on structure.
        - threshold_prop : float
            The proportion threshold 'r' for the `binarize_adj` post-processing step.
        
        
    Example
    -------
    .. code-block:: python

        from graphfairness.methods.preprocess import EDITS
        from graphfairness.models import GCN
        from graphfairness.datasets import FairDataset

        # Load data
        dataset = FairDataset(root='./data', name='bail')
        data = dataset.data
        n_feat = data.features.shape[1]
        n_nodes = data.features.shape[0]

        # Initialize a placeholder GNN backbone. It will be rebuilt inside EDITS.
        gnn_model = GCN(nfeat=n_feat, nhid=[16], nclass=1, dropout=0.5)
        
        # Create EDITS instance with all necessary configs
        fair_model = EDITS(
            gnn_model, 
            dataset='bail',
            nfeat=n_feat, 
            node_num=n_nodes,
            nhid=[16],
            dropout=0.5,
            # ... other necessary parameters from command line ...
        )
        
        # Train the model. This will first run EDITS pre-processing, 
        # then train the downstream GCN.
        fair_model.train(data, epochs=1000, validation=True)
        
        # Evaluate the model on the test set
        metrics = fair_model.evaluate(data)
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Demographic Parity: {metrics['dp']:.4f}")
        print(f"Equal Opportunity: {metrics['eo']:.4f}")

    Note
    ----
    * This implementation is a **pre-processing** method. The `train` method encapsulates the entire
      pipeline: EDITS adversarial training, a multi-step post-processing pipeline derived from the
      original source code, GNN rebuilding, and finally the
      downstream GNN training.
    * The set of hyperparameters (`recon_weight`, `adv_weight_feat`, `threshold_prop`, etc.) is
      highly sensitive to the dataset and requires careful tuning to find the optimal balance
      between utility (e.g., AUC) and fairness (e.g., Parity).
    * This implementation requires all hyperparameters for the EDITS algorithm to be passed
      through the `**cfg` dictionary during initialization.
    """

    def __init__(self, model, **cfg):
        super().__init__(model, **cfg)
        self.cfg = BunchDict(cfg)
        edits_args = BunchDict({'lr': self.cfg.get('lr_edits', 0.003), 'weight_decay': self.cfg.get('wd_edits', 1e-7), 'device': self.cfg.device})
        self.edits_model = EDITS_model(args=edits_args, nfeat=self.cfg.nfeat, node_num=self.cfg.node_num, nclass=1, nfeat_out=int(self.cfg.node_num / 10), adj_lambda=self.cfg.get('adj_lambda', 1e-1), layer_threshold=self.cfg.get('layer_threshold', 2), dropout=self.cfg.get('dropout_edits', 0.2))
        self.edits_model.cfg = self.cfg
        self.edits_model.adj_renew.cfg = self.cfg
        self.final_features = None
        self.final_edge_index = None
        self.final_edge_weight = None

       
        dataset_name = self.cfg.get('dataset', 'unknown_dataset')
        current_seed = self.cfg.get('current_seed', 0)
        self.weight_path = f'./weights/{dataset_name}_seed{current_seed}_best_model.pt'
        print(f"INFO: Weights will be saved to and loaded from: {self.weight_path}")

    def train(self, data, epochs, validation=True, **train_wargs):
        
        edits_epochs = train_wargs.get('edits_epochs', 500)
        lr_edits = self.cfg.get('lr_edits', 0.003)
        print("--- Running EDITS pre-processing ---")
        self.edits_model = self.edits_model.to(data.features.device)
        print(">>> Step 1/5 (EDITS Input): Normalizing features using L2 column norm.")
        features_for_edits = data.features / data.features.norm(dim=0)
        features_for_edits = torch.nan_to_num(features_for_edits, nan=0.0)
        num_nodes = data.features.shape[0]
        if isinstance(data.edge_index, SparseTensor):
            row, col, _ = data.edge_index.coo()
        else:
            row, col = data.edge_index[0], data.edge_index[1]
        adj_ori = sp.coo_matrix((np.ones(row.shape[0]), (row.cpu(), col.cpu())), shape=(num_nodes, num_nodes), dtype=np.float32)
        adj_ori = adj_ori + adj_ori.T.multiply(adj_ori.T > adj_ori) - adj_ori.multiply(adj_ori.T > adj_ori) + sp.eye(adj_ori.shape[0])
        adj_tensor_train = sparse_mx_to_torch_sparse_tensor(adj_ori).to(data.features.device)
        print(">>> Step 2/5 (EDITS Training): Running adversarial optimization.")
        tpbar_edits = tqdm(total=edits_epochs, desc="EDITS Pre-processing", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        for epoch in range(edits_epochs):
            if epoch > 400:
                lr_edits = 0.001
            losses = self.edits_model.optimize(adj_tensor_train, features_for_edits, data.idx_train, data.sens, epoch, lr_edits)
            tpbar_edits.set_postfix({'loss_G': f"{losses['loss_g']:.4f}", 'loss_A': f"{losses['loss_a']:.4f}"})
            tpbar_edits.update(1)
        tpbar_edits.close()

        
        print("--- Applying post-processing steps from original pipeline ---")
        self.edits_model.eval()
        with torch.no_grad():
            A_debiased_tensor, _, _, _, _ = self.edits_model(adj_tensor_train, features_for_edits)
            print(">>> Step 3a/5 (Post-proc): Zeroing out most biased features (from train.py).")
            param = self.edits_model.state_dict()
            z_threshold = self.cfg.get('z_threshold', 0)
            indices_to_zero = torch.argsort(param["x_debaising.s"])[:z_threshold]
            features_after_zeroing = data.features.clone().to(self.cfg.device)
            for i in indices_to_zero:
                features_after_zeroing[:, i] = 0
            print(">>> Step 3b/5 (Post-proc): Applying 'binarize_adj' and normalization to graph.")
            threshold_prop = self.cfg.get('threshold_prop')
            if threshold_prop is None:
                raise ValueError("threshold_prop must be provided.")
            A_binarized = binarize_adj(A_debiased_tensor, adj_ori, threshold_prop)
            A_final_normalized = normalize_scipy(A_binarized)
            A_final_coo = A_final_normalized.tocoo()
            final_edge_index = torch.from_numpy(np.vstack((A_final_coo.row, A_final_coo.col))).long().to(self.cfg.device)
            final_edge_weight = torch.from_numpy(A_final_coo.data).float().to(self.cfg.device)
            print(">>> Step 3c/5 (Post-proc): Applying feature selection (nonzero).")
            nonzero_mask = features_after_zeroing.sum(axis=0) != 0
            features_after_nonzero = features_after_zeroing[:, nonzero_mask]
            if self.cfg.dataset != 'german':
                print(f">>> Step 3d/5 (Post-proc): Applying final feature_norm for '{self.cfg.dataset}'.")
                min_vals = features_after_nonzero.min(axis=0, keepdim=True).values
                max_vals = features_after_nonzero.max(axis=0, keepdim=True).values
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1
                final_features = 2 * (features_after_nonzero - min_vals) / range_vals - 1
            else:
                print(">>> Step 3d/5 (Post-proc): Skipping final feature_norm for 'german'.")
                final_features = features_after_nonzero
            print(f">>> Final feature dimension for GCN: {final_features.shape[1]}")
            self.final_features, self.final_edge_index, self.final_edge_weight = final_features, final_edge_index, final_edge_weight


        print("--- Step 4/5 (Downstream Training): Rebuilding and training GCN. ---")
        gcn_nhid = self.cfg.get('nhid', [16])
        gcn_dropout = self.cfg.get('dropout', 0.05)
        self.model = GCN(nfeat=self.final_features.shape[1], nhid=gcn_nhid, nclass=1, dropout=gcn_dropout).to(self.cfg.device)
        lr, weight_decay = self.cfg.get('lr', 1e-3), self.cfg.get('weight_decay', 1e-5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        print("--- Step 5/5 (Downstream Training): Starting GCN training loop. ---")
        debiased_data = copy.copy(data)
        debiased_data.features = self.final_features
        debiased_data.edge_index = self.final_edge_index
        debiased_data.edge_weight = self.final_edge_weight
        if hasattr(debiased_data, 'adj_t'):
            del debiased_data.adj_t
        super().train(debiased_data, epochs, validation)

    @torch.no_grad()
    def evaluate_step(self, data, is_predict=False):
        self.model.eval()
        if self.final_features is None or self.final_edge_index is None:
            raise RuntimeError("Train the model first to get the debiased graph.")
        output = self.model(self.final_features, self.final_edge_index, self.final_edge_weight)
        if is_predict:
            return output
        else:
            return self.criterion(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())

def normalize_scipy(mx):
    
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def binarize_adj(A_debiased, adj_ori, threshold_proportion):
    
    A_debiased_np = A_debiased.detach().cpu().numpy()
    the_con1 = A_debiased_np - adj_ori.toarray()
    
    pos_max = np.max(the_con1)
    neg_min = np.min(the_con1)

    if pos_max == 0 and neg_min == 0:
        return adj_ori.tocoo()

    the_con1 = np.where(the_con1 > pos_max * threshold_proportion, 1, the_con1)
    the_con1 = np.where(the_con1 < neg_min * threshold_proportion, -1, the_con1)
    the_con1 = np.where(np.abs(the_con1) != 1, 0, the_con1)

    A_final = adj_ori + sp.coo_matrix(the_con1)
    
    assert A_final.max() <= 1, "Matrix values should not exceed 1"
    assert A_final.min() >= 0, "Matrix values should not be less than 0"
    
    return A_final

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class PGD(torch.optim.Optimizer):
    def __init__(self, params, proxs, lr=1e-3, alphas=[]):
        defaults = dict(lr=lr, alphas=alphas); super(PGD, self).__init__(params, defaults); self.proxs = proxs
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p, prox, alpha in zip(group['params'], self.proxs, group['alphas']): p.data = prox(p.data, alpha=alpha * group['lr'])

class prox_operators:
    @staticmethod
    def prox_l1(x, alpha): return torch.sign(x) * torch.maximum(torch.abs(x) - alpha, torch.zeros_like(x))
    @staticmethod
    def prox_nuclear(x, alpha): U, S, V = torch.svd(x); S = torch.sign(S) * torch.maximum(torch.abs(S) - alpha, torch.zeros_like(S)); return U @ torch.diag(S) @ V.T

class X_debaising(nn.Module):
    def __init__(self, in_features): super(X_debaising, self).__init__(); self.in_features = in_features; self.s = Parameter(torch.FloatTensor(in_features), requires_grad=True); self.reset_parameters()
    def reset_parameters(self): self.s.data.uniform_(1, 1)
    def forward(self, feature): return torch.mm(feature, torch.diag(self.s))

class EstimateAdj(nn.Module):
    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__(); n = adj.shape[0]; self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n), requires_grad=True); self._init_estimation(adj); self.symmetric = symmetric; self.device = device
    def _init_estimation(self, adj):
        with torch.no_grad():
            if adj.is_sparse: self.estimated_adj.data.copy_(adj.to_dense())
            elif isinstance(adj, torch.Tensor): self.estimated_adj.data.copy_(adj)
            else:
                if sp.issparse(adj): adj = adj.toarray()
                self.estimated_adj.data.copy_(torch.from_numpy(adj))
    def forward(self): return self.estimated_adj

class Adj_renew(nn.Module):
    def __init__(self, node_num, nfeat, nfeat_out, adj_lambda, device):
        super(Adj_renew, self).__init__(); self.device = device; self.node_num, self.nfeat, self.nfeat_out, self.adj_lambda = node_num, nfeat, nfeat_out, adj_lambda
    def fit(self, adj, lr):
        self.estimator = EstimateAdj(adj, symmetric=False, device=self.device).to(self.device); self.optimizer_adj = optim.SGD(self.estimator.parameters(), momentum=0.9, lr=lr); self.optimizer_l1 = PGD(self.estimator.parameters(), proxs=[prox_operators.prox_l1], lr=lr, alphas=[5e-4])
    def forward(self): return self.estimator.estimated_adj
    def train_adj(self, features, adj, adv_loss, epoch, lr):
        for param_group in self.optimizer_adj.param_groups: param_group["lr"] = lr
        self.estimator.train(); self.optimizer_adj.zero_grad(); adj_dense = adj.to_dense().to(self.device) if adj.is_sparse else adj.to(self.device); delta = self.estimator.estimated_adj - adj_dense
        loss_fro = torch.sum(delta.mul(delta)); fro_weight = self.cfg.get('fro_weight'); adv_weight_adj = self.cfg.get('adv_weight_adj')
        if fro_weight is None or adv_weight_adj is None: raise ValueError("fro_weight and adv_weight_adj must be provided.")
        loss_diffiential = fro_weight * loss_fro + adv_weight_adj * adv_loss; loss_diffiential.backward(retain_graph=True)
        self.optimizer_adj.step(); self.optimizer_l1.zero_grad(); self.optimizer_l1.step()
        self.estimator.estimated_adj.data.copy_(torch.clamp(self.estimator.estimated_adj.data, min=0, max=1))
        self.estimator.estimated_adj.data.copy_((self.estimator.estimated_adj.data + self.estimator.estimated_adj.data.transpose(0, 1)) / 2)

class EDITS_model(nn.Module):
    def __init__(self, args, nfeat, node_num, nclass, nfeat_out, adj_lambda, layer_threshold=2, dropout=0.2):
        super(EDITS_model, self).__init__(); self.device = args.device; self.x_debaising = X_debaising(nfeat); self.layer_threshold = layer_threshold; self.adj_renew = Adj_renew(node_num, nfeat, nfeat_out, adj_lambda, self.device); self.fc = nn.Linear(nfeat * (layer_threshold + 1), nclass); self.lr = args.lr
        self.optimizer_feature_l1 = PGD(self.x_debaising.parameters(), proxs=[prox_operators.prox_l1], lr=self.lr, alphas=[5e-6])
        G_params = list(self.x_debaising.parameters()); self.optimizer_G = torch.optim.RMSprop(G_params, lr=self.lr, eps=1e-04, weight_decay=args.weight_decay); self.optimizer_A = torch.optim.RMSprop(self.fc.parameters(), lr=self.lr, eps=1e-04, weight_decay=args.weight_decay); self.dropout = nn.Dropout(dropout)
    def propagation_cat_new_filter(self, X_de, A_norm, layer_threshold):
        X_agg, X_de_current = X_de, X_de
        for i in range(layer_threshold): X_de_current = A_norm @ X_de_current; X_agg = torch.cat((X_agg, X_de_current), dim=1)
        return X_agg
    def forward(self, A, X):
        X_de = self.x_debaising(X); adj_new = self.adj_renew(); agg_con = self.propagation_cat_new_filter(X_de, adj_new, layer_threshold=self.layer_threshold)
        D_pre = self.fc(agg_con); D_pre = self.dropout(D_pre); return adj_new, X_de, D_pre, D_pre, agg_con
    def optimize(self, adj, features, idx_train, sens, epoch, lr):
        self.lr = lr;
        for param_group in self.optimizer_G.param_groups: param_group["lr"] = lr
        for param_group in self.optimizer_A.param_groups: param_group["lr"] = lr
        self.train(); self.optimizer_G.zero_grad(); self.fc.requires_grad_(False)
        if epoch == 0: self.adj_renew.fit(adj, self.lr)
        
        _, X_debiased, predictor_sens, _, _ = self.forward(adj, features)
        
        
        predictor_sens_train = predictor_sens[idx_train]
        positive_eles = torch.masked_select(predictor_sens_train.squeeze(), sens[idx_train] > 0)
        negative_eles = torch.masked_select(predictor_sens_train.squeeze(), sens[idx_train] <= 0)
        adv_loss_g = - (torch.mean(positive_eles) - torch.mean(negative_eles))
        
        recon_weight = self.cfg.get('recon_weight'); adv_weight_feat = self.cfg.get('adv_weight_feat')
        if recon_weight is None or adv_weight_feat is None: raise ValueError("recon_weight and adv_weight_feat must be provided.")
        loss_g = recon_weight * (X_debiased - features).norm(2) + adv_weight_feat * adv_loss_g
        loss_g.backward(retain_graph=True); self.optimizer_G.step(); self.optimizer_feature_l1.zero_grad(); self.optimizer_feature_l1.step()
        
        _, X_debiased, predictor_sens, _, _ = self.forward(adj, features)
        
        
        predictor_sens_train = predictor_sens[idx_train]
        positive_eles = torch.masked_select(predictor_sens_train.squeeze(), sens[idx_train] > 0)
        negative_eles = torch.masked_select(predictor_sens_train.squeeze(), sens[idx_train] <= 0)
        adv_loss_adj = - (torch.mean(positive_eles) - torch.mean(negative_eles))
        
        self.adj_renew.train_adj(X_debiased, adj, adv_loss_adj, epoch, lr)
        param = self.state_dict(); param["x_debaising.s"] = torch.clamp(param["x_debaising.s"], min=0, max=1); self.load_state_dict(param)
        loss_a_val = 0
        for i in range(8):
            self.fc.requires_grad_(True); self.optimizer_A.zero_grad(); 
            _, _, predictor_sens, _, _ = self.forward(adj, features)

            
            predictor_sens_train = predictor_sens[idx_train]
            positive_eles = torch.masked_select(predictor_sens_train.squeeze(), sens[idx_train] > 0)
            negative_eles = torch.masked_select(predictor_sens_train.squeeze(), sens[idx_train] <= 0)

            loss_a = torch.mean(positive_eles) - torch.mean(negative_eles); loss_a.backward(retain_graph=True); self.optimizer_A.step()
            for p in self.fc.parameters(): p.data.clamp_(-0.02, 0.02)
            if i == 7: loss_a_val = -loss_a.item()
        return {"loss_g": loss_g.item(), "loss_a": loss_a_val, "adv_loss": adv_loss_g.item()}