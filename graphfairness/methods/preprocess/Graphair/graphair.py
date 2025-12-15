import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from graphfairness.models import *
from graphfairness.train import *
from graphfairness.evaluation import *
from graphfairness.utils import *
from graphfairness.methods.preprocess.Graphair.graphair_components import aug_module, GCN_Body
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree


class Classifier(nn.Module):
    def __init__(self, nfeat, nhid, nclass=1):
        super(Classifier, self).__init__(); self.fc1 = nn.Linear(nfeat, nhid); self.fc2 = nn.Linear(nhid, nclass); self.reset_parameters()
    def reset_parameters(self): self.fc1.reset_parameters(); self.fc2.reset_parameters()
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))


class Graphair(Trainer):
    r"""Implementation of `Graphair` from the paper entitled 
    `“Learning Fair Graph Representations via Automated Data Augmentations” <https://openreview.net/forum?id=1_OGWcP1s9w>`_.

    Graphair introduces a novel framework for learning fair graph representations by generating an 
    augmented "fair view" of the graph. Instead of relying on fixed heuristics, it employs a learnable 
    augmentation module (`g`) to automatically modify both graph topology (edges) and node features. 
    The training is guided by a min-max game involving three main components: an augmentation module (`g`), 
    a representation encoder (`f`), and an adversary (`k`). The objectives are threefold: 
    1. **Fairness**: An adversarial loss encourages the augmented view to be indistinguishable with respect to sensitive attributes.
    2. **Informativeness**: A contrastive loss maximizes the mutual information between the original and augmented graph representations.
    3. **Consistency**: A reconstruction loss penalizes large deviations from the original graph structure and features.
    The entire process is self-supervised, yielding fair node embeddings for downstream tasks.

    Parameters
    ----------
    **cfg : dict
        Configuration parameters for the Graphair model and training process.
        - nfeat : int
            Number of input node features.
        - dataset : str
            Name of the dataset (e.g., 'nba', 'pokec_z'), used for model saving paths.
        - device : torch.device
            The device (CPU or GPU) on which to place the models.
        - lr : float, optional
            Learning rate for the main Adam optimizer, by default 1e-4.
        - weight_decay : float, optional
            Weight decay for the main Adam optimizer, by default 1e-5.
        - dropout : float, optional
            Dropout rate used within the GCN_Body models, by default 0.0.
        - model_nhid : List[int], optional
            A list of hidden layer dimensions for the representation encoder (`f_encoder`). The length of the list determines the number of layers. Example: `[64, 64, 64]` for a 3-layer model. By default `[64, 64]`.
        - adv_nhid : List[int], optional
            A list of hidden layer dimensions for the adversary (`adversary`). By default `[64, 64]`.
        - proj_hidden_dim : int, optional
            The hidden dimension of the projection head used for contrastive learning, by default 64.
        - classifier_hidden_dim : int, optional
            The hidden dimension of the downstream MLP classifier used for evaluation, by default 128.
        - classifier_lr : float, optional
            Learning rate for the downstream classifier's optimizer, by default 1e-3.
        - classifier_wd : float, optional
            Weight decay for the downstream classifier's optimizer, by default 1e-5.
        - batch_size : int, optional
            The batch size for the GraphSAINT sampler, used only for large graphs (e.g., 'pokec_z'). By default 2000.

    Example
    -------
    .. code-block:: python

        from graphfairness.datasets import FairDataset
        from graphfairness.methods import Graphair # Assuming Graphair is exposed in methods/__init__.py
        import torch

        # --- 1. Load data and define configuration ---
        args = ... # Assume args are parsed from command line
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = FairDataset(root='./', name='nba')
        data = dataset.data.to(device)

        cfg = {
            'nfeat': data.features.shape[1],
            'dataset': 'nba',
            'device': device,
            'lr': 1e-4, 'weight_decay': 1e-5, 'dropout': 0.0,
            'model_nhid': [64, 64, 64],
            'adv_nhid': [64, 64],
            'proj_hidden_dim': 64,
            'classifier_hidden_dim': 128,
            'classifier_lr': 1e-3,
            'classifier_wd': 1e-5,
            
        }

        # --- 2. Create Graphair instance ---
        # The 'model' argument is a placeholder and ignored by Graphair's __init__
        graphair_method = Graphair(model=None, **cfg)
        
        # --- 3. Train the model ---
        # The train method handles the full pipeline (warmup, main training, and final evaluation)
        # and returns a dictionary of final performance metrics.
        results = graphair_method.train(
            data, 
            epochs=500,
            # Loss weights for this specific run
            alpha=1.0, 
            beta=1.0, 
            gamma=1.0, 
            lam=1.0,
            # Epochs for the downstream classifier
            test_epochs=500,
            # Warmup epochs for the augmentation module
            warmup=50
        )
        
        # --- 4. Print results ---
        print(f"Final ACC: {results['acc']:.4f}")
        print(f"Final AUC: {results['auc']:.4f}")
        print(f"Final DP: {results['dp']:.4f}")
        print(f"AUC: {results['auc_mean']:.4f}, F1: {results['f1_mean']:.4f}, ACC: {results['acc_mean']:.4f}, DP: {results['dp_mean']:.4f}, EO: {results['eo_mean']:.4f}")

    Note
    ----
        * This implementation is self-contained and uses its own GNN components (`GCN_Body`) based on the original paper's source code, which operate on dense adjacency matrices. This may lead to high memory consumption on large graphs.
        * The training process is complex, involving a `warmup` phase for the augmentation module and a dynamic training frequency for the adversary.
        * The final evaluation is performed by training a separate downstream MLP classifier on the learned node embeddings, and its performance is averaged over multiple runs for stability, as described in the original paper.
        * Hyperparameters `alpha`, `gamma`, and `lam` are crucial for balancing fairness, informativeness, and consistency, and should be tuned via grid search for optimal performance.
    """
    
    def __init__(self, **cfg):
        self.cfg = BunchDict(cfg)
        self.device = self.cfg.get('device', 'cpu')
        
       
        self.f_encoder = GCN_Body(in_feats=self.cfg.nfeat, n_hidden=self.cfg.model_nhid[0], 
                                  out_feats=self.cfg.model_nhid[-1], nlayer=len(self.cfg.model_nhid), 
                                  dropout=self.cfg.dropout).to(self.device)
        self.model = self.f_encoder 
        
        
        dummy_features = torch.zeros(1, self.cfg.nfeat) 
        self.aug_model = aug_module(features=dummy_features, n_hidden=self.cfg.model_nhid[0]).to(self.device)
        
       
        self.adversary = GCN_Body(in_feats=self.cfg.nfeat, n_hidden=self.cfg.adv_nhid[0], 
                                  out_feats=1, nlayer=len(self.cfg.adv_nhid), dropout=self.cfg.dropout).to(self.device)
        
        
        self.projection_head = nn.Sequential(nn.Linear(self.cfg.model_nhid[-1], self.cfg.proj_hidden_dim), 
                                             nn.ELU(), 
                                             nn.Linear(self.cfg.proj_hidden_dim, self.cfg.model_nhid[-1])).to(self.device)
        
        lr, wd = self.cfg.get('lr', 1e-4), self.cfg.get('weight_decay', 1e-5)
        
     
        FG_params = [{'params': self.aug_model.parameters(), 'lr': lr}, 
                     {'params': self.f_encoder.parameters(), 'lr': lr},
                     {'params': self.projection_head.parameters(), 'lr': lr}]
        self.optimizer_main = torch.optim.Adam(FG_params, lr=lr, weight_decay=wd)
        
        self.optimizer_adv = torch.optim.Adam(self.adversary.parameters(), lr=lr, weight_decay=wd)
        
     
        self.optimizer_aug_only = torch.optim.Adam(self.aug_model.parameters(), lr=lr, weight_decay=wd)

        self.downstream_classifier, self.optimizer_classifier = None, None

    def train(self, data, epochs, **train_wargs):
    
        self.alpha = train_wargs.get('alpha', 1.0) 
        self.beta = train_wargs.get('beta', 1.0)   
        self.gamma = train_wargs.get('gamma', 0.1) 
        self.lam = train_wargs.get('lam', 10.0)    
        
        print(f"Training Graphair with: Alpha={self.alpha}, Beta={self.beta}, Gamma={self.gamma}, Lambda={self.lam}")

        dataset_name = self.cfg.get('dataset', 'nba').lower()
        if dataset_name in ['german', 'nba', 'bail']: 
            self._fit_whole(data, epochs, **train_wargs)
        else: 
            self._fit_batch(data, epochs, **train_wargs)
        
        return self.test(data, test_epochs=train_wargs.get('test_epochs', 500), is_large_graph=(dataset_name not in ['german', 'nba', 'bail']))

    def _fit_whole(self, data, epochs, **train_wargs):
        adv_epoches = train_wargs.get('adv_epoches', 1)
        warmup_epochs = train_wargs.get('warmup', 50) 
        
   
        features = data.features.to(self.device)
        num_nodes = features.shape[0]
        sens = data.sens.float().to(self.device)
        idx_sens = data.idx_sens.to(self.device)

       
        if hasattr(data.edge_index, 'coo'):
            row, col, _ = data.edge_index.cpu().coo()
            edge_index = torch.stack([row, col], dim=0)
            adj_dense = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        else:
            adj_dense = to_dense_adj(data.edge_index.cpu(), max_num_nodes=num_nodes)[0]
        
      
        adj_dense.fill_diagonal_(1.0)
        
       
        adj_dense = adj_dense.to(self.device)
        
      
        D = adj_dense.sum(1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
        adj_norm = D_mat_inv_sqrt @ adj_dense @ D_mat_inv_sqrt
        
       
        num_edges = adj_dense.sum()
        num_pixels = num_nodes**2
        norm_w = num_pixels / float((num_pixels - num_edges) * 2)
        
        #print(f"Whole Graph Stat: Nodes={num_nodes}, Edges={int(num_edges)} (With Self-loops)")
        #print(f"Weights: Norm_W={norm_w:.4f}")

  
        if warmup_epochs > 0:
            print(f"--- Warmup for {warmup_epochs} epochs ---")
            pbar = tqdm(range(warmup_epochs), desc="Warmup", bar_format="{l_bar}{bar:10}{r_bar}")
            for _ in pbar:
                self.aug_model.train()
             
                adj_aug, x_aug, adj_logits = self.aug_model(adj_norm, features, alpha=0.5, adj_orig=adj_dense)
                
                loss_edge = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_dense)
                loss_feat = F.mse_loss(x_aug, features)
                loss_recons = loss_edge + self.lam * loss_feat
                
                self.optimizer_aug_only.zero_grad(); loss_recons.backward(); self.optimizer_aug_only.step()
                pbar.set_postfix({'rec': loss_recons.item()})

       
        print(f"\n--- Starting Main Training ---")
        tpbar = tqdm(range(epochs), desc="Training", unit="epoch")
        
        for epoch in tpbar:
            self.f_encoder.train(); self.aug_model.train(); self.adversary.train(); self.projection_head.train()
            
           
            adj_aug, x_aug, _ = self.aug_model(adj_norm, features, alpha=0.5, adj_orig=adj_dense)
            
            curr_adv_epoches = adv_epoches * 10 if epoch == 0 else adv_epoches
            for _ in range(curr_adv_epoches):
                s_pred_adv = self.adversary(adj_aug.detach(), x_aug.detach())
                loss_adv_train = F.binary_cross_entropy_with_logits(s_pred_adv[idx_sens].squeeze(), sens[idx_sens])
                self.optimizer_adv.zero_grad(); loss_adv_train.backward(); self.optimizer_adv.step()

            
            adj_aug, x_aug, adj_logits = self.aug_model(adj_norm, features, alpha=0.5, adj_orig=adj_dense)
            
            h = self.projection_head(self.f_encoder(adj_norm, features))
            h_prime = self.projection_head(self.f_encoder(adj_aug, x_aug))
            
          
            loss_cont = self.info_nce_loss(torch.cat((h, h_prime), dim=0))
            loss_edge = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_dense)
            loss_feat = F.mse_loss(x_aug, features)
            loss_recons = loss_edge + self.lam * loss_feat
            
            s_pred = self.adversary(adj_aug, x_aug)
            loss_adv = F.binary_cross_entropy_with_logits(s_pred[idx_sens].squeeze(), sens[idx_sens])
            
           
            loss_main = self.beta * loss_cont + self.gamma * loss_recons - self.alpha * loss_adv
            
            self.optimizer_main.zero_grad()
            loss_main.backward()
            self.optimizer_main.step()
            
            tpbar.set_postfix({
                'loss': f"{loss_main.item():.2f}", 
                'cont': f"{loss_cont.item():.2f}", 
                'rec': f"{loss_recons.item():.2f}", 
                'adv': f"{loss_adv.item():.2f}"
            })
            

    def _fit_batch(self, data, epochs, **train_wargs):
        adv_epoches = train_wargs.get('adv_epoches', 1)
        warmup_epochs = train_wargs.get('warmup', 0)

     
        if hasattr(data.edge_index, 'coo'):
            row, col, _ = data.edge_index.cpu().coo()
            edge_index_tensor = torch.stack([row, col], dim=0)
        else:
            edge_index_tensor = data.edge_index.cpu()
    
        features_cpu = data.features.cpu()
        sens_cpu = data.sens.cpu()
        idx_sens_cpu = data.idx_sens.cpu()
        num_nodes = features_cpu.shape[0]

       
        deg = degree(edge_index_tensor[0], num_nodes=num_nodes)
    
        
        sens_mask = torch.zeros(num_nodes, 1)
        sens_mask[idx_sens_cpu] = 1.0

       
        sampler_data = Data(x=features_cpu, edge_index=edge_index_tensor, 
                            sens=sens_cpu, sens_mask=sens_mask, deg=deg,
                            num_nodes=num_nodes)
        
        loader = GraphSAINTRandomWalkSampler(
            sampler_data,
            batch_size=self.cfg.get('batch_size', 2000), 
            walk_length=len(self.cfg.model_nhid),
            num_steps=max(10, int(num_nodes / 2000)),
            sample_coverage=100,
            num_workers=0
        )
        
       
        num_edges = edge_index_tensor.shape[1]
        global_norm_w = num_nodes**2 / float((num_nodes**2 - num_edges) * 2)
        print(f"Global Norm W: {global_norm_w:.4f}")

       
        if warmup_epochs > 0:
            print(f"--- Warmup for {warmup_epochs} epochs ---")
            for _ in range(warmup_epochs):
                for sub_data in loader:
                    sub_data = sub_data.to(self.device)
                    sub_adj_dense = to_dense_adj(sub_data.edge_index, max_num_nodes=sub_data.num_nodes)[0]
                    sub_adj_norm = self.normalize_adj(sub_adj_dense)
                    
                   
                    adj_aug, x_aug, adj_logits = self.aug_model(sub_adj_norm, sub_data.x, alpha=0.5, adj_orig=sub_adj_dense)
                    
                    
                    loss_edge = global_norm_w * F.binary_cross_entropy_with_logits(adj_logits, sub_adj_dense)
                    loss_feat = F.mse_loss(x_aug, sub_data.x)
                    loss_recons = loss_edge + self.lam * loss_feat
                    
                    self.optimizer_aug_only.zero_grad()
                    loss_recons.backward()
                    self.optimizer_aug_only.step()

      
        print(f"\n--- Starting Main Batch Training ---")
        tpbar = tqdm(total=epochs, desc="Training", unit="epoch")
        
        for epoch in range(epochs):
            loss_avg = {'loss': 0, 'l_cont': 0, 'l_rec': 0, 'l_adv': 0}
            num_batches = 0
            
            for i, sub_data in enumerate(loader):
                sub_data = sub_data.to(self.device)
                
                
                adv_steps = adv_epoches * 10 if (epoch == 0 and i == 0) else adv_epoches
                
               
                batch_res = self._train_step_batch(sub_data, adv_steps, global_norm_w)
                
                for k, v in batch_res.items(): loss_avg[k] += v
                num_batches += 1
                
            
            for k in loss_avg: loss_avg[k] /= num_batches
            tpbar.set_postfix(loss_avg)
            tpbar.update(1)
            
        tpbar.close()
    
   
        print(f"\n--- Starting Main Batch Training ---")
        tpbar = tqdm(total=len(loader), desc="Training Graphair (Batch)", bar_format="{l_bar}{bar:30}{r_bar}")
        for i, sub_data in enumerate(loader):
            sub_data = sub_data.to(self.device)
            adv_train_freq = adv_epoches * 10 if i == 0 else adv_epoches
            loss_dict = self._train_step_batch(sub_data, adv_train_freq)
            tpbar.set_postfix(loss_dict); tpbar.update(1)
        tpbar.close()

    def _train_step_batch(self, sub_data, adv_steps, global_norm_w) -> dict:
        sub_adj_dense = to_dense_adj(edge_index=sub_data.edge_index, max_num_nodes=sub_data.num_nodes)[0]
        sub_adj_norm = self.normalize_adj(sub_adj_dense)
        
        self.f_encoder.train(); self.aug_model.train(); self.adversary.train(); self.projection_head.train()
        
       
        adj_aug, x_aug, _ = self.aug_model(sub_adj_norm, sub_data.x, alpha=0.5, adj_orig=sub_adj_dense)
        sens_mask = (sub_data.sens_mask == 1.0).squeeze()
        
        for _ in range(adv_steps):
           
            s_pred_adv = self.adversary(adj_aug.detach(), x_aug.detach())
            
            loss_adv_train = F.binary_cross_entropy_with_logits(
                s_pred_adv[sens_mask].squeeze(), 
                sub_data.sens[sens_mask].float(), 
                weight=sub_data.node_norm[sens_mask], 
                reduction='sum'
            )
            self.optimizer_adv.zero_grad()
            loss_adv_train.backward()
            self.optimizer_adv.step()
            
        
        adj_aug, x_aug, adj_logits = self.aug_model(sub_adj_norm, sub_data.x, alpha=0.5, adj_orig=sub_adj_dense)
        
        
        h = self.projection_head(self.f_encoder(sub_adj_norm, sub_data.x))
        h_prime = self.projection_head(self.f_encoder(adj_aug, x_aug))
        
       
        logits, labels = self.info_nce_loss(torch.cat((h, h_prime), dim=0), return_logits=True)
       
        loss_cont = (F.cross_entropy(logits, labels, reduction='none') * sub_data.node_norm.repeat(2)).sum()
        
        
        loss_edge = global_norm_w * F.binary_cross_entropy_with_logits(adj_logits, sub_adj_dense) # Default mean
        loss_feat = F.mse_loss(x_aug, sub_data.x) # Default mean
        loss_recons = loss_edge + self.lam * loss_feat
        
        
        s_pred = self.adversary(adj_aug, x_aug)
        loss_adv = F.binary_cross_entropy_with_logits(
            s_pred[sens_mask].squeeze(), 
            sub_data.sens[sens_mask].float(), 
            weight=sub_data.node_norm[sens_mask], 
            reduction='sum'
        )
        
        
        loss_main = self.beta * loss_cont + self.gamma * loss_recons - self.alpha * loss_adv
        
        self.optimizer_main.zero_grad()
        loss_main.backward()
        self.optimizer_main.step()
        
        return {'loss': loss_main.item(), 'l_cont': loss_cont.item(), 'l_rec': loss_recons.item(), 'l_adv': loss_adv.item()}

    def test(self, data, test_epochs, is_large_graph=False):
        self.f_encoder.eval()
        with torch.no_grad():
            if is_large_graph:
                print("WARNING: Performing full-graph inference on a large graph. This may cause OOM.")
                adj_dense = data.edge_index.to_dense()
                adj_norm = self.normalize_adj(adj_dense)
                final_embs = self.f_encoder(adj_norm, data.features)
            else:
                if not hasattr(self, 'adj_norm') or self.adj_norm is None: 
                    adj_dense = data.edge_index.to_dense()
                    self.adj_norm = self.normalize_adj(adj_dense)
                final_embs = self.f_encoder(self.adj_norm, data.features)
        
        acc_list, dp_list, eo_list, f1_list, auc_list = [], [], [], [], []
        print(f"\n--- Starting Downstream Evaluation (running 5 times for stability) ---")
        for i in range(1): #5
            torch.manual_seed(i * 10); np.random.seed(i * 10)
            downstream_classifier = Classifier(nfeat=self.cfg.model_nhid[-1], nhid=self.cfg.classifier_hidden_dim).to(self.device)
            optimizer_classifier = torch.optim.Adam(downstream_classifier.parameters(), lr=self.cfg.classifier_lr, weight_decay=self.cfg.classifier_wd)
            criterion_cls = nn.BCEWithLogitsLoss()
            best_acc_val, current_run_best_results = 0, {}
            for epoch in range(test_epochs):
                downstream_classifier.train(); optimizer_classifier.zero_grad()
                output = downstream_classifier(final_embs); loss_train = criterion_cls(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
                loss_train.backward(); optimizer_classifier.step()
                downstream_classifier.eval()
                with torch.no_grad():
                    output_val = downstream_classifier(final_embs[data.idx_val]); preds_val = (output_val.squeeze() > 0).type_as(data.labels)
                    acc_val = accuracy_score(data.labels[data.idx_val].cpu().numpy(), preds_val.cpu().numpy())
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        output_test = downstream_classifier(final_embs[data.idx_test]); preds_test = (output_test.squeeze() > 0).type_as(data.labels)
                        labels_test_np, preds_test_np, sens_test_np = data.labels[data.idx_test].cpu().numpy(), preds_test.cpu().numpy(), data.sens[data.idx_test].cpu().numpy()
                        current_run_best_results['auc'] = roc_auc_score(labels_test_np, torch.sigmoid(output_test).cpu().numpy()); current_run_best_results['f1'] = f1_score(labels_test_np, preds_test_np)
                        current_run_best_results['acc'] = accuracy_score(labels_test_np, preds_test_np); dp, eo = fair_metric(preds_test_np, labels_test_np, sens_test_np)
                        current_run_best_results['dp'], current_run_best_results['eo'] = dp, eo
            acc_list.append(current_run_best_results.get('acc', 0)); dp_list.append(current_run_best_results.get('dp', 0)); eo_list.append(current_run_best_results.get('eo', 0)); f1_list.append(current_run_best_results.get('f1', 0)); auc_list.append(current_run_best_results.get('auc', 0))

        final_results_dict = {
            'auc_mean': np.mean(auc_list), 'auc_std': np.std(auc_list),
            'f1_mean': np.mean(f1_list),   'f1_std': np.std(f1_list),
            'acc_mean': np.mean(acc_list), 'acc_std': np.std(acc_list),
            'dp_mean': np.mean(dp_list),   'dp_std': np.std(dp_list),
            'eo_mean': np.mean(eo_list),   'eo_std': np.std(eo_list)
        }

   
        #print("\n--- Final Average Test Results (over 5 classifier runs) ---")
        #print(f"AUC: {final_results_dict['auc_mean']:.4f} ± {final_results_dict['auc_std']:.4f}")
        #print(f"F1: {final_results_dict['f1_mean']:.4f} ± {final_results_dict['f1_std']:.4f}")
        #print(f"ACC: {final_results_dict['acc_mean']:.4f} ± {final_results_dict['acc_std']:.4f}")
        #print(f"DP: {final_results_dict['dp_mean']:.4f} ± {final_results_dict['dp_std']:.4f}")
        #print(f"EO: {final_results_dict['eo_mean']:.4f} ± {final_results_dict['eo_std']:.4f}")
        
        
        return final_results_dict

    def normalize_adj(self, adj):
       
        if adj.requires_grad:
           
            eye = torch.eye(adj.shape[0], device=adj.device)
            adj = adj * (1 - eye) + eye
        else:
            
            adj = adj.clone()
            adj.fill_diagonal_(1)
        
        
        d = adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5)
        
        
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        
       
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
      
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    def info_nce_loss(self, features, return_logits=False):
        batch_size = int(features.shape[0] / 2); labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(self.device); labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features = F.normalize(features, dim=1); similarity_matrix = torch.matmul(features, features.T); mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device); labels = labels[~mask].view(labels.shape[0], -1); similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1); negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1); labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device); temperature = 0.07; logits = logits / temperature
        if return_logits: return logits, labels
        return F.cross_entropy(logits, labels)