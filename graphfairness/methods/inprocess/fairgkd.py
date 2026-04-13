import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
from torch_geometric.nn import GCNConv
from graphfairness.train import Trainer
from torch.nn.modules.loss import _Loss
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from graphfairness.evaluation import *

class FairGKD(Trainer):
    r"""Implementation of `FairGKD` from the paper entitled `“The Devil is in the Data: Learning 
    Fair Graph Neural Networks via Partial Knowledge Distillation” <https://arxiv.org/pdf/2311.17373>`.
    
    FairGKD is a fairness-aware framework that distills knowledge from a "Synthetic Teacher" 
    into a student GNN. The synthetic teacher is constructed by partial data training which is found 
    improving the fairness of trained models. Specifically, the teacher consists of two experts: 
    a MLP expert to reduce structural bias and a GNN expert to reduce feature bias. 
    A projector merges these fair representations for the student to emulate.

    Parameters
    ----------
    model : nn.Module
        The student GNN model to be trained.
    **cfg : dict
        Additional configuration parameters:
        - lr : float, optional
            Learning rate for optimization, by default 1e-3.
        - weight_decay : float, optional
            Weight decay for regularization, by default 1e-5.
        - tem : float, optional
            Temperature parameter for contrastive loss, by default 0.5.
        - lr_w : float, optional
            Learning rate for the adaptive weight module, by default 0.025.
        - gamma : float, optional
            Hyperparameter to enhance the disadvantaged loss, by default 0.25.
        - nfeat : int
            Number of input features.
        - nhid : list of int, optional
            Hidden layer dimensions, by default [16].
        - nclass : int, optional
            Number of target classes, by default 1.

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
        
        # Initialize FairGKD with configuration
        config = {
            'nfeat': n_feat,
            'nclass': 1,
            'nhid': [16],
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'tem': 0.5,
            'lr_w': 0.025,
            'gamma': 0.25
        }
        fair_model = FairGKD(model=model, **config)
        
        # Train the model
        fair_model.train(fair_dataset, epochs=200, validation=True)
        
        # Evaluate the model
        metrics = fair_model.evaluate(fair_dataset)
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Demographic Parity: {metrics['dp']:.4f}")
        print(f"Equal Opportunity: {metrics['eo']:.4f}")

    Note
    ----
    * The method involves a two-stage process: first training the experts and the projector 
      to create a fair teacher, then training the student GNN using adaptive distillation.
    * The `SynTeacher` generates fair embeddings by masking topology (MLP expert) and 
      neutralizing features (GNN expert on all-ones features).
    * `AdaWeight` dynamically balances the trade-off between task accuracy and fairness alignment.
    """
    def __init__(self, model, **cfg):
        """
        Internally builds: Vanilla GNN, Synthetic Teacher, and Projector.
        """
        super().__init__(model, **cfg)
        
        # hyperparameters
        self.lr = self.cfg.get('lr', 1e-3)
        self.weight_decay = self.cfg.get('weight_decay', 1e-5)
        self.nhid = self.cfg.get('nhid', [16])[0]
        self.tem = self.cfg.get('tem', 0.5)
        self.lr_w = self.cfg.get('lr_w', 0.025)
        self.gamma = self.cfg.get('gamma', 0.25)
        self.device = next(model.parameters()).device
        self.nfeat = self.cfg.get('nfeat')
        self.nclass = self.cfg.get('nclass', 1)

        # f_{cg} in the original paper, vanilla model for projector training
        self.vanilla_model = copy.deepcopy(model)
        # f_{t} in the original paper, synthetic teacher
        self.syn_t = SynTeacher(nfeat=self.nfeat, nhid=self.nhid, nclass=self.nclass)
        
        # loss
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion_cont = ContLoss(tem=self.tem)
        self.weight_compute = None
        self.h_fair = None

        # optimizer
        self.optimizer_van = torch.optim.Adam(self.vanilla_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer_mlp = torch.optim.Adam(self.syn_t.para_mlp, lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer_gnn = torch.optim.Adam(self.syn_t.para_gnn, lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer_proj = torch.optim.Adam(self.syn_t.para_proj, lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self, data, epochs, validation=True):
        self.vanilla_model = self.vanilla_model.to(self.device)
        self.syn_t = self.syn_t.to(self.device)
        
        # vanilla model training
        h_van, _ = self._run_vanilla_training(data, epochs)

        # two fairness experts training
        h_fair_mlp = self._run_expert_mlp_training(data, epochs)
        h_fair_gnn = self._run_expert_gnn_training(data, epochs)

        # projector training
        proj_input = torch.cat((h_fair_mlp, h_fair_gnn), 1)
        self.h_fair = self._run_projector_training(data, epochs, proj_input, h_van)

        tpbar = tqdm(total=epochs, desc=f"Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        best_val_tradeoff = 0
        for epoch in range(epochs):
            loss_train = self.train_step(data)
            
            if validation:
                ret_val = self.evaluate_step(data)
                
                if ret_val["auc_val"]+ret_val["f1_val"]+ret_val["acc_val"]-ret_val["dp_val"]-ret_val["eo_val"] > best_val_tradeoff:
                    best_val_tradeoff = ret_val["auc_val"]+ret_val["f1_val"]+ret_val["acc_val"]-ret_val["dp_val"]-ret_val["eo_val"]
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.weight_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss_train': "{:.2f}".format(loss_train["loss"])})
                tpbar.update(1)

        if tpbar is not None:
            tpbar.close()

    def train_step(self, data) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        
        h_stu, output = self.model.get_embs_and_outs(data.features, data.edge_index)
        
        loss_bce = self.criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        loss_kd = self.criterion_cont(h_stu[data.idx_train], self.h_fair[data.idx_train])
        
        if self.weight_compute is None:
            self.weight_compute = AdaWeight(loss_bce.detach(), loss_kd.detach(), lr=self.lr_w, gamma=self.gamma)
        
        lad1, lad2 = self.weight_compute.compute(loss_bce.item(), loss_kd.item())
        
        loss_total = lad1 * loss_bce + lad2 * loss_kd
        loss_total.backward()
        self.optimizer.step()
        
        return dict(loss=loss_total.item(),
                    loss_bce=loss_bce.item(),
                    loss_kd=loss_kd.item())

    @torch.no_grad()
    def evaluate_step(self, data, is_predict=False):
        """
        shared by self.evaluate and self.predict
        """
        self.model.eval()
        h, output = self.model.get_embs_and_outs(data.features, data.edge_index)
        output = output.detach()
        preds = (output.squeeze() > 0).type_as(data.labels)
        if is_predict:
            return output
        else:
            auc_val = roc_auc_score(data.labels.cpu().numpy()[data.idx_val.cpu()],
                                        output.detach().cpu().numpy()[data.idx_val.cpu()])
            f1_val = f1_score(data.labels[data.idx_val].cpu().numpy(), preds[data.idx_val].cpu().numpy())
            acc_val = accuracy_score(data.labels[data.idx_val].cpu().numpy(), preds[data.idx_val].cpu().numpy())
            parity_val, equality_val = fair_metric(preds[data.idx_val].cpu().numpy(), data.labels[data.idx_val].cpu().numpy(),
                                                    data.sens[data.idx_val].cpu().numpy())
            return dict(auc_val=auc_val,
                        f1_val=f1_val,
                        acc_val=acc_val,
                        dp_val=parity_val,
                        eo_val=equality_val)    

    def _run_vanilla_training(self, data, epochs):
        best_val_loss = 1e5
        for _ in range(epochs):
            self.vanilla_model.train()
            self.optimizer_van.zero_grad()
            _, out = self.vanilla_model.get_embs_and_outs(data.features, data.edge_index)
            loss = self.criterion(out[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
            loss.backward()
            self.optimizer_van.step()
            
            self.vanilla_model.eval()
            with torch.no_grad():
                h, out = self.vanilla_model.get_embs_and_outs(data.features, data.edge_index)
                v_loss = self.criterion(out[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_h = h
                    best_out = out
        return best_h, best_out

    def _run_expert_mlp_training(self, data, epochs):
        for _ in range(epochs):
            self.syn_t.train()
            self.optimizer_mlp.zero_grad()
            h, out = self.syn_t.forward_mlp(data.features)
            loss = self.criterion(out[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
            loss.backward()
            self.optimizer_mlp.step()
            
        self.syn_t.eval()
        with torch.no_grad():
            h_mlp, _ = self.syn_t.forward_mlp(data.features)
        return h_mlp

    def _run_expert_gnn_training(self, data, epochs):
        f_one = torch.ones_like(data.features).to(self.device)
        for _ in range(epochs):
            self.syn_t.train()
            self.optimizer_gnn.zero_grad()
            h, out = self.syn_t.forward_gnn(f_one, data.edge_index)
            loss = self.criterion(out[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
            loss.backward()
            self.optimizer_gnn.step()
            
        self.syn_t.eval()
        with torch.no_grad():
            h_gnn, _ = self.syn_t.forward_gnn(f_one, data.edge_index)
        return h_gnn

    def _run_projector_training(self, data, epochs, proj_in, h_van_label):
        for _ in range(epochs):
            self.syn_t.train()
            self.optimizer_proj.zero_grad()
            h_proj = self.syn_t.projector(proj_in)
            loss = self.criterion_cont(h_proj[data.idx_train], h_van_label[data.idx_train])
            loss.backward()
            self.optimizer_proj.step()
        self.syn_t.eval()
        with torch.no_grad():
            return self.syn_t.projector(proj_in)

# projector
class Projector(nn.Module):
    r"""Multilayer Perceptron based Projector used to merge teacher representations.
    
    Parameters
    ----------
    nfeat : int
        Input feature dimension (concatenated expert embeddings).
    nclass : int
        Output dimension (hidden dimension of the target representation space).
    """
    def __init__(self, nfeat: int, nclass: int):
        super(Projector, self).__init__()
        self.lin1 = nn.Linear(nfeat, nclass)
        self.lin2 = nn.Linear(nclass, nclass)
        self.lin3 = nn.Linear(nclass, nclass)
    
    def forward(self, h, relu=False):
        if relu:
            y = F.relu(self.lin1(h))
            y = F.relu(self.lin2(y))
        else:
            y = self.lin1(h)
            y = self.lin2(y)
        y = self.lin3(y)
        return y

# synthetic teacher
class SynTeacher(nn.Module):
    r"""Synthetic Teacher model that combines fairness-aware experts.
    The teacher consists of an MLP expert (blind to topology) and a GNN expert 
    (blind to features by using neutral inputs).

    Parameters
    ----------
    nfeat : int
        Number of input features.
    nhid : int
        Hidden dimension size for experts and projector.
    nclass : int, optional
        Number of output classes for the temporary classifiers, by default 1.
    dropout : float, optional
        Dropout rate, by default 0.5.
    """
    def __init__(self, nfeat: int, nhid: int, nclass: int=1, dropout: float=0.5):
        super(SynTeacher, self).__init__()
        # mlp fairness expert
        self.expert_mlp = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.Linear(nhid, nhid)
        )
        # gnn fairness expert
        self.expert_gnn = GCNConv(nfeat, nhid)
        
        # projector
        self.projector = Projector(2 * nhid, nhid)
        
        # temp. classifier for two fairness expert
        self.c1 = nn.Linear(nhid, nclass)
        self.c2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

        # optimizer
        self.para_mlp = list(self.expert_mlp.parameters()) + list(self.c1.parameters())
        self.para_gnn = list(self.expert_gnn.parameters()) + list(self.c2.parameters())
        self.para_proj = list(self.projector.parameters())

    def forward_mlp(self, x):
        h = x
        for l, layer in enumerate(self.expert_mlp):
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)
        y = self.c1(h)
        return h, y

    def forward_gnn(self, x_ones, edge_index):
        h = self.expert_gnn(x_ones, edge_index)
        y = self.c2(self.dropout(F.relu(h)))
        return h, y

# contrastive loss
class ContLoss(_Loss):
    r"""Contrastive Loss for Knowledge Distillation.
    Calculates the alignment between student and teacher representations using 
    cosine similarity and a temperature-scaled cross-entropy-like loss (NT-Xent style).

    Parameters
    ----------
    reduction : str, optional
        Reduction method ('mean' or 'sum'), by default 'mean'.
    tem : float, optional
        Temperature scale for similarity, by default 0.5.
    """
    def __init__(self, reduction='mean', tem: float=0.5):
        super(ContLoss, self).__init__()
        self.reduction = reduction
        self.tem: float = tem

    def _sim(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return torch.mm(h1, h2.t())

    def _loss(self, h1: torch.Tensor, h2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tem)
        intra_sim = f(self._sim(h1, h1))
        inter_sim = f(self._sim(h1, h2))
        return -torch.log(inter_sim.diag() / (inter_sim.sum(1) + intra_sim.sum(1) - intra_sim.diag()))

    def forward(self, h1: torch.Tensor, h2: torch.Tensor):
        l1 = self._loss(h1, h2)
        l2 = self._loss(h2, h1)
        ret = (l1 + l2) / 0.5
        ret = ret.mean() if self.reduction=='mean' else ret.sum()
        return ret

# adaptive weight
class AdaWeight:
    r"""Adaptive Weight module for dynamic loss balancing.
    Automatically adjusts the weights between classification loss and distillation loss 
    based on their relative training rates and the provided gamma sensitivity.

    Parameters
    ----------
    loss1_init : torch.Tensor
        Initial value of the classification loss.
    loss2_init : torch.Tensor
        Initial value of the distillation loss.
    weight_loss1 : float, optional
        Initial weight for the first loss, by default 0.5.
    lr : float, optional
        Learning rate for weight updates, by default 0.025.
    gamma : float, optional
        Power factor for relative loss change, by default 0.25.
    """
    def __init__(self, loss1_init, loss2_init, weight_loss1=0.5, lr=0.025, gamma=0.25):
        self.loss1_init = loss1_init
        self.loss2_init = loss2_init
        self.weight_loss1 = weight_loss1
        self.lr = lr
        self.gamma = gamma

    def compute(self, loss1, loss2):
        rela_loss1 = (loss1 / self.loss1_init.item())**self.gamma
        rela_loss2 = (loss2 / self.loss2_init.item())**self.gamma
        rela_weight_loss1 = rela_loss1 / (rela_loss1 + rela_loss2)
        
        self.weight_loss1 = self.lr * rela_weight_loss1 + (1 - self.lr) * self.weight_loss1
        self.weight_loss2 = 1 - self.weight_loss1
        return self.weight_loss1, self.weight_loss2
