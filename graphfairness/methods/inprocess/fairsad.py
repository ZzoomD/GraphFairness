import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from graphfairness.train import Trainer
from graphfairness.utils import BunchDict
from graphfairness.evaluation.metrics import fair_metric
from torch.nn.modules.loss import _Loss

class DistCor(nn.Module):
    """
    Distance Correlation Loss for macro-disentanglement.
    """
    def forward(self, X1, X2):
        def _distance_matrix(X):
            return torch.cdist(X, X, p=2)

        def _centralize(D):
            row_mean = D.mean(dim=1, keepdim=True)
            col_mean = D.mean(dim=0, keepdim=True)
            grand_mean = D.mean()
            return D - row_mean - col_mean + grand_mean

        A = _centralize(_distance_matrix(X1))
        B = _centralize(_distance_matrix(X2))
        
        dcov2 = (A * B).sum() / (X1.size(0)**2)
        var_a = (A * A).sum() / (X1.size(0)**2)
        var_b = (B * B).sum() / (X1.size(0)**2)
        return dcov2 / (torch.sqrt(var_a * var_b) + 1e-8)

class ChannelMasker(nn.Module):
    """
    Channel masking mechanism to decorrelate sensitive attribute-related components.
    """
    def __init__(self, hid_dim):
        super(ChannelMasker, self).__init__()
        # Initialized with Uniform distribution as per original code
        self.weights = nn.Parameter(torch.distributions.Uniform(0, 1).sample((hid_dim, 2)))

    def forward(self, x):
        mask = F.gumbel_softmax(self.weights, tau=1, hard=False)[:, 0]
        return x * mask

class FeatCov(_Loss):
    def __init__(self):
        super(FeatCov, self).__init__()

    def forward(self, features, sens):
        cov = 0
        for k in range(features.shape[1]):
            cov += torch.abs(torch.mean((sens - torch.mean(sens)) * (features[:, k] - torch.mean(features[:, k]))))
        return cov

class FairSAD(Trainer, nn.Module):
    r"""Implementation of `FairSAD` from the paper entitled `“Fair Graph Representation Learning via 
    Sensitive Attribute Disentanglement” <https://arxiv.org/pdf/2405.07011>`.
    
    FairSAD enhances the fairness of GNNs via Sensitive Attribute Disentanglement (SAD), which separates 
    the sensitive attribute-related information into an independent component to mitigate its impact. 
    It utilizes a channel masking mechanism to adaptively identify the sensitive attribute-related component
    and subsequently decorrelates it.

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
        - channels : int
            Number of disentangled channels, by default 4

    Example
    -------
    .. code-block:: python
        from graphfairness.methods.inprocess.fairsad import FairSAD
        from graphfairness.datasets import FairDataset
        from graphfairness.models import ModelBuilder
        import torch

        # Load data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = FairDataset(root='./', name='german')
        fair_dataset = dataset.data.to(device)
        n_feat = fair_dataset.features.shape[1]
        
        # Build DisGCN model using ModelBuilder
        model_builder = ModelBuilder(device)
        model = model_builder.build(model_name='fairsad',
                                    nfeat=n_feat,
                                    nclass=1,
                                    nhid=[16],
                                    dropout=0.5)
        
        # Create FairSAD instance
        fair_model = FairSAD(model, nfeat=n_feat, nhid=[16], nclass=1, dropout=0.5, channels=4)
        
        # Train the model
        fair_model.train(fair_dataset, epochs=200, validation=True, alpha=0.1, beta=0.1)
        
        # Evaluate the model
        metrics = fair_model.evaluate(fair_dataset)
        print(f"Accuracy: {metrics['acc_val']:.4f}")
        print(f"AUC: {metrics['auc_val']:.4f}")
        print(f"Demographic Parity: {metrics['dp_val']:.4f}")
        print(f"Equal Opportunity: {metrics['eo_val']:.4f}")
    """
    def __init__(self, model, **cfg):
        super().__init__(model)
        self.model = model
        
        # hyperparameter
        self.cfg = BunchDict(cfg)
        self.channels = self.cfg.get('channels', 4)
        self.nclass = self.cfg.get('nclass', 1)
        self.nhid = self.cfg.nhid[0] if isinstance(self.cfg.nhid, list) else self.cfg.nhid
        self.per_channel_dim = self.nhid // self.channels
        self.lr = self.cfg.get('lr', 1e-3)
        self.weight_decay = self.cfg.get('weight_decay', 1e-5)
        self.device = next(model.parameters()).device
        
        # related modules construction
        self.masker = ChannelMasker(self.nhid).to(self.device)
        self.classifier = nn.Linear(self.nhid, self.nclass).to(self.device)
        self.channel_cls = nn.Linear(self.per_channel_dim, self.channels).to(self.device)
        
        # optimizer
        self.optimizer_g = torch.optim.Adam(
            list(self.model.parameters()) + list(self.masker.parameters()) + list(self.classifier.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        self.optimizer_c = torch.optim.Adam(self.channel_cls.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # loss function
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_dc = DistCor()
        self.criterion_chan_cls = nn.CrossEntropyLoss()
        self.criterion_mask = FeatCov()
        
        self.model.init_parameters()
        self.model.init_edge_weight()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def train(self, data, epochs, validation=True, **train_wargs):
        alpha = train_wargs.get('alpha', 0.1)
        beta = train_wargs.get('beta', 0.1)
        tpbar = tqdm(total=epochs, desc="FairSAD Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        
        for _ in range(epochs):
            loss_train = self.train_step(data, alpha, beta)
            
            if validation:
                ret_val = self.evaluate_step(data)
                
                res = ret_val['acc_val'] + ret_val['auc_val'] - ret_val['dp_val'] - ret_val['eo_val']
                if res > getattr(self, 'best_res', 0):
                    self.best_res = res
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.weight_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss': "{:.2f}".format(loss_train['loss'])})
                tpbar.update(1)
                
        if tpbar is not None:
            tpbar.close()

    def train_step(self, data, alpha, beta) -> dict:
        self.model.train()
        self.masker.train()
        self.classifier.train()
        self.channel_cls.train()
        
        self.optimizer_g.zero_grad()
        self.optimizer_c.zero_grad()
        
        h = self.model(data.features, data.edge_index)
        h_masked = self.masker(h)
        output = self.classifier(h_masked)
        
        # downstream task loss
        loss_cls = self.criterion_bce(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        
        # channel identification loss
        loss_chan = 0
        for i in range(self.channels):
            chan_feat = h[:, i*self.per_channel_dim : (i+1)*self.per_channel_dim]
            chan_output = self.channel_cls(chan_feat)
            chan_target = torch.full((chan_output.size(0),), i, dtype=torch.long, device=h.device)
            loss_chan += self.criterion_chan_cls(chan_output, chan_target)
        
        # distance correlation loss
        loss_disen = 0
        for i in range(self.channels):
            for j in range(i + 1, self.channels):
                loss_disen += self.criterion_dc(
                    h[data.idx_train, i*self.per_channel_dim : (i+1)*self.per_channel_dim],
                    h[data.idx_train, j*self.per_channel_dim : (j+1)*self.per_channel_dim]
                )
        
        # masker loss
        s_train = data.sens[data.idx_train].unsqueeze(1)
        h_train = h[data.idx_train]
        loss_mask = self.criterion_mask(h_train, s_train)
        
        total_loss = loss_cls + alpha * (loss_chan + loss_disen) + beta * loss_mask
        total_loss.backward()
        self.optimizer_g.step()
        self.optimizer_c.step()
        
        return dict(loss=total_loss.item())

    @torch.no_grad()
    def evaluate_step(self, data, is_predict=False):
        """
        shared by self.evaluate and self.predict
        """
        self.model.eval()
        self.masker.eval()
        self.classifier.eval()
        h = self.model(data.features, data.edge_index)
        h_masked = self.masker(h)
        output = self.classifier(h_masked)
        if is_predict: 
            return output
        else:
            preds = (output.squeeze() > 0).type_as(data.labels)
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