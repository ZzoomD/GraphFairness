import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from graphfairness.train import Trainer
from graphfairness.utils import BunchDict
from graphfairness.evaluation.metrics import fair_metric


class CFC(Trainer):
    r"""Implementation of `CFC` from the paper entitled `“Compositional Fairness Constraints for Graph Embeddings” <https://arxiv.org/pdf/1905.10674>`.

    CFC employs an adversarial framework to enforce fairness constraints. It consists of a GNN encoder 
    to generate node embeddings, a compositional filter to remove sensitive information, a task classifier, 
    and a discriminator that attempts to predict sensitive attributes from the filtered embeddings.

    Parameters
    ----------
    model : nn.Module
        The GNN backbone model used as an encoder (outputs embeddings, not logits).
    **cfg : dict
        Additional configuration parameters
        - lr : float, optional
            Learning rate for optimization, by default 1e-3
        - weight_decay : float, optional
            Weight decay for regularization, by default 1e-5
        - nhid : int
            Hidden dimension size (embedding size)
        - nclass : int
            Number of output classes for the main task
        - lambda_ : float
            Adversarial regularization strength
        - d_steps : int
            Number of discriminator updates per generator update

    Example
    -------
    .. code-block:: python

        from graphfairness.methods.inprocess.cfc import CFC
        from graphfairness.models import GCN
        
        # load data
        dataset = FairDataset(root='./', name='german')
        n_feat = dataset.data.features.shape[1]
        n_hid = 64

        # Initialize the GNN backbone (as Encoder, output dim = n_hid)
        gnn_encoder = GCN(nfeat=n_feat, nhid=[n_hid], nclass=n_hid, dropout=0.5)
        
        # Create CFC instance
        cfc_model = CFC(gnn_encoder, nhid=n_hid, nclass=1, lambda_=1.0, d_steps=5)
        
        # Train the model
        cfc_model.train(data, epochs=500, validation=True)
        
        # Evaluate the model
        metrics = cfc_model.evaluate(data)
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Demographic Parity: {metrics['dp']:.4f}")

    Note
    ----
    * The 'model' passed to CFC should act as an encoder, meaning its output dimension should match 'nhid'.
    """
    def __init__(self, model, **cfg):
        super().__init__(model)
        self.model = model 
        
        self.cfg = BunchDict(cfg)
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)
        self.embed_dim = self.cfg.nhid if isinstance(self.cfg.nhid, int) else self.cfg.nhid[-1]
        
        # Compositional filter: Transforms embeddings to remove sensitive info
        self.filter = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.embed_dim)
        )

        # Discriminator: Tries to predict sensitive attributes from filtered embeddings
        self.discriminator = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), self.embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(self.embed_dim, 1) # out_dim=1
        )
        self.classifier = nn.Linear(self.embed_dim, self.cfg.nclass) # Main Task Head

        # Optimizers: 
        # - Generator: Updates encoder, filter, and classifier
        # - Discriminator: Updates only the discriminator
        self.optimizer_g = torch.optim.Adam(
            list(self.model.parameters()) + 
            list(self.filter.parameters()) + 
            list(self.classifier.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=lr, weight_decay=weight_decay
        )

        self.criterion_task = nn.BCEWithLogitsLoss()
        self.criterion_disc = nn.BCEWithLogitsLoss()

    def train(self, data, epochs, validation=True, **train_wargs):
        """
        Train the CFC model with adversarial training.
        """
        lambda_ = train_wargs.get('lambda_', self.cfg.get('lambda_', 1.0))
        d_steps = train_wargs.get('d_steps', self.cfg.get('d_steps', 5))

        # Move all components to the same device as data
        device = data.features.device
        self.model.to(device)
        self.filter.to(device)
        self.discriminator.to(device)
        self.classifier.to(device)

        best_auc_val, best_acc_val = 0.0, 0.0

        best_val_score = -float('inf') 

        tpbar = tqdm(total=epochs, desc=f"Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        
        for epoch in range(epochs):
            # Perform one training iteration (generator + discriminator steps)
            loss_train = self.train_step(data, lambda_, d_steps)

            if validation:
                ret_val = self.evaluate_step(data)
                
                val_score = ret_val['auc'] + ret_val['f1'] + ret_val['acc'] - (ret_val['dp'] + ret_val['eo'])

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_auc_val = ret_val['auc'] # for logging/record
                    
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    # Save model checkpoints (encoder, filter, classifier)
                    torch.save({
                        'encoder': self.model.state_dict(),
                        'filter': self.filter.state_dict(),
                        'classifier': self.classifier.state_dict()
                    }, self.weight_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss_g': "{:.4f}".format(loss_train["loss_g"]), 'loss_d': "{:.4f}".format(loss_train["loss_d"])})
                tpbar.update(1)

        if tpbar is not None:
            tpbar.close()
        
        # Load best model weights after training
        if os.path.exists(self.weight_path):
            checkpoint = torch.load(self.weight_path)
            self.model.load_state_dict(checkpoint['encoder'])
            self.filter.load_state_dict(checkpoint['filter'])
            self.classifier.load_state_dict(checkpoint['classifier'])

    def evaluate(self, data, split='test'):
        """
        Evaluate the model on specified data split (train/test)
        """
        if os.path.exists(self.weight_path):
            checkpoint = torch.load(self.weight_path, map_location=data.features.device)
            if isinstance(checkpoint, dict) and 'encoder' in checkpoint:
                self.model.load_state_dict(checkpoint['encoder'])
                self.filter.load_state_dict(checkpoint['filter'])
                self.classifier.load_state_dict(checkpoint['classifier'])
            else:
                self.model.load_state_dict(checkpoint)
        
        # Set indices for evaluation (val indices reused for test/train)
        if split == 'test':
            data.idx_val = data.idx_test
        elif split == 'train':
            data.idx_val = data.idx_train

        return self.evaluate_step(data)
    
    def train_step(self, data, lambda_, d_steps) -> dict:
        """
        Single training step: Alternates between training discriminator and generator.
        """
        # Set all components to training mode
        self.model.train()
        self.filter.train()
        self.classifier.train()
        self.discriminator.train()
        
        edge_index = data.edge_index
        if hasattr(data.edge_index, 'indices'):
            edge_index = torch.stack(data.edge_index.indices()).to(data.features.device)
        
        y = data.labels.float().view(-1, 1)
        s = data.sens.float().view(-1, 1)
        train_mask = data.idx_train

        # --- Phase 1: Train Discriminator ---
        # Train discriminator for d_steps to better distinguish sensitive attributes
        loss_d_item = 0
        for _ in range(d_steps):
            self.optimizer_d.zero_grad()
            
            with torch.no_grad():
                z = self.model(data.features, edge_index)
                z_filtered = self.filter(z)
            
            # Discriminator predicts sensitive attribute from filtered embeddings
            d_logits = self.discriminator(z_filtered[train_mask].detach())
            loss_d = self.criterion_disc(d_logits, s[train_mask])
            
            loss_d.backward()
            self.optimizer_d.step()
            loss_d_item = loss_d.item()

        # --- Phase 2: Train Generator (Encoder + Filter + Classifier) ---
        self.optimizer_g.zero_grad()
        
        # Forward pass: encoder -> filter -> classifier
        z = self.model(data.features, edge_index)
        z_filtered = self.filter(z)
        
        c_logits = self.classifier(z_filtered)
        loss_task = self.criterion_task(c_logits[train_mask], y[train_mask])
        
        # Adversarial loss: Generator tries to fool discriminator
        d_logits_for_g = self.discriminator(z_filtered[train_mask])
        loss_adv_for_d = self.criterion_disc(d_logits_for_g, s[train_mask])
        loss_adv = - loss_adv_for_d

        # Total generator loss: Task performance + adversarial regularization
        loss_g = loss_task + lambda_ * loss_adv
        
        loss_g.backward()
        self.optimizer_g.step()

        return dict(loss_g=loss_g.item(),
                    loss_task=loss_task.item(),
                    loss_adv=loss_adv.item(),
                    loss_d=loss_d_item)

    @torch.no_grad()
    def evaluate_step(self, data, is_predict=False):
        """
        Compute evaluation metrics (AUC, F1, ACC, DP, EO) on validation/test data
        """
        self.model.eval()
        self.filter.eval()
        self.classifier.eval()
        
        edge_index = data.edge_index
        if hasattr(data.edge_index, 'indices'):
            edge_index = torch.stack(data.edge_index.indices()).to(data.features.device)

        # Forward pass: Encoder -> Filter -> Classifier
        z = self.model(data.features, edge_index)
        z_filtered = self.filter(z)
        output = self.classifier(z_filtered) # Logits
        
        preds = (torch.sigmoid(output).squeeze() > 0.5).type_as(data.labels)
        
        if is_predict:
            return output
        else:
            # Compute AUC (handle edge case: single class in labels)
            try:
                if len(np.unique(data.labels[data.idx_val].cpu().numpy())) < 2:
                    auc = 0.5
                else:
                    auc = roc_auc_score(data.labels.cpu().numpy()[data.idx_val.cpu()],
                                            torch.sigmoid(output).detach().cpu().numpy()[data.idx_val.cpu()])
            except:
                auc = 0.5

            f1 = f1_score(data.labels[data.idx_val].cpu().numpy(), preds[data.idx_val].cpu().numpy())
            acc = accuracy_score(data.labels[data.idx_val].cpu().numpy(), preds[data.idx_val].cpu().numpy())
            parity_val, equality_val = fair_metric(preds[data.idx_val].cpu().numpy(), 
                                                   data.labels[data.idx_val].cpu().numpy(),
                                                   data.sens[data.idx_val].cpu().numpy())
            
            return dict(auc=auc,
                        f1=f1,
                        acc=acc,
                        dp=parity_val,
                        eo=equality_val)
