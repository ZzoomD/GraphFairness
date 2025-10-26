import torch.nn as nn
import torch
from graphfairness.train import *
from graphfairness.models import *
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
from graphfairness.evaluation.metrics import *
from graphfairness.utils import BunchDict
from tqdm import tqdm
import os
from torch_sparse import SparseTensor

class FairDrop(Trainer):
    r"""Implementation of `FairDrop` from the paper entitled `“FairDrop: Biased Edge Dropout for Enhancing 
    Fairness in Graph Representation Learning” <https://arxiv.org/pdf/2104.14210>`.

    FairDrop is a method for enhancing fairness in graph representation learning through biased edge dropout. 
    Specifically, FairDrop drops edges between nodes that belong to the same sensitive attribute group, 
    thereby reducing interactions among nodes within that group. 

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

    Example
    -------
    .. code-block:: python

        from graphfairness.methods.preprocess.fairdrop import FairDrop
        from graphfairness.models import GCN
        import torch_geometric as pyg

        # load data
        dataset = FairDataset(root='./', name='german')
        n_feat = dataset.data.features.shape[1]

        # Initialize the GNN backbone
        gnn_model = GCN(nfeat=n_feat, nhid=[16], nclass=2, dropout=0.5)
        
        # Create FairDrop instance
        fair_model = FairDrop(gnn_model)
        
        # Train the model
        fair_model.train(data, epochs=200, validation=True, delta=0.25)
        
        # Evaluate the model
        metrics = fair_model.evaluate(data)
        print(f"Accuracy: {metrics['acc_val']:.4f}")
        print(f"AUC: {metrics['auc_val']:.4f}")
        print(f"Demographic Parity: {metrics['dp_val']:.4f}")
        print(f"Equal Opportunity: {metrics['eo_val']:.4f}")
    """
    def __init__(self, model, **cfg):
        super().__init__(model, **cfg)
        self.model = model

        self.cfg = BunchDict(cfg)
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def train(self, data, epochs, validation=True, **train_wargs):
        """
        Train the model for a specified number of epochs on the given dataset with fair edge masks.

        Parameters
        ----------
        data : DictObject
            Object containing graph data with features, edge_index, labels, idx_train, etc.
        epochs : int
            Number of training epochs
        validation : bool, optional
            Whether to perform validation during training and save the best model, default is True

        Returns
        -------
        None
            Updates model parameters during training and saves the best model
        """
        # delta range from 0 to 0.5
        delta = train_wargs.get('delta', 0.25)
        tpbar = tqdm(total=epochs, desc=f"Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")

        for epoch in range(epochs):
            if epoch == 0 or (epoch+1) % 10 == 0:
                if epoch == 0:
                    edge_index_ori = data.edge_index.detach().clone()
                fair_mask = self.get_dropedge_mask(edge_index_ori, data.sens, delta)
                data.edge_index = SparseTensor.from_edge_index(torch.stack([edge_index_ori.storage.row()[fair_mask], edge_index_ori.storage.col()[fair_mask]], dim=0), 
                                                                sparse_sizes=(data.features.shape[0], data.features.shape[0]), )
            loss_train = self.train_step(data)

            if validation:
                loss_val = self.evaluate_step(data)
            
                if loss_val.item() < self.best_loss:
                    self.best_loss = loss_val.item()
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.weight_path)

            if tpbar is not None:
                tpbar.set_postfix({'loss_train': "{:.2f}".format(loss_train['loss'])})
                tpbar.update(1)

        if tpbar is not None:
            tpbar.close()

    def get_dropedge_mask(self, edge_index, sens, delta):
        """
        Generate a fair mask for dropping edges based on the sensitive attribute.

        Parameters
        ----------
        edge_index : torch.SparseTensor
            Tensor of shape (2, E) representing the edge indices in the graph
        sens : torch.FloatTensor
            Tensor of shape (N,) representing the sensitive attribute for each node
        delta : float
            The probability for randomized response

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (E,) indicating which edges to keep (True) and which to drop (False)
        """
        row, col, _ = edge_index.coo()
        # get the sensitive attribute for each edge
        sens_edge = sens[row] != sens[col]
        rand_mask = (torch.FloatTensor(1, sens_edge.shape[0]).uniform_() < 0.5+delta).to(sens_edge.device)
        # generate the fair mask for dropping edges
        fair_mask = torch.where(rand_mask, sens_edge, ~sens_edge)
        return fair_mask.squeeze(0)

    
    