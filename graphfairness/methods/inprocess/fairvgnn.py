from graphfairness.train import Trainer
from torch import nn
from torch.nn import functional as F
import torch
from graphfairness.utils import BunchDict
import math
from graphfairness.models.channel_masker import channel_masker
from sklearn.metrics import accuracy_score, roc_auc_score
from graphfairness.evaluation import *
import os
from tqdm import tqdm

class FairVGNN(Trainer):
    r"""Implementation of `FairVGNN` from the paper.

    FairVGNN is a fairness-aware graph neural network that incorporates variational graph representation learning
    with adversarial training to reduce discrimination against sensitive attributes. It uses a feature mask generator
    to learn fair feature representations and a discriminator to detect sensitive information in node embeddings.

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
        - g_lr : float
            Learning rate for the generator
        - d_lr : float
            Learning rate for the discriminator
        - e_lr : float
            Learning rate for the encoder (GNN layers)
        - c_lr : float
            Learning rate for the classifier (fully connected layers)
        - K : int
            Number of samples for the Gumbel-Softmax distribution
        - d_epochs : int
            Number of epochs to train the discriminator in each iteration
        - c_epochs : int
            Number of epochs to train the classifier in each iteration
        - g_epochs : int
            Number of epochs to train the generator in each iteration
        - ratio : float
            Regularization ratio for the generator loss
        - clip_e : float
            Clipping value for the model weights

    Example
    -------
    .. code-block:: python

        from graphfairness.methods.inprocess.fairvgnn import FairVGNN
        from graphfairness.models import GCN
        from graphfairness.data import FairDataset

        # Load data
        dataset = FairDataset(root='./', name='german')
        data = dataset[0]
        n_feat = data.features.shape[1]
        
        # Initialize the GNN backbone
        gnn_model = GCN(nfeat=n_feat, nhid=[16], nclass=2, dropout=0.5)
        
        # Create FairVGNN instance with configuration
        config = {
            'g_lr': 1e-3,
            'd_lr': 1e-3,
            'e_lr': 1e-3,
            'c_lr': 1e-3,
            'K': 5,
            'd_epochs': 5,
            'c_epochs': 5,
            'g_epochs': 5,
            'ratio': 0.01,
            'clip_e': 0.1
        }
        fair_model = FairVGNN(gnn_model, **config)
        
        # Train the model
        fair_model.train(data, epochs=200, validation=True)
        
        # Evaluate the model
        metrics = fair_model.evaluate(data)
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Demographic Parity: {metrics['dp']:.4f}")
        print(f"Equal Opportunity: {metrics['eo']:.4f}")

    Note
    ----
    * FairVGNN uses a three-player game between a generator, discriminator, and classifier to achieve fairness.
    * The generator learns to mask features to prevent sensitive information from being encoded in node representations.
    * The discriminator tries to predict sensitive attributes from node embeddings, while the generator aims to fool it.
    * The classifier focuses on maintaining high prediction accuracy using the masked features.
    * The training process involves alternating optimization of these three components in each epoch.
    """
    def __init__(self, model, **cfg):
        super().__init__(model)
        self.model = model
        
        self.cfg = BunchDict(cfg)
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)

        self.generator = channel_masker(self.model.conv[0].in_channels)
        self.optimizer_g = torch.optim.Adam([dict(params=self.generator.weights, weight_decay=weight_decay)], lr=self.cfg.g_lr)

        self.discriminator = nn.Linear(self.model.conv[-1].out_channels, 1)
        self.optimizer_d = torch.optim.Adam([dict(params=self.discriminator.parameters(), weight_decay=weight_decay)], lr=self.cfg.d_lr)

        self.optimizer_e = torch.optim.Adam([dict(params=self.model.conv.parameters(), weight_decay=weight_decay)], lr=self.cfg.e_lr)
        self.optimizer_c = torch.optim.Adam([dict(params=self.model.fc.parameters(), weight_decay=weight_decay)], lr=self.cfg.c_lr)

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.generator_path = './weights/best_generator.pt'

        # initialize model parameters
        self.generator.reset_parameters()
        self.discriminator.reset_parameters()

    def train(self, data, epochs, validation=True, **train_wargs):
        best_val_tradeoff = 0
        best_val_loss = math.inf
        
        self.model = self.model.to(data.features.device)
        self.generator = self.generator.to(data.features.device)
        self.discriminator = self.discriminator.to(data.features.device)

        tpbar = tqdm(total=epochs, desc=f"Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        for epoch in range(epochs):
            loss_train = self.train_step(data)

            if validation:
                ret_val = self.evaluate_step(data)

                if ret_val["auc_val"]+ret_val["f1_val"]+ret_val["acc_val"]-ret_val["dp_val"]-ret_val["eo_val"] > best_val_tradeoff:
                    best_val_tradeoff = ret_val["auc_val"]+ret_val["f1_val"]+ret_val["acc_val"]-ret_val["dp_val"]-ret_val["eo_val"]
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.weight_path)
                    torch.save(self.generator.state_dict(), self.generator_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss_train': "{:.2f}".format(loss_train["loss_c"])})
                tpbar.update(1)

        if tpbar is not None:
            tpbar.close()

    def train_step(self, data) -> dict:
        self.generator.eval()
        feature_weights, masks, = self.generator(), []
        for k in range(self.cfg.K):
            mask = F.gumbel_softmax(feature_weights, tau=1, hard=False)[:, 0]
            masks.append(mask)

        # train discriminator to recognize the sensitive group
        self.discriminator.train()
        self.model.conv.train()
        for epoch_d in range(self.cfg.d_epochs):
            self.optimizer_d.zero_grad()
            self.optimizer_e.zero_grad()

            loss_d = 0
            for k in range(self.cfg.K):
                features = data.features * masks[k].detach()
                h, _ = self.model.get_embs_and_outs(features, data.edge_index)
                output = self.discriminator(h)

                loss_d += self.criterion(output[data.idx_train].view(-1), data.sens[data.idx_train])

            loss_d = loss_d / self.cfg.K
            loss_d.backward()
            self.optimizer_d.step()
            self.optimizer_e.step()

        # train classifier
        self.model.train()
        for epoch_c in range(self.cfg.c_epochs):
            self.optimizer_c.zero_grad()
            self.optimizer_e.zero_grad()

            loss_c = 0
            for k in range(self.cfg.K):
                features = data.features * masks[k].detach()
                output = self.model(features, data.edge_index)

                loss_c += F.binary_cross_entropy_with_logits(
                    output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())

            loss_c = loss_c / self.cfg.K
            loss_c.backward()
            self.optimizer_e.step()
            self.optimizer_c.step()

        # train generator to fool discriminator
        self.generator.train()
        self.model.conv.train()
        self.discriminator.eval()
        for epoch_g in range(self.cfg.g_epochs):
            self.optimizer_g.zero_grad()
            self.optimizer_e.zero_grad()

            loss_g = 0
            feature_weights = self.generator()
            for k in range(self.cfg.K):
                mask = F.gumbel_softmax(feature_weights, tau=1, hard=False)[:, 0]

                features = data.features * mask
                h, _ = self.model.get_embs_and_outs(features, data.edge_index)
                output = self.discriminator(h)

                loss_g += F.mse_loss(output[data.idx_train].view(-1),
                                        0.5 * torch.ones_like(output[data.idx_train].view(-1))) + self.cfg.ratio * F.mse_loss(mask.view(-1), torch.ones_like(mask.view(-1)))

            loss_g = loss_g / self.cfg.K
            loss_g.backward()
            self.optimizer_g.step()
            self.optimizer_e.step()

        # weights clip
        weights = torch.stack(masks).mean(dim=0)
        for i in range(self.model.conv[-1].lin.weight.data.shape[1]):
            self.model.conv[-1].lin.weight.data[:, i].data.clamp_(-self.cfg.clip_e * weights[i], self.cfg.clip_e * weights[i])

        return dict(loss_d=loss_d.item(), 
                    loss_c=loss_c.item(), 
                    loss_g=loss_g.item())
    
    def evaluate(self, data, weight_path=None):
        """
        Evaluate model performance on the test set, calculating accuracy, F1 score, AUC, and fairness metrics.

        Parameters
        ----------
        data : DictObject
            Object containing graph data with features, edge_index, labels, sens, idx_test, etc.
        weight_path : str, optional
            Path to model weight file, default is './weights/best_model.pt'

        Returns
        -------
        dict
            Dictionary containing evaluation metrics including:
            - auc : float
                AUC value on the test set
            - f1 : float
                F1 score on the test set
            - acc : float
                Accuracy on the test set
            - dp : float
                Demographic Parity metric
            - eo : float
                Equal Opportunity metric
        """
        weight_path = self.weight_path if weight_path is None else weight_path
        path_list = weight_path.split("/")
        path_list[-1] = "best_generator.pt"
        generator_path = "/".join(path_list)
        self.model.load_state_dict(torch.load(weight_path))
        self.generator.load_state_dict(torch.load(generator_path))
        output = self.evaluate_step(data, is_predict=True)
        preds = (output.squeeze() > 0).type_as(data.labels)
        auc_test = roc_auc_score(data.labels.cpu().numpy()[data.idx_test.cpu()],
                                    output.detach().cpu().numpy()[data.idx_test.cpu()])
        f1_test = f1_score(data.labels[data.idx_test].cpu().numpy(), preds[data.idx_test].cpu().numpy())
        acc = accuracy_score(data.labels[data.idx_test].cpu().numpy(), preds[data.idx_test].cpu().numpy())
        parity, equality = fair_metric(preds[data.idx_test].cpu().numpy(), data.labels[data.idx_test].cpu().numpy(),
                                        data.sens[data.idx_test].cpu().numpy())
        return dict(auc=auc_test, f1=f1_test, acc=acc, dp=parity, eo=equality)


    @torch.no_grad()
    def evaluate_step(self, data, is_predict=False):
        """
        Evaluate model performance on the validation set or generate predictions.
        Used by both evaluate() and predict() methods.

        Parameters
        ----------
        data : DictObject
            Object containing graph data with features, edge_index, labels, sens, idx_val, etc.
        is_predict : bool, optional
            If True, returns raw predictions instead of evaluation metrics, by default False

        Returns
        -------
        dict or torch.Tensor
            If is_predict is True, returns raw predictions tensor
            Otherwise returns a dictionary containing evaluation metrics on the validation set
        """
        self.model.eval()
        self.generator.eval()

        with torch.no_grad():
            outputs = []
            feature_weights = self.generator()
            for k in range(self.cfg.K):
                features = data.features * F.gumbel_softmax(feature_weights, tau=1, hard=True)[:, 0]

                output = self.model(features, data.edge_index)
                outputs.append(output)

            output = torch.stack(outputs).mean(dim=0)
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

