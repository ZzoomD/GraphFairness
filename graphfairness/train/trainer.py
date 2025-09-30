import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from graphfairness.evaluation.metrics import *
from graphfairness.utils import BunchDict
from tqdm import tqdm
import os

class Trainer:
    r"""Base trainer class for training graph neural networks, graph transformers, and other models.
    
    This class provides a general framework for model training, evaluation, and prediction,
    supporting core functionalities such as training loops, loss computation, model saving,
    and evaluation metric calculation. It is primarily designed for training and evaluating
    fair graph neural networks.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train, typically a graph neural network (e.g., GCN, SAGE, GIN).
        The input model is built using the ModelBuilder class.
    **cfg : dict
        Configuration parameters for training, including:
        - lr : float, optional
            Learning rate, default is 1e-3
        - weight_decay : float, optional
            Weight decay coefficient, default is 1e-5

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model to train
    cfg : BunchDict
        Training configuration parameters stored as a BunchDict for attribute-style access
    optimizer : torch.optim.Optimizer
        Optimizer, default is Adam
    criterion : torch.nn.Module
        Loss function, default is BCEWithLogitsLoss
    best_loss : float
        Record of the best validation loss for model selection
    weight_path : str
        Path to save the best model weights
    
    Example
    -------
    >>> from graphfairness.train import Trainer
    >>> from graphfairness.models.model_utils import ModelBuilder
    >>> from graphfairness.datasets import FairDataset
    >>> 
    >>> # Load dataset
    >>> dataset = FairDataset(root='./data', name='german')
    >>> data = dataset.data
    >>> 
    >>> # Build model using ModelBuilder
    >>> builder = ModelBuilder()
    >>> model = builder.build(model_name='gcn', nfeat=data.features.shape[1], nclass=1, 
    ...                      nhid=[64], dropout=0.5)
    >>> 
    >>> # Initialize trainer
    >>> trainer = Trainer(model, lr=0.01, weight_decay=5e-4)
    >>> 
    >>> # Train the model
    >>> trainer.train(data, epochs=100, validation=True)
    >>> 
    >>> # Evaluate the model on test set
    >>> metrics = trainer.evaluate(data)
    >>> print(f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, Accuracy: {metrics['acc']:.4f}")
    >>> print(f"Demographic Parity: {metrics['dp']:.4f}, Equal Opportunity: {metrics['eo']:.4f}")
    >>> 
    >>> # Make predictions on test set
    >>> predictions = trainer.predict(data)
    >>> print(f"Predictions shape: {predictions.shape}")
    >>> print(f"First 10 predictions: {predictions[:10]}")
    
    >>> # Load custom weights and evaluate
    >>> metrics_custom = trainer.evaluate(data, weight_path='./custom_weights.pt')
    >>> print(f"Custom weights - AUC: {metrics_custom['auc']:.4f}")
    """
    def __init__(self, model, **cfg):
        super(Trainer, self).__init__()
        self.model = model
        
        self.cfg = BunchDict(cfg)
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.best_loss = 100
        self.weight_path = './weights/best_model.pt'

    def train(self, data, epochs, validation=True):
        """
        Train the model for a specified number of epochs on the given dataset.

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
        tpbar = tqdm(total=epochs, desc=f"Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")

        for epoch in range(epochs):
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

    def train_step(self, data) -> dict:
        """
        One-step training on the inputs.
        
        Parameters
        ----------
        data : DictObject
            Object containing graph data with features, edge_index, labels, idx_train, etc.

        Returns
        -------
        dict
            Dictionary containing the training loss value with format {'loss': float}
        """
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(data.features, data.edge_index)
        loss_train = self.criterion(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        loss_train.backward()
        self.optimizer.step()

        return dict(loss=loss_train.item())
    
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
        self.model.load_state_dict(torch.load(self.weight_path if weight_path is None else weight_path))
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
        Perform forward pass for model evaluation on the validation set,
        shared by both self.evaluate and self.predict methods.

        Parameters
        ----------
        data : DictObject
            Object containing graph data with features, edge_index, labels, idx_val, etc.
        is_predict : bool, optional
            If True, return raw model output; if False, return validation loss, default is False

        Returns
        -------
        torch.Tensor or float
            If is_predict=True, returns raw model output tensor;
            If is_predict=False, returns validation loss value
        """
        self.model.eval()
        output = self.model(data.features, data.edge_index)
        if is_predict:
            return output
        else:
            return self.criterion(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())

    def predict(self, data):
        """
        Generate model predictions.

        Parameters
        ----------
        data : DictObject
            Object containing graph data with features, edge_index, etc.

        Returns
        -------
        torch.Tensor
            Binary classification prediction results tensor with shape [num_nodes]
        """
        output = self.evaluate_step(data, is_predict=True)
        preds = (output.squeeze() > 0).type_as(data.labels)
        return preds

