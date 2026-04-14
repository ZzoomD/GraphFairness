"""Model utilities module for GraphFairness framework.

This module provides utility functions and classes for building and managing graph neural network models.
It includes a model registry and a builder class to simplify model instantiation and configuration.
"""
import torch.optim as optim
from .gcn import GCN
from .sage import SAGE
from .gin import GIN
from .disgcn import DisGCN
from typing import List
import torch

"""
name2model is a dictionary that maps model names to their corresponding classes.
This allows for dynamic model selection based on configuration parameters.
"""
name2model = {
    'gcn': GCN,
    'graphsage': SAGE,
    'gin': GIN,
    'fairsad': DisGCN
}

class ModelBuilder:
    """Model builder class for GraphFairness framework.
    
    This class provides a unified interface for creating graph neural network models
    based on the specified model name. It handles model instantiation, parameter
    configuration, and device placement.
    
    Parameters
    ----------
    device : torch.device, optional
        The device (CPU or GPU) on which to place the model. If not specified,
        it will automatically use GPU if available, otherwise CPU.
        
    Attributes
    ----------
    device : torch.device
        The device on which models will be placed.
    
    Example
    -------
    >>> from graphfairness.models.model_utils import ModelBuilder
    
    >>> # Create a model builder with automatic device selection
    >>> builder = ModelBuilder()
    >>> # Build a GCN model with specified parameters
    >>> model = builder.build(model_name='gcn', nfeat=128, nclass=2, nhid=[64, 32], dropout=0.5)
    >>> # Build a GraphSAGE model
    >>> model = builder.build(model_name='graphsage', nfeat=100, nclass=1)
    """
    def __init__(self, device=None):
        """Initialize the model builder with the specified device.
        
        Parameters
        ----------
        device : torch.device, optional
            The device (CPU or GPU) on which to place the model. If not specified,
            it will automatically use GPU if available, otherwise CPU.
        """
        super(ModelBuilder, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def build(self, model_name:str, nfeat:int, nclass: int=1, nhid:List[int]=[16], dropout: float=0.5, **model_args) -> torch.nn.Module:
        """Build and configure a graph neural network model based on the specified parameters.
        
        This method creates an instance of the specified model class, configures it with
        the provided parameters, and moves it to the specified device.
        
        Parameters
        ----------
        model_name : str
            Name of the model to build. Must be one of the keys in name2model dictionary.
        nfeat : int
            Number of input features for each node.
        nclass : int, optional
            Number of output classes for prediction, default is 1.
        nhid : List[int], optional
            List of hidden layer dimensions, default is [16].
        dropout : float, optional
            Dropout probability, default is 0.5.
        **model_args : dict
            Additional model-specific parameters (e.g., channels=4 for DisGCN).
        
        Returns
        -------
        torch.nn.Module
            The built and configured graph neural network model, moved to the specified device.
        
        Raises
        ------
        NotImplementedError
            If the specified model_name is not in the name2model dictionary.
        """
        # checking model name
        if model_name not in name2model:
            raise NotImplementedError(f"Invalid model name: {model_name}")
        
        # get model class from name2model dict
        model_class = name2model[model_name]
        model = model_class(
                            nfeat=nfeat,
                            nhid=nhid,
                            nclass=nclass,
                            dropout=dropout,
                            **model_args
                            )
        return model.to(self.device)
