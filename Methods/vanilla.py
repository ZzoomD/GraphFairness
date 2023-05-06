import torch
from Train import *


def train_vanilla(model, optimizer, criterion, data, epochs, model_type, weight_path):
    train_model(model, optimizer, criterion, data, epochs, model_type, weight_path)
