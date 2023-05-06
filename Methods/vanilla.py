import torch
from Train import *


class Vanilla(Trainer):
    def __init__(self, model, optimizer, criterion):
        super().__init__(model, optimizer, criterion)

