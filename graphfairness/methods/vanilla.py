import torch
from graphfairness.train import *


class Vanilla(Trainer):
    def __init__(self, model, **cfg):
        super().__init__(model, **cfg)
        self.model = model

        self.cfg = BunchDict(cfg)
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.criterion = torch.nn.BCEWithLogitsLoss()

