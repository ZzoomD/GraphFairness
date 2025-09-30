import torch.nn as nn
import torch
from graphfairness.train import *
from graphfairness.models import *
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
from graphfairness.evaluation.metrics import *
from tqdm import tqdm
import os

class FairGNN(Trainer):
    def __init__(self, model, **cfg):
        super().__init__(model)
        self.model = model
        
        self.cfg = BunchDict(cfg)
        lr = self.cfg.get('lr', 1e-3)
        weight_decay = self.cfg.get('weight_decay', 1e-5)

        self.estimator = GCN(nfeat=self.cfg.nfeat,
                             nhid=self.cfg.nhid,
                             nclass=self.cfg.nclass,
                             dropout=self.cfg.dropout)
        self.adversary = nn.Linear(self.cfg.nhid[-1], 1)

        self.optimizer_e = torch.optim.Adam(self.estimator.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_g = torch.optim.Adam(list(self.model.parameters())+list(self.estimator.parameters()),
                                            lr=lr, weight_decay=weight_decay)
        self.optimizer_a = torch.optim.Adam(self.adversary.parameters(), lr=lr, weight_decay=weight_decay)

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self, data, epochs, validation=True, **train_wargs):
        # parsing parameters
        alpha = train_wargs.get('alpha', 4)
        beta = train_wargs.get('beta', 0.01)
        sens_number = train_wargs.get('sens_number', 200)

        # split data for training the sensitive attribute estimator
        idx_sens_train = self.get_sens_train(data, sens_number)
        idx_sens_train = idx_sens_train.to(data.labels.device)

        self.estimator = self.estimator.to(data.features.device)
        self.adversary = self.adversary.to(data.features.device)
        est_weight_path = self.train_estimator(data, epochs, idx_sens_train)

        best_auc_val, best_acc_val = 0.0, 0.0
        best_fair_val = 100.0
        self.estimator.load_state_dict(torch.load(est_weight_path))
        tpbar = tqdm(total=epochs, desc=f"Training", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
        for epoch in range(epochs):
            loss_train = self.train_step(data, idx_sens_train, alpha, beta)

            if validation:
                ret_val = self.evaluate_step(data)

                # if ret_val["auc_val"] > best_auc_val and ret_val["acc_val"] > best_acc_val:
                #     if ret_val["dp_val"]+ ret_val["eo_val"] < best_fair_val:
                if ret_val["auc_val"] > best_auc_val:
                    best_auc_val = ret_val["auc_val"]
                    best_acc_val = ret_val["acc_val"]
                    best_fair_val = ret_val["dp_val"] + ret_val["eo_val"]
                    os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.weight_path)
            
            if tpbar is not None:
                tpbar.set_postfix({'loss_train': "{:.2f}".format(loss_train["loss"])})
                tpbar.update(1)

        if tpbar is not None:
            tpbar.close()
    
    def train_estimator(self, data, epochs, idx_sens_train):
        best_acc = 0.0
        est_weight_path = f"./weights/fairgnn_estimator_{data.dataset}.pt"
        for epoch in range(epochs):
            self.estimator.train()
            self.optimizer_e.zero_grad()
    
            s_output = self.estimator(data.features, data.edge_index)
            loss_e = F.binary_cross_entropy_with_logits(s_output[idx_sens_train], data.sens[idx_sens_train].unsqueeze(1).float())

            loss_e.backward()
            self.optimizer_e.step()

            if epoch % 10 == 0:
                self.estimator.eval()
                s_output = self.estimator(data.features, data.edge_index)
                s_pred = (s_output.squeeze() > 0).type_as(data.sens)
                acc_val = accuracy_score(data.sens[data.idx_val].cpu(), s_pred[data.idx_val].cpu())
                if acc_val > best_acc:
                    best_acc = acc_val
                    os.makedirs(os.path.dirname(est_weight_path), exist_ok=True)
                    torch.save(self.estimator.state_dict(), est_weight_path)
        return est_weight_path

    def train_step(self, data, idx_sens_train, alpha, beta) -> dict:
        self.model.train()
        self.estimator.train()
        self.adversary.train()

        # update estimator and gnn_classifier (model)
        self.adversary.requires_grad_(False)
        self.optimizer_g.zero_grad()

        s_output = self.estimator(data.features, data.edge_index)
        embs, y_output = self.model.get_embs_and_outs(data.features, data.edge_index)
        s_adv = self.adversary(embs)

        s_pred = torch.sigmoid(s_output.detach())
        s_pred[idx_sens_train] = data.sens[idx_sens_train].unsqueeze(1).float()
        y_pred = torch.sigmoid(y_output)
        
        loss_cls_train = self.criterion(y_output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
        loss_cov_train = torch.abs(torch.mean((s_pred - torch.mean(s_pred)) * (y_pred - torch.mean(y_pred))))
        loss_adv_train = self.criterion(s_adv, s_pred)

        loss_g = loss_cls_train + alpha * loss_cov_train - beta * loss_adv_train
        loss_g.backward()
        self.optimizer_g.step()

        # update adversary
        self.adversary.requires_grad_(True)
        self.optimizer_a.zero_grad()

        s_adv_advtrain = self.adversary(embs.detach())
        loss_adv_advtrain = self.criterion(s_adv_advtrain, s_pred)
        loss_adv_advtrain.backward()
        self.optimizer_a.step()

        return dict(loss=loss_g.item(),
                    loss_cls_train=loss_cls_train.item(),
                    loss_cov_train=loss_cov_train.item(),
                    loss_adv_train=loss_adv_train.item(),
                    loss_adv_advtrain=loss_adv_advtrain.item())
    
    @torch.no_grad()
    def evaluate_step(self, data, is_predict=False):
        """
        shared by self.evaluate and self.predict
        """
        self.model.eval()
        output = self.model(data.features, data.edge_index)
        output = output.detach()
        preds = (output.squeeze() > 0).type_as(data.sens)
        if is_predict:
            return output
        else:
            auc_val = roc_auc_score(data.labels.cpu().numpy()[data.idx_val.cpu()],
                                    output.detach().cpu().numpy()[data.idx_val.cpu()])
            # f1_test = f1_score(data.labels[data.idx_val].cpu().numpy(), preds[data.idx_val].cpu().numpy())
            acc_val = accuracy_score(data.labels[data.idx_val].cpu().numpy(), preds[data.idx_val].cpu().numpy())
            parity_val, equality_val = fair_metric(preds[data.idx_val].cpu().numpy(), data.labels[data.idx_val].cpu().numpy(),
                                                    data.sens[data.idx_val].cpu().numpy())
            return dict(auc_val=auc_val,
                        acc_val=acc_val,
                        dp_val=parity_val,
                        eo_val=equality_val)

    def get_sens_train(self, data, sens_number):
        sens_idx = set(np.where(data.sens.cpu() >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(data.idx_test)))
        idx_sens_train = list(sens_idx - set(data.idx_val) - set(idx_test))
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
        return idx_sens_train
    