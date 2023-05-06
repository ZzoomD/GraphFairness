import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from .metrics import *
from Train.train_methods import model_forward


def evaluate(model, weight_path, data, model_type='gnn'):
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    h, output = model_forward(model, data, model_type)
    output_preds = (output.squeeze() > 0).type_as(data.labels)
    auc_roc_test = roc_auc_score(data.labels.cpu().numpy()[data.idx_test.cpu()],
                                 output.detach().cpu().numpy()[data.idx_test.cpu()])
    f1_s = f1_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    acc = accuracy_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    parity, equality = fair_metric(output_preds[data.idx_test].cpu().numpy(), data.labels[data.idx_test].cpu().numpy(),
                                   data.sens[data.idx_test].numpy())
    return auc_roc_test, f1_s, acc, parity, equality


def evaluate_mask(model, weight_path, data, model_type='gnn_mask'):
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    h, output = model_forward(model, data, model_type)
    output_preds = (output.squeeze() > 0).type_as(data.labels)
    auc_roc_test = roc_auc_score(data.labels.cpu().numpy()[data.idx_test.cpu()],
                                 output.detach().cpu().numpy()[data.idx_test.cpu()])
    f1_s = f1_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    acc = accuracy_score(data.labels[data.idx_test].cpu().numpy(), output_preds[data.idx_test].cpu().numpy())
    parity, equality = fair_metric(output_preds[data.idx_test].cpu().numpy(), data.labels[data.idx_test].cpu().numpy(),
                                   data.sens[data.idx_test].numpy())
    return auc_roc_test, f1_s, acc, parity, equality