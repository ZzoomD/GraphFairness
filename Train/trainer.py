import torch


"""a model forward function for GNNs, MLPs, others"""
def model_forward(model, data, model_type):
    if model_type == 'gnn':
        h, output = model(data.features, data.edge_index)
    elif model_type == 'mlp':
        h, output = model(data.features)
    elif model_type == 'gnn_mask':
        h, output = model.forward_eval_mask(data.features, data.edge_index, mask_idx=0)
    return h, output


class Trainer:
    """a trainer to train GNNs or other pre-processing fairness method"""
    def __init__(self, model, optimizer, criterion):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, data, epochs, model_type, weight_path):
        best_loss = 100
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = model_forward(self.model, data, model_type)
            loss_train = self.criterion(output[data.idx_train] if isinstance(output, torch.Tensor) else output[1][data.idx_train],
                                        data.labels[data.idx_train].unsqueeze(1).float())
            loss_train.backward()
            self.optimizer.step()

            self.model.eval()
            output = model_forward(self.model, data, model_type)
            loss_val = self.criterion(output[data.idx_val] if isinstance(output, torch.Tensor) else output[1][data.idx_val],
                                      data.labels[data.idx_val].unsqueeze(1).float())

            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                torch.save(self.model.state_dict(), weight_path)

