import torch


def train_model(model, optimizer, criterion, data, epochs, model_type, weight_path):
    best_loss = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model_forward(model, data, model_type)
        loss_train = criterion(output[data.idx_train] if isinstance(output, torch.Tensor) else output[1][data.idx_train],
                               data.labels[data.idx_train].unsqueeze(1).float())
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model_forward(model, data, model_type)
        loss_val = criterion(output[data.idx_val] if isinstance(output, torch.Tensor) else output[1][data.idx_val],
                             data.labels[data.idx_val].unsqueeze(1).float())

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            torch.save(model.state_dict(), weight_path)


def model_forward(model, data, model_type):
    if model_type == 'gnn':
        h, output = model(data.features, data.edge_index)
    elif model_type == 'mlp':
        h, output = model(data.features)
    elif model_type == 'gnn_mask':
        h, output = model.forward_eval_mask(data.features, data.edge_index, mask_idx=0)
    return h, output