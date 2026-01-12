import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from ...utils.utils import DistCor
from ...evaluation.metrics import fair_metric


class FairDLA:
    """
    FairDLA (Fairness-aware Disentangled Learning Algorithm) method implementation
    """
    
    def __init__(self, nfeat, nhid, nclass, channels=2, dropout=0.5, lr=0.001, weight_decay=1e-5):
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.channels = channels
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Model components
        self.encoder = DisGCN(nfeat=nfeat, nhid=nhid, nclass=nclass, chan_num=channels, 
                              layer_num=2, dropout=dropout)
        self.classifier = nn.Linear(nhid, nclass)
        
        self.per_channel_dim = nhid // channels
        self.channel_cls = nn.Linear(self.per_channel_dim, channels)
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=lr, weight_decay=weight_decay)
            
        self.optimizer_c = torch.optim.Adam(
            list(self.channel_cls.parameters()),
            lr=lr, weight_decay=weight_decay)
        
        # Loss functions
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_dc = DistCor()
        self.criterion_mul_cls = nn.CrossEntropyLoss()
        
        # Initialize parameters
        self.encoder.init_parameters()
        self.encoder.init_edge_weight()
        
        # Training parameters
        self.perturb_epsilon = 0.3
        self.random_attack_num_samples = 10
        self.adv = 0
        self.avgy = False

    def train_fit(self, data, epochs, alpha=0.25, device='cpu'):
        """
        Train the FairDLA model
        
        Parameters:
        data: dataset with features, edge_index, labels, idx_train, idx_val, idx_test, sens
        epochs: number of training epochs
        alpha: weight coefficient for disentanglement
        device: computing device
        """
        best_res_val = float('-inf')
        roc_test, f1_test, acc_test, parity_test, equality_test = 0, 0, 0, 0, 0
        
        # Move data to device
        data.features = data.features.to(device)
        data.edge_index = data.edge_index.to(device)
        data.labels = data.labels.to(device)
        data.sens = data.sens.to(device)
        
        # Compute sensitive attribute vector
        with torch.no_grad():
            pre_emb = self.encoder(data.features, data.edge_index)[data.idx_train]
            self.sens_train = data.sens[data.idx_train]
            self.sens_avg = compute_attribute_vectors_avg_diff(pre_emb, self.sens_train)
        
        for epoch in range(epochs):
            self.encoder.train()
            self.classifier.train()
            self.channel_cls.train()
            
            self.optimizer_g.zero_grad()
            self.optimizer_c.zero_grad()
            
            # Forward pass
            h = self.encoder(data.features, data.edge_index)
            
            if self.adv == 0:
                if self.avgy:
                    noisy_embeds, y_repeated = self.augment_data_y(
                        h[data.idx_train], data.labels[data.idx_train], self.sens_avg)
                else:
                    noisy_embeds, y_repeated = self.augment_data(
                        h[data.idx_train], data.labels[data.idx_train], self.sens_avg)
                
                train_emb = torch.cat([h[data.idx_train], noisy_embeds])
                y_targets = torch.cat([data.labels[data.idx_train], y_repeated])
                output = self.classifier(train_emb)
                loss_cls_train = self.criterion_bce(output, y_targets.unsqueeze(1).float())
            else:
                # Adversarial training
                if self.avgy:
                    train_emb_adv = self.get_adv_examples_y(
                        h[data.idx_train], data.labels[data.idx_train], self.sens_avg)
                else:
                    train_emb_adv = self.get_adv_examples(h[data.idx_train], self.sens_avg)
                
                z_embed_adv = self.classifier(train_emb_adv)
                z_embed = self.classifier(h[data.idx_train])
                loss_ood = torch.linalg.norm(z_embed - z_embed_adv, ord=2, dim=1).mean()
                loss_cls_train = self.criterion_bce(
                    z_embed, data.labels[data.idx_train].unsqueeze(1).float()) + 0.7 * loss_ood
            
            # Channel identification loss
            loss_chan_train = 0
            for i in range(self.channels):
                chan_output = self.channel_cls(h[:, i*self.per_channel_dim:(i+1)*self.per_channel_dim])
                chan_tar = torch.ones(chan_output.shape[0], dtype=int) * i
                chan_tar = chan_tar.to(device)
                loss_chan_train += self.criterion_mul_cls(chan_output, chan_tar)
            
            # Distance correlation loss
            loss_disen_train = 0
            len_per_channel = int(h.shape[1] / self.channels)
            for i in range(self.channels):
                for j in range(i + 1, self.channels):
                    loss_disen_train += self.criterion_dc(
                        h[data.idx_train, i * len_per_channel:(i + 1) * len_per_channel],
                        h[data.idx_train, j * len_per_channel:(j + 1) * len_per_channel])
            
            # Total loss
            if self.adv == 0:
                loss_train = loss_cls_train + alpha * (loss_chan_train + loss_disen_train)
            else:
                loss_train = self.adv * loss_cls_train + alpha * (loss_chan_train + loss_disen_train)
            
            # Backward pass
            loss_train.backward()
            self.optimizer_g.step()
            self.optimizer_c.step()
            
            # Evaluation
            if epoch % 10 == 0:
                self.encoder.eval()
                self.classifier.eval()
                
                with torch.no_grad():
                    h = self.encoder(data.features, data.edge_index)
                    y_output_val = self.classifier(h)
                    y_output_val = y_output_val.detach()
                    y_pred_val = (y_output_val.squeeze() > 0).type_as(data.sens)
                    
                    acc_val = accuracy_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                    roc_val = roc_auc_score(data.labels[data.idx_val].cpu(), y_output_val[data.idx_val].cpu())
                    f1_val = f1_score(data.labels[data.idx_val].cpu(), y_pred_val[data.idx_val].cpu())
                    parity, equality = fair_metric(
                        y_pred_val[data.idx_val].cpu().numpy(),
                        data.labels[data.idx_val].cpu().numpy(),
                        data.sens[data.idx_val].cpu().numpy())
                    
                    res_val = acc_val + roc_val - parity - equality
                    
                    if res_val > best_res_val:
                        best_res_val = res_val
                        acc_test = accuracy_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                        roc_test = roc_auc_score(data.labels[data.idx_test].cpu(), y_output_val[data.idx_test].cpu())
                        f1_test = f1_score(data.labels[data.idx_test].cpu(), y_pred_val[data.idx_test].cpu())
                        parity_test, equality_test = fair_metric(
                            y_pred_val[data.idx_test].cpu().numpy(),
                            data.labels[data.idx_test].cpu().numpy(),
                            data.sens[data.idx_test].cpu().numpy())
        
        # Return final test results
        return roc_test, f1_test, acc_test, parity_test, equality_test
    
    def augment_data(self, embed, y, sens_attr_vector):
        """
        Augment data with sensitive attribute perturbations
        """
        assert y.dim() == 1 and sens_attr_vector is not None
        y_repeated = y.repeat_interleave(self.random_attack_num_samples)
        assert embed.dim() == 2 and embed.size(0) == y.size(0)
        
        noisy_latents = embed.repeat_interleave(self.random_attack_num_samples, dim=0).clone().detach()
        coeffs = (2 * torch.rand(noisy_latents.shape[0], 1, device=noisy_latents.device) - 1) * self.perturb_epsilon
        noisy_latents += sens_attr_vector * coeffs
        
        return noisy_latents, y_repeated
    
    def augment_data_y(self, embed, y, sens_attr_vectors):
        """
        Augment data with sensitive attribute perturbations based on label values
        """
        assert y.dim() == 1 and sens_attr_vectors is not None

        y_repeated = y.repeat_interleave(self.random_attack_num_samples)
        assert embed.dim() == 2 and embed.size(0) == y.size(0)

        noisy_latents = embed.repeat_interleave(self.random_attack_num_samples, dim=0).clone().detach()
        coeffs = (2 * torch.rand(noisy_latents.shape[0], 1, device=noisy_latents.device) - 1) * self.perturb_epsilon

        sens_attr_vector_tensor = torch.zeros_like(noisy_latents)
        for label in sens_attr_vectors.keys():
            mask = (y_repeated == label)
            sens_attr_vector_tensor[mask] = sens_attr_vectors[label]

        noisy_latents += sens_attr_vector_tensor * coeffs

        return noisy_latents, y_repeated
    
    def get_adv_examples(self, embed, attr_vectors_diff):
        """
        Generate adversarial examples
        """
        noisy_emb_all = []
        losses_all = []
        
        for _ in range(self.random_attack_num_samples):
            noisy_emb = embed.clone()
            sens_attr_vector_repeated = torch.repeat_interleave(
                attr_vectors_diff.unsqueeze(0), embed.shape[0], dim=0)
            coeffs = (2 * torch.rand(embed.shape[0], 1, device=embed.device) - 1) * self.perturb_epsilon
            noisy_emb += sens_attr_vector_repeated * coeffs
            noisy_emb_all.append(noisy_emb)
            loss = self.calc_loss(embed, noisy_emb)
            losses_all.append(loss.clone().detach())
        
        losses_all = torch.stack(losses_all, dim=1)
        _, idx = torch.max(losses_all, dim=1)
        adv_examples = []
        
        for i, sample_idx in enumerate(idx.cpu().tolist()):
            adv_examples.append(noisy_emb_all[sample_idx][i])
        
        return torch.stack(adv_examples, 0)
    
    def get_adv_examples_y(self, embed, labels, attr_vectors_diff):
        """
        Generate adversarial examples based on label values
        """
        noisy_emb_all = []
        losses_all = []
        for _ in range(self.random_attack_num_samples):
            noisy_emb = embed.clone()
            
            # 为每个样本选择对应标签的敏感属性向量差异
            sens_attr_vector_repeated = torch.stack([attr_vectors_diff[label.item()] for label in labels], dim=0)
            
            coeffs = (2 * torch.rand(embed.shape[0], 1, device=embed.device) - 1) * self.perturb_epsilon
            noisy_emb += sens_attr_vector_repeated * coeffs
            noisy_emb_all.append(noisy_emb)
            loss = self.calc_loss(embed, noisy_emb)
            losses_all.append(loss.clone().detach())
            
        losses_all = torch.stack(losses_all, dim=1)
        _, idx = torch.max(losses_all, dim=1)
        adv_examples = []
        for i, sample_idx in enumerate(idx.cpu().tolist()):
            adv_examples.append(noisy_emb_all[sample_idx][i])
        return torch.stack(adv_examples, 0)
    
    def calc_loss(self, embed, embed_adv):
        """
        Calculate loss between original and adversarial embeddings
        """
        z_embed_adv = self.classifier(embed_adv)
        z_embed = self.classifier(embed)
        l_2 = torch.linalg.norm(z_embed - z_embed_adv, ord=2, dim=1)
        return l_2


class DisenLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, channels, reduce=True):
        super(DisenLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.per_channel_dim = self.out_dim // self.channels
        self.reduce = reduce

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(channels):
            if reduce:
                self.lin_layers.append(nn.Linear(in_features=in_dim, out_features=self.per_channel_dim))
                self.conv_layers.append(Linear(in_channels=self.per_channel_dim, out_features=self.per_channel_dim,
                                              bias=False, weight_initializer='glorot'))
            else:
                self.conv_layers.append(Linear(in_channels=self.in_dim, out_features=self.per_channel_dim,
                                              bias=False, weight_initializer='glorot'))
        
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.per_channel_dim), dtype=torch.float), requires_grad=True)
            for i in range(self.channels))

    def get_reddim_k(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feat = self.lin_layers[k](x)
            z_feats.append(z_feat)
        return z_feats

    def get_k_feature(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feats.append(x)
        return z_feats

    def forward(self, x, edge_index, edge_weight):
        assert self.channels == edge_weight.shape[1], "axis dimension in direction 1 need to be equal to channels number"
        
        if self.reduce:
            z_feats = self.get_reddim_k(x)
        else:
            z_feats = self.get_k_feature(x)
        
        c_feats = []
        for k, layer in enumerate(self.conv_layers):
            c_temp = layer(z_feats[k])
            edge_index_copy = edge_index.clone()
            if not edge_index_copy.has_value():
                edge_index_copy = edge_index_copy.fill_value(1., dtype=None)
            edge_index_copy.storage.set_value_(edge_index_copy.storage.value() * edge_weight[:, k])
            out = self.propagate(edge_index_copy, x=c_temp)
            if self.bias_list is not None:
                out = out + self.bias_list[k]
            c_feats.append(F.normalize(out, p=2, dim=1))
        
        output = torch.cat(c_feats, dim=1)
        return output

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class DisGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, chan_num, layer_num, dropout=0.5):
        super(DisGCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout_rate = dropout
        self.chan_num = chan_num
        self.layer_num = layer_num
        self.edge_weight = None

        self.assigner = NeiborAssigner(nfeat, chan_num)
        self.disenlayers = nn.ModuleList()
        
        for i in range(layer_num-1):
            if i == 0:
                self.disenlayers.append(DisenLayer(nfeat, nhid, chan_num))
            else:
                self.disenlayers.append(DisenLayer(nhid, nhid, chan_num))
        
        self.dropout = nn.Dropout(dropout)
        self.init_parameters()

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.normal_(item, mean=0, std=1)
    
    def init_edge_weight(self):
        for m in self.assigner.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        assert isinstance(edge_index, SparseTensor), "Expected input is sparse tensor"
        feats_pair = torch.cat([x[edge_index.storage._col, :], x[edge_index.storage._row, :]], dim=1)
        edge_weight = self.assigner(feats_pair.detach())
        
        for layer in self.disenlayers:
            x = layer(x, edge_index, edge_weight)
            x = self.dropout(x)
        
        return x


class NeiborAssigner(nn.Module):
    def __init__(self, nfeats, channels):
        super(NeiborAssigner, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * nfeats, out_features=channels),
            nn.Linear(in_features=channels, out_features=channels)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features_pair):
        alpha_score = self.layers(features_pair)
        alpha_score = torch.softmax(alpha_score, dim=1)
        return alpha_score


def compute_attribute_vectors_avg_diff(embeddings, sens):
    """
    Compute average difference of attribute vectors
    """
    sens = sens.long()
    num_groups = sens.max() + 1
    group_embeddings = []
    
    for g in range(num_groups):
        group_mask = (sens == g)
        if group_mask.sum() > 0:
            group_embedding = embeddings[group_mask].mean(dim=0)
            group_embeddings.append(group_embedding)
        else:
            group_embeddings.append(torch.zeros_like(embeddings[0]))
    
    if len(group_embeddings) >= 2:
        return group_embeddings[0] - group_embeddings[1]
    else:
        return torch.zeros_like(embeddings[0])


def compute_attribute_vectors_avg_diff_y(embeddings, sens, labels):
    """
    Compute average difference of attribute vectors based on label values
    """
    sens_attr_vectors = {}
    unique_labels = torch.unique(labels)
    
    for label in unique_labels:
        label_mask = (labels == label)
        label_sens = sens[label_mask]
        label_embeddings = embeddings[label_mask]
        
        label_sens = label_sens.long()
        num_groups = label_sens.max() + 1
        group_embeddings = []
        
        for g in range(num_groups):
            group_mask = (label_sens == g)
            if group_mask.sum() > 0:
                group_embedding = label_embeddings[group_mask].mean(dim=0)
                group_embeddings.append(group_embedding)
            else:
                group_embeddings.append(torch.zeros_like(label_embeddings[0]))
        
        if len(group_embeddings) >= 2:
            sens_attr_vectors[label.item()] = group_embeddings[0] - group_embeddings[1]
        else:
            sens_attr_vectors[label.item()] = torch.zeros_like(label_embeddings[0])
    
    return sens_attr_vectors