import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None): # (B, H, L_q, d_k)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn # (B, H, L_q, d_v)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q = q.size(0) 
        N_q = q.size(1) 
        B_k = k.size(0)
        N_k = k.size(1) 
        B_v = v.size(0) 
        N_v = v.size(1) 

        residual = q

        q = self.w_qs(q).view(B_q, N_q, n_head, d_q) 
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask) 

        q = q.transpose(1, 2).contiguous().view(B_q, N_q, -1) 
        q = self.fc(q) 
        q = q + residual

        return q, attn


class FFN1(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, channels, dropout=0.1):
        super(FFN1, self).__init__()
        self.lin1 = nn.Linear(channels, channels)  # position-wise
        self.lin2 = nn.Linear(channels, channels)  # position-wise
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x) + residual

        return x


class FFN2(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(FFN2, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        return x


class ICALayer(nn.Module):
    def __init__(self, n_head, channels, dropout=0.1):
        super(ICALayer, self).__init__()
        self.node_norm = nn.LayerNorm(channels)
        self.node_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.patch_norm = nn.LayerNorm(channels)
        self.patch_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.node_ffn = FFN1(channels, dropout)
        self.patch_ffn = FFN1(channels, dropout)
        self.fuse_lin = nn.Linear(2 * channels, channels)


    def forward(self, x, patch, attn_mask=None): 
        x = self.node_norm(x)
        patch_x = x[patch] 
        patch_x, attn = self.node_transformer(patch_x, patch_x, patch_x, attn_mask)
        patch_x = self.node_ffn(patch_x)

        x[patch] = patch_x 

        return x


class ICABlock(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, dropout1=0.5, dropout2=0.1):
        super(ICABlock, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = FFN2(in_channels, hidden_channels)
        self.ICALayers = nn.ModuleList()
        for _ in range(0, layers):
            self.ICALayers.append(
                ICALayer(n_head, hidden_channels, dropout=dropout2))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, patch):
        patch_mask = (patch != self.num_nodes - 1).float().unsqueeze(-1) 
        attn_mask = torch.matmul(patch_mask, patch_mask.transpose(1, 2)).int() 

        x = self.attribute_encoder(x) 
        for i in range(0, self.layers):
            x = self.ICALayers[i](x, patch, attn_mask) 
        x = self.dropout(x)
        x = self.classifier(x) 
        return x


class FairGP(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 activation=F.relu, layers: int=2, n_head: int=4, dropout1=0.5, dropout2=0.1):
        super(FairGP, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.activation = activation
        self.dropout = nn.Dropout(dropout1)
        
        self.ica = ICABlock(num_nodes, in_channels, hidden_channels, out_channels, layers, n_head, dropout1, dropout2)


    def forward(self, x, patch, edge_index=None):
        z = self.ica(x, patch)
        return z
    
    def get_embs_and_outs(self, x, patch, edge_index=None):
        # Implementation for FairGP similar to get_embs_and_outs in other models
        # But FairGP forward returns logits directly.
        # We need to hack a bit or modify ICABlock to return embeddings.
        
        # Access internal structure
        patch_mask = (patch != self.num_nodes - 1).float().unsqueeze(-1) 
        attn_mask = torch.matmul(patch_mask, patch_mask.transpose(1, 2)).int() 

        x = self.ica.attribute_encoder(x) 
        for i in range(0, self.ica.layers):
            x = self.ica.ICALayers[i](x, patch, attn_mask) 
        
        embs = x # features before classifier
        
        x = self.ica.dropout(x)
        outs = self.ica.classifier(x) 
        
        return embs, outs
