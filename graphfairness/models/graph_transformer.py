"""
Graph Transformer Components for FairGT
Contains basic Transformer building blocks
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def init_params(module, nlayer):
    """Initialize model parameters"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(nlayer))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network with GELU activation"""
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer"""
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class GraphTransformer(nn.Module):
    """
    Basic Graph Transformer Backbone
    
    Args:
        nfeat: Number of input features
        nhid: List of hidden dimensions or single int
        nclass: Number of output classes
        nhead: Number of attention heads
        nlayer: Number of transformer layers
        dropout: Dropout rate
    """
    def __init__(self, nfeat: int, nhid: List[int] = [64], nclass: int = 1,
                 nhead: int = 2, nlayer: int = 1, dropout: float = 0.3):
        super().__init__()

        # Handle nhid as list or int
        if isinstance(nhid, int):
            nhid = [nhid]
        
        self.input_dim = nfeat
        self.hidden_dim = nhid[-1] if nhid else 64
        self.ffn_dim = 2 * self.hidden_dim
        self.num_heads = nhead
        self.nlayer = nlayer
        self.n_class = nclass
        self.dropout_rate = dropout
        self.attention_dropout_rate = dropout

        # Initial embedding layer
        self.att_embeddings = nn.Linear(self.input_dim, self.hidden_dim)

        # Transformer encoder layers
        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, 
                                self.attention_dropout_rate, self.num_heads)
                    for _ in range(self.nlayer)]
        self.layers = nn.ModuleList(encoders)
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(self.hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.fc = nn.Linear(int(self.hidden_dim / 2), self.n_class)

        # Initialize parameters
        self.apply(lambda module: init_params(module, nlayer=self.nlayer))
        
        # Initialize weights
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index=None, edge_weight=None):
        """
        Forward pass - basic Transformer encoding
        
        Args:
            x: [N, nfeat] or [N, seq_len, nfeat] Node features
            edge_index: Edge indices (for compatibility, not used)
            edge_weight: Edge weights (for compatibility, not used)
            
        Returns:
            output: [N, nclass] Classification logits
        """
        # Handle both 2D and 3D input
        if x.dim() == 2:
            # [N, nfeat] -> [N, 1, nfeat] for single node features
            h = x.unsqueeze(1)
        else:
            # [N, seq_len, nfeat] for sequence input
            h = x
        
        # Initial embedding
        h = self.att_embeddings(h)  # [N, seq_len, hidden_dim]
        
        # Pass through transformer layers
        for enc_layer in self.layers:
            h = enc_layer(h)
        
        # Final layer norm
        h = self.final_ln(h)  # [N, seq_len, hidden_dim]
        
        # Global pooling: mean pooling over sequence dimension
        h = h.mean(dim=1)  # [N, hidden_dim]
        
        # Output projection
        h = torch.relu(self.out_proj(h))
        output = self.fc(h)
        
        return output
    
    def get_embs_and_outs(self, x, edge_index=None, edge_weight=None):
        """
        Get embeddings and outputs (for compatibility with GraphFairness)
        
        Returns:
            embs: [N, hidden_dim] Node embeddings
            outs: [N, nclass] Classification outputs
        """
        # Handle both 2D and 3D input
        if x.dim() == 2:
            h = x.unsqueeze(1)
        else:
            h = x
        
        # Initial embedding
        h = self.att_embeddings(h)
        
        # Pass through transformer layers
        for enc_layer in self.layers:
            h = enc_layer(h)
        
        # Final layer norm and pooling
        h = self.final_ln(h)
        embs = h.mean(dim=1)  # [N, hidden_dim]
        
        # Output projection
        h_out = torch.relu(self.out_proj(embs))
        outs = self.fc(h_out)
        
        return embs, outs
