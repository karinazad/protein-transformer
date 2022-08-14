import math
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class PositionalEncoding(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 max_len: int,
                 ):
        """
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """
        super().__init__()

        position_indices = torch.arange(0, max_len).unsqueeze(1)
        const = torch.exp(torch.arange(0, embed_dim, 2) * - (math.log(10000.0) / embed_dim))

        # Create an empty tensor
        pos_emb = torch.zeros(max_len, embed_dim, requires_grad=False)

        # Get every even/odd position and create overlapping sin/cos waves
        pos_emb[:, 0::2] = torch.sin(position_indices * const) # every even column
        pos_emb[:, 1::2] = torch.cos(position_indices * const) # every odd column
        pos_emb = pos_emb.unsqueeze(0)

        # Add to state_dict
        self.register_buffer("pos_emb", pos_emb)


    def forward(self, x: Tensor) -> Tensor:
        # print(x.size())
        # print(self.pos_emb.size())

        # Sum the original embedding and the positional embedding
        x = x + self.pos_emb[:x.size(0)]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=None, dropout=0.1, num_heads=1, hidden_dim = None, **kwargs):
        super().__init__()

        if not hidden_dim:
            hidden_dim = 4 * embed_dim

        self.attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.ffn =  nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)


    def forward(self, x, mask):
        # Multi-head Attention
        x_res = x.clone().detach()
        x, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.dropout_1(x)
        x += x_res
        x = self.norm_1(x)

        # Linear
        x_res = x.clone().detach()
        x = self.ffn(x)
        x = self.dropout_2(x)
        x += x_res
        x = self.norm_2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(**kwargs) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_weights = layer.attention(x, mask=mask, attn_mask=True)
            attention_maps.append(attn_weights)
            x = layer(x)
        return attention_maps


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    # From:
    # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 100,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)


    def forward(self, x):
        x = x.to(torch.double)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)

        if self.dropout_rate:
            x = self.dropout(x)

        x = self.fc2(x)

        return x