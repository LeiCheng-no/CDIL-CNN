import torch
import torch.nn as nn
from performer_pytorch import Performer
from linformer import Linformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from time_utils import PositionalEncoding


class TransformerHead(nn.Module):
    def __init__(self,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     ):
        super(TransformerHead, self).__init__()
        self.linear = nn.Linear(1, dim, bias=True)
        self.posenc = PositionalEncoding(dim, n_vec)
        encoder_layers = TransformerEncoderLayer(dim, heads, dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, depth)
        self.final = nn.Linear(dim, n_class)

    def forward(self, x):
        x = self.linear(x)
        x = self.posenc(x)
        # print(x.shape)
        # x = x.permute(1, 0, 2)
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = self.final(torch.mean(x, dim=1))
        # print(x.shape)
        return x


class PerformerHead(nn.Module):
    def __init__(self,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     ):
        super(PerformerHead, self).__init__()
        self.linear = nn.Linear(1, dim, bias=True)
        self.posenc = PositionalEncoding(dim, n_vec)
        self.performer = Performer(
            dim = dim,
            depth=depth,
            heads = heads,
            dim_head=dim,
            causal = True
        )
        self.final = nn.Linear(dim, n_class)

    def forward(self, x):
        x = self.linear(x)
        x = self.posenc(x)
        x = self.performer(x)
        x = self.final(torch.mean(x, dim=1))
        return x


class LinformerHead(nn.Module):
    def __init__(self,
     dim,
     heads,
     depth,
     n_vec,
     n_class,
     ):
        super(LinformerHead, self).__init__()
        self.linear = nn.Linear(1, dim, bias=True)
        self.posenc = PositionalEncoding(dim, n_vec)
        self.linformer = Linformer(
            dim=dim,
            seq_len=n_vec,
            depth=depth,
            heads=heads,
            k=dim,
            one_kv_head=True,
            share_kv=True
        )
        self.final = nn.Linear(dim, n_class)

    def forward(self, x):
        x = self.linear(x)
        x = self.posenc(x)
        x = self.linformer(x)
        x = self.final(torch.mean(x, dim=1))
        return x
