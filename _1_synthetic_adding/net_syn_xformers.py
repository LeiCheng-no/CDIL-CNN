import sys
import torch
import torch.nn as nn
from performer_pytorch import Performer
from linformer import Linformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformers(nn.Module):
    def __init__(self, model_name, in_dim, n_class, n_vec, dim, depth, heads):
        super(Transformers, self).__init__()
        self.linear = nn.Linear(in_dim, dim, bias=True)
        if model_name == 'Transformer':
            encoder_layers = TransformerEncoderLayer(dim, heads, dim, batch_first=True)
            self.former = TransformerEncoder(encoder_layers, depth)
        elif model_name == 'Linformer':
            self.former = Linformer(dim=dim, seq_len=n_vec, depth=depth, heads=heads, k=dim, one_kv_head=True, share_kv=True)
        elif model_name == 'Performer':
            self.former = Performer(dim=dim, depth=depth, heads=heads, dim_head=dim, causal=True)
        else:
            print('no this model.')
            sys.exit()
        self.final = nn.Linear(dim, n_class)

    def forward(self, x):
        x = self.linear(x)
        x = self.former(x)
        x = self.final(torch.mean(x, dim=1))
        return x
