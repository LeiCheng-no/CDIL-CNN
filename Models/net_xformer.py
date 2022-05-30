import sys
import math
import torch
import torch.nn as nn
from linformer import Linformer
from performer_pytorch import Performer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


#  Refer to https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]  # [seq_len, batch_size, dim]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class Xformer(nn.Module):
    def __init__(self, model, input_size, output_size, dim, seq_len, depth, heads, use_pos=True):
        super(Xformer, self).__init__()
        self.model = model
        self.use_pos = use_pos
        self.linear = nn.Linear(input_size, dim)
        if use_pos:
            self.pos_enc = PositionalEncoding(dim, seq_len)
        
        if model == 'Transformer':
            encoder_layers = TransformerEncoderLayer(dim, heads, dim)
            self.former = TransformerEncoder(encoder_layers, depth)
        elif model == 'Linformer':
            self.former = Linformer(dim=dim, seq_len=seq_len, depth=depth, heads=heads, k=dim, one_kv_head=True, share_kv=True)
        elif model == 'Performer':
            self.former = Performer(dim=dim, depth=depth, heads=heads, dim_head=dim, causal=True)
        else:
            print('no this model.')
            sys.exit()
        
        self.final = nn.Linear(dim, output_size)

    def forward(self, x):
        x = self.linear(x)
        if self.use_pos:
            x = self.pos_enc(x)   # out: num, length, dim
        if self.model == 'Transformer':
            x = x.permute(1, 0, 2)
            # print(x.shape)
            # sys.exit()
            x = self.former(x)
            x = x.permute(1, 0, 2)
        else:
            # print(x.shape)
            # sys.exit()
            x = self.former(x)
        x = self.final(torch.mean(x, dim=1))
        return x
