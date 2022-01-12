import sys
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# CDIL
class circle(nn.Module):
    def __init__(self, circle_size):
        super(circle, self).__init__()
        self.circle_size = circle_size

    def forward(self, x):
        x_new = torch.cat([x[:, :, -self.circle_size:], x, x[:, :, :self.circle_size]], dim=2)
        return x_new.contiguous()


# TCN
class tcn(nn.Module):
    def __init__(self, tcn_size):
        super(tcn, self).__init__()
        self.tcn_size = tcn_size

    def forward(self, x):
        x_new = x[:, :, :-self.tcn_size]
        return x_new.contiguous()


# block: CDIL and TCN
class Block(nn.Module):
    def __init__(self, model_name, n_inputs, n_outputs, kernel_size, dilation, padding, dropout):
        super(Block, self).__init__()
        self.dropout = nn.Dropout(dropout)

        if model_name == 'CDIL':
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, dilation=dilation))
            self.padding = circle(padding)
            self.net = nn.Sequential(self.padding, self.conv1, self.dropout)
        elif model_name == 'TCN':
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation))
            self.cut = tcn(padding)
            self.net = nn.Sequential(self.conv1, self.cut, self.dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=(1,)) if n_inputs != n_outputs else None
        self.nonlinear = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            self.downsample.bias.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.nonlinear(out) + res


# conv: CDIL and TCN
class ConvPart(nn.Module):
    def __init__(self, model_name, num_inputs, num_channels, kernel_size, dropout=0):
        super(ConvPart, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            if model_name == 'CDIL':
                this_padding = int(dilation_size*(kernel_size-1)/2)
            elif model_name == 'TCN':
                this_padding = (kernel_size-1) * dilation_size
            else:
                print('no this model.')
                sys.exit()
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [Block(model_name, in_channels, out_channels, kernel_size, dilation=dilation_size, padding=this_padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# model: CDIL and TCN
class ConvNet(nn.Module):
    def __init__(self, model_name, input_size, num_channels, kernel_size, use_embed, char_vocab, fix_length, dropout=0):
        super(ConvNet, self).__init__()
        self.use_embed = use_embed
        if self.use_embed:
            self.embedding = nn.Embedding(char_vocab, input_size)

        self.conv = ConvPart(model_name, input_size, num_channels, kernel_size, dropout)

        self.name = model_name
        self.fix_lengh = fix_length

    def forward(self, x, mask=None):  # x: num, length, dim
        # print(x.shape)
        # print(x)
        if self.use_embed:
            x = self.embedding(x).squeeze(-2)
        # print(x.shape)
        # print(x)
        x = x.permute(0, 2, 1).to(dtype=torch.float)
        # print(x.shape)
        # print(x)
        y_conv = self.conv(x)   # x, y: num, channel(dim), length
        # print(y_conv.shape)
        if self.name == 'TCN':
            if not self.fix_lengh:
                # print(mask.shape)
                P = mask.unsqueeze(1).expand(y_conv.size(0), y_conv.size(1)).unsqueeze(2)
                # print(P.shape)
                # print(P)
                y_end = y_conv.gather(2, P).squeeze(2)
                # print(y_end.shape)
            else:
                y_end = y_conv[:, :, -1]
        else:
            y_end = torch.mean(y_conv, dim=2)
        return y_end
