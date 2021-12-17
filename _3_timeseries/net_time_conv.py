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


# block: CNN, CDIL and TCN
class Block(nn.Module):
    def __init__(self, model_name, n_inputs, n_outputs, kernel_size, dilation, padding, dropout):
        super(Block, self).__init__()
        self.dropout = nn.Dropout(dropout)

        if model_name == 'CDIL':
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=0, dilation=dilation))
            self.padding = circle(padding)
            self.net = nn.Sequential(self.padding, self.conv1, self.dropout)
        elif model_name == 'TCN':
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation))
            self.cut = tcn(padding)
            self.net = nn.Sequential(self.conv1, self.cut, self.dropout)
        elif model_name == 'CNN':
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation))
            self.net = nn.Sequential(self.conv1, self.dropout)

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


# conv: CNN, CDIL and TCN
class ConvPart(nn.Module):
    def __init__(self, model_name, num_inputs, num_channels, kernel_size, dropout):
        super(ConvPart, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            if model_name == 'TCN' or model_name == 'CDIL':
                dilation_size = 2 ** i
                if model_name == 'TCN':
                    this_padding = (kernel_size-1) * dilation_size
                else:
                    this_padding = int(dilation_size*(kernel_size-1)/2)
            elif model_name == 'CNN':
                dilation_size = 1
                this_padding = int((kernel_size-1)/2)
            else:
                print('no this model.')
                sys.exit()
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [Block(model_name, in_channels, out_channels, kernel_size, dilation=dilation_size, padding=this_padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# model: CNN, CDIL and TCN
class ConvNet(nn.Module):
    def __init__(self, model_name, input_size, output_size, seq_len, num_channels, kernel_size, use_cuda, dropout=0):
        super(ConvNet, self).__init__()
        self.n_vec = seq_len
        self.use_cuda = use_cuda
        self.name = model_name
        self.conv = ConvPart(model_name, input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # print(x.shape)
        # print(x)
        x = x.permute(0, 2, 1).to(dtype=torch.float)
        # print(x.shape)
        # print(x)
        y_conv = self.conv(x)   # x, y: num, channel(dim), length
        if self.name == 'TCN':
            y = self.linear(y_conv[:, :, -1])
        else:
            y = self.linear(torch.mean(y_conv, dim=2))
        return y
