import torch.nn as nn
from torch.nn.utils import weight_norm


class CDIL_Block(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, dropout):
        super(CDIL_Block, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, padding_mode='circular', dilation=dilation))
        self.net = nn.Sequential(self.conv1, self.dropout)

        self.res_shape = nn.Conv1d(n_inputs, n_outputs, kernel_size=(1,)) if n_inputs != n_outputs else None
        self.nonlinear = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.normal_(0, 0.01)
        if self.res_shape is not None:
            self.res_shape.weight.data.normal_(0, 0.01)
            self.res_shape.bias.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.res_shape is None else self.res_shape(x)
        return self.nonlinear(out) + res


class CDIL_ConvPart(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0):
        super(CDIL_ConvPart, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            this_padding = int(dilation_size*(kernel_size-1)/2)

            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [CDIL_Block(in_channels, out_channels, kernel_size, dilation=dilation_size, padding=this_padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
