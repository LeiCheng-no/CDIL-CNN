import sys
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, model_name, input_size, output_size, seq_len, hidden_size, num_layers, use_cuda, dropout=0, bi=False):
        super(RNN, self).__init__()

        self.n_vec = seq_len
        self.use_cuda = use_cuda

        if model_name == 'LSTM':
            self.rnn_func = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bi)
        elif model_name == 'GRU':
            self.rnn_func = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bi)
        else:
            print('no this model.')
            sys.exit()

        if bi:
            self.linear = nn.Linear(hidden_size * 4, output_size)
        else:
            self.linear = nn.Linear(hidden_size, output_size)
        self.bi = bi

    def forward(self, x):
        # print(x.shape)
        # print(x)
        x = x.permute(1, 0, 2)
        # print(x.shape)
        y_rnn, _ = self.rnn_func(x)  # x, y_rnn: length, num, dim
        if self.bi:
            y_class = torch.cat((y_rnn[0, :, :], y_rnn[-1, :, :]), dim=1)
        else:
            y_class = y_rnn[-1, :, :]
        y = self.linear(y_class)
        return y
