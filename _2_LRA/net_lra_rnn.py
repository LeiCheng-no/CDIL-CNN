import sys
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, model_name, input_size, hidden_size, num_layers, use_embed, char_vocab, fix_length, dropout=0):
        super(RNN, self).__init__()
        self.use_embed = use_embed
        if self.use_embed:
            self.embedding = nn.Embedding(char_vocab, input_size)

        if model_name == 'LSTM':
            self.rnn_func = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False)
        elif model_name == 'GRU':
            self.rnn_func = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False)
        else:
            print('no this model.')
            sys.exit()

        self.fix_lengh = fix_length

    def forward(self, x, mask):   # x: num, length
        # print(x.shape)
        # print(x)
        if self.use_embed:
            x = self.embedding(x).squeeze(-2)  # out: x: num, length, dim
        # print(self.use_embed)
        # print(x.shape)
        # print(x)
        x = x.permute(1, 0, 2)
        # print(x.shape)
        y_rnn, _ = self.rnn_func(x)  # x, y_rnn: length, num, dim
        # print(y_rnn.shape)

        if not self.fix_lengh:
            P = mask.unsqueeze(1).expand(y_rnn.size(1), y_rnn.size(2)).unsqueeze(0)
            # print(P.shape)
            # print(P)
            y_end = y_rnn.gather(0, P).squeeze(0)
            # print(y_end.shape)
        else:
            y_end = y_rnn[-1, :, :]
            # print(y_end.shape)
        return y_end
