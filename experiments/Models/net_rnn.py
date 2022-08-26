import sys
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, task, model, input_size, output_size, hidden_size, num_layers, use_embed=False, char_vocab=None, fix_length=True):
        super(RNN, self).__init__()
        self.task = task
        self.use_embed = use_embed
        self.fix_length = fix_length

        if self.use_embed:
            self.embedding = nn.Embedding(char_vocab, input_size)

        if model == 'LSTM':
            self.rnn_func = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        elif model == 'GRU':
            self.rnn_func = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        else:
            print('no this model.')
            sys.exit()

        if self.task != 'retrieval_4000':
            self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, mask=None):
        if self.use_embed:
            x = self.embedding(x)  # out: num, length, dim
        x = x.permute(1, 0, 2)  # out: length, num, dim
        # print(x.shape)
        # sys.exit()
        y_rnn, _ = self.rnn_func(x)

        if self.fix_length:
            y_class = y_rnn[-1, :, :]
        else:
            P = mask.unsqueeze(1).expand(y_rnn.size(1), y_rnn.size(2)).unsqueeze(0)
            y_class = y_rnn.gather(0, P).squeeze(0)

        if self.task == 'retrieval_4000':
            return y_class
        else:
            y = self.linear(y_class)
            return y
