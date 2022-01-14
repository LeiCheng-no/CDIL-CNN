import torch
import torch.nn as nn
from cdil import CDIL_ConvPart


class CDIL_CNN(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_channel, layers, kernel_size=3, dropout=0, use_embed=False, char_vocab=None):
        super(CDIL_CNN, self).__init__()

        self.use_embed = use_embed
        if self.use_embed:
            self.embedding = nn.Embedding(char_vocab, input_dim)

        self.conv = CDIL_ConvPart(input_dim, [hidden_channel] * layers, kernel_size, dropout)
        self.classifier = nn.Linear(hidden_channel, out_dim)

    def forward(self, x):
        # print(x.shape)
        if self.use_embed:
            x = self.embedding(x)
            # print(x.shape)
            x = x.permute(0, 2, 1).to(dtype=torch.float)
        y_conv = self.conv(x)  # x, y: num, channel(dim), length
        # print(y_conv.shape)
        y = self.classifier(torch.mean(y_conv, dim=2))
        # print(y.shape)
        return y


def main():
    # 1. cdil-cnn convolutional part outputs the same size(length) as the input sequence
    print('demo1:', '='*30)
    SEQ_LENGTH = 100  # remain unchanged
    INPUT_DIM = 10

    BATCH = 32
    x = torch.rand(BATCH, INPUT_DIM, SEQ_LENGTH)
    print(x.shape)  # torch.Size([32, 10, 100])

    HIDDEN_CHANNEL = 50
    LAYER = 4
    cdil_conv_part1 = CDIL_ConvPart(INPUT_DIM, [HIDDEN_CHANNEL] * LAYER, kernel_size=3, dropout=0)
    y_conv = cdil_conv_part1(x)
    print(y_conv.shape)  # torch.Size([32, 50, 100])

    cdil_conv_part2 = CDIL_ConvPart(INPUT_DIM, [20, 30, 40], kernel_size=3, dropout=0)
    y_conv = cdil_conv_part2(x)
    print(y_conv.shape)  # torch.Size([32, 40, 100])

    # 2. cdil-cnn model (classifier) using input sequences without embedding
    print('demo2:', '=' * 30)
    USE_EMBED = False

    SEQ_LENGTH = 200
    INPUT_DIM = 3
    OUTPUT_CLASS = 11

    BATCH = 64
    HIDDEN_CHANNEL = 20
    LAYER = 3

    cdil_model_noembed = CDIL_CNN(INPUT_DIM, OUTPUT_CLASS, HIDDEN_CHANNEL, LAYER, 3, 0, USE_EMBED)
    x = torch.rand(BATCH, INPUT_DIM, SEQ_LENGTH)
    print(x.shape)  # torch.Size([64, 3, 200])
    y = cdil_model_noembed(x)
    print(y.shape)  # torch.Size([64, 11])

    # 3.  cdil-cnn model (classifier) using input sequences with embedding
    print('demo3:', '=' * 30)
    USE_EMBED = True
    CHAR_VOCAB = 5

    SEQ_LENGTH = 300
    EMBED_DIM = 4
    OUTPUT_CLASS = 13

    BATCH = 16
    HIDDEN_CHANNEL = 30
    LAYER = 4

    cdil_model_embed = CDIL_CNN(EMBED_DIM, OUTPUT_CLASS, HIDDEN_CHANNEL, LAYER, 3, 0, USE_EMBED, CHAR_VOCAB)

    x = torch.randint(CHAR_VOCAB, (BATCH, SEQ_LENGTH))
    print(x.shape)  # torch.Size([16, 300])
    y = cdil_model_embed(x)
    print(y.shape)  # torch.Size([64, 13])


if __name__ == "__main__":
    main()
