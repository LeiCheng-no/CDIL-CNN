import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datetime import datetime
from tqdm import tqdm


def seed_everything(seed=1234):
    """
    Fixes random seeds, to get reproducible results.
    :param seed: a random seed across all the used packages
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.data[index]
        Y = self.labels[index]
        return X, Y

    def __len__(self):
        return len(self.labels)


def count_params(net):
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return n_params


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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


def TrainModel(
        net,
        trainloader,
        valloader,
        testloader,
        n_epochs,
        eval_freq,
        optimizer,
        loss,
        problem,
        length,
        saving_best,
        loginf
):
    for epoch in range(n_epochs):
        # Training
        running_loss = 0
        t_start = datetime.now()
        for _, (X, Y) in tqdm(enumerate(trainloader), total=len(trainloader)):
            X = X.cuda()
            Y = Y.cuda()
            optimizer.zero_grad()
            pred = net(X)
            output = loss(pred.squeeze(), Y)
            output.backward()
            optimizer.step()

            running_loss += output.item()
        t_end = datetime.now()

        loginf("Epoch {} - Training loss:  {} — Time:  {}sec".format(
            epoch,
            running_loss / len(trainloader),
            (t_end - t_start).total_seconds()
            )
        )

        # Validation
        if epoch % eval_freq == 0:
            net.eval()
            total_val = 0
            total_test = 0
            correct_val = 0
            correct_test = 0
            val_loss = 0.0
            test_loss = 0.0
            with torch.no_grad():
                # Validation loop
                for _, (X, Y) in tqdm(enumerate(valloader), total=len(valloader)):
                    X = X.cuda()
                    Y = Y.cuda()
                    pred = net(X)
                    val_loss += loss(pred.squeeze(), Y).item()
                    _, predicted = pred.max(1)
                    total_val += Y.size(0)
                    correct_val += predicted.eq(Y).sum().item()
                loginf("Val  loss: {}".format(val_loss / len(valloader)))
                accuracy_val = 100. * correct_val / total_val
                loginf("Val  accuracy: {}".format(accuracy_val))
                loginf('_' * 40)
                net.train()

                if accuracy_val >= saving_best:
                    saving_best = accuracy_val
                    net.eval()
                    # Testing loop
                    for _, (X, Y) in tqdm(enumerate(testloader), total=len(testloader)):
                        X = X.cuda()
                        Y = Y.cuda()
                        pred = net(X)
                        test_loss += loss(pred.squeeze(), Y).item()
                        _, predicted = pred.max(1)
                        total_test += Y.size(0)
                        correct_test += predicted.eq(Y).sum().item()
                    loginf("Test loss: {}".format(test_loss / len(testloader)))
                    accuracy_test = 100. * correct_test / total_test
                    loginf("Test accuracy: {}".format(accuracy_test))
                    loginf('_' * 80)
                    net.train()

