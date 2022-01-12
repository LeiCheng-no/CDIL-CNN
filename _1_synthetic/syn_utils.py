import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset


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


def TrainModel(
        net,
        trainloader,
        valloader,
        testloader,
        n_epochs,
        eval_freq,
        optimizer,
        loss,
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
                for _, (X, Y) in enumerate(valloader):
                    X = X.cuda()
                    Y = Y.cuda()
                    pred = net(X)
                    val_loss += loss(pred.squeeze(), Y).item()
                    _, predicted = pred.max(1)
                    total_val += Y.size(0)
                    correct_val += (torch.abs(pred.squeeze() - Y) < 0.04).sum()
                loginf("Val  loss: {}".format(val_loss / len(valloader)))
                accuracy_val = correct_val / total_val * 100
                loginf("Val  accuracy: {}".format(accuracy_val))
                loginf('_' * 40)
                net.train()

                if accuracy_val >= saving_best:
                    saving_best = accuracy_val
                    net.eval()
                    # Testing loop
                    for _, (X, Y) in enumerate(testloader):
                        X = X.cuda()
                        Y = Y.cuda()
                        pred = net(X)
                        test_loss += loss(pred.squeeze(), Y).item()
                        _, predicted = pred.max(1)
                        total_test += Y.size(0)
                        correct_test += (torch.abs(pred.squeeze() - Y) < 0.04).sum()
                    loginf("Test loss: {}".format(test_loss / len(testloader)))
                    accuracy_test = correct_test / total_test * 100
                    loginf("Test accuracy: {}".format(accuracy_test))
                    loginf('_' * 80)
                    net.train()

        if accuracy_test == 100:
            sys.exit()
