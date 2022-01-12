import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime


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


def count_params(net):
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return n_params


class Net(nn.Module):
    def __init__(self, model_part, dim, n_class):
        super(Net, self).__init__()
        self.model = model_part
        self.linear = nn.Linear(dim, n_class)

    def forward(self, x, mask):
        y_dim = self.model(x, mask)
        y = self.linear(y_dim)
        return y


class NetDual(nn.Module):
    def __init__(self, model_part, dim, n_class):
        super(NetDual, self).__init__()
        self.model = model_part
        self.linear = nn.Linear(dim*4, n_class)

    def forward(self, x1, mask1, x2, mask2):
        y_dim1 = self.model(x1, mask1)
        y_dim2 = self.model(x2, mask2)
        y_class = torch.cat([y_dim1, y_dim2, y_dim1 * y_dim2, y_dim1 - y_dim2], dim=1)
        y = self.linear(y_class)
        return y


def TrainModel(
        net,
        fix_length,
        trainloader,
        valloader,
        testloader,
        n_epochs,
        eval_freq,
        optimizer,
        loss,
        problem,
        saving_best,
        loginf
):
    if problem == 'retrieval_4000':
        for epoch in range(n_epochs):
            # Training
            running_loss = 0
            t_start = datetime.now()
            for _, one_input in tqdm(enumerate(trainloader), total=len(trainloader)):
                X0 = one_input['input_ids_0']
                X1 = one_input['input_ids_1']
                Y = one_input['label']
                if not fix_length:
                    decision_position0 = torch.sum(one_input['mask_0'], dim=1).long() - 1
                    decision_position1 = torch.sum(one_input['mask_1'], dim=1).long() - 1
                else:
                    decision_position0 = torch.zeros(1)
                    decision_position1 = torch.zeros(1)
                X0 = X0.cuda()
                X1 = X1.cuda()
                Y = Y.cuda()
                decision_position0 = decision_position0.cuda()
                decision_position1 = decision_position1.cuda()

                optimizer.zero_grad()
                pred = net(X0, decision_position0, X1, decision_position1)

                output = loss(pred, Y)
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
                    for _, one_input in enumerate(valloader):
                        X0 = one_input['input_ids_0']
                        X1 = one_input['input_ids_1']
                        Y = one_input['label']
                        if not fix_length:
                            decision_position0 = torch.sum(one_input['mask_0'], dim=1).long() - 1
                            decision_position1 = torch.sum(one_input['mask_1'], dim=1).long() - 1
                        else:
                            decision_position0 = torch.zeros(1)
                            decision_position1 = torch.zeros(1)
                        X0 = X0.cuda()
                        X1 = X1.cuda()
                        Y = Y.cuda()
                        decision_position0 = decision_position0.cuda()
                        decision_position1 = decision_position1.cuda()

                        optimizer.zero_grad()
                        pred = net(X0, decision_position0, X1, decision_position1)

                        val_loss += loss(pred, Y).item()
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
                    with torch.no_grad():
                        # Testing loop
                        for _, one_input in enumerate(testloader):
                            X0 = one_input['input_ids_0']
                            X1 = one_input['input_ids_1']
                            Y = one_input['label']
                            if not fix_length:
                                decision_position0 = torch.sum(one_input['mask_0'], dim=1).long() - 1
                                decision_position1 = torch.sum(one_input['mask_1'], dim=1).long() - 1
                            else:
                                decision_position0 = torch.zeros(1)
                                decision_position1 = torch.zeros(1)
                            X0 = X0.cuda()
                            X1 = X1.cuda()
                            Y = Y.cuda()
                            decision_position0 = decision_position0.cuda()
                            decision_position1 = decision_position1.cuda()

                            optimizer.zero_grad()
                            pred = net(X0, decision_position0, X1, decision_position1)

                            test_loss += loss(pred, Y).item()
                            _, predicted = pred.max(1)
                            total_test += Y.size(0)

                            correct_test += predicted.eq(Y).sum().item()
                        loginf("Test loss: {}".format(test_loss / len(testloader)))
                        accuracy_test = 100. * correct_test / total_test
                        loginf("Test accuracy: {}".format(accuracy_test))
                        loginf('_' * 100)

                        # saving_epoch = epoch
                        # torch.save(net.state_dict(), '{}_epoch{}_acc{}_test{}.pt'.format(problem+str(length), saving_epoch, saving_best, accuracy_test))
                    net.train()
    else:
        for epoch in range(n_epochs):
            # Training
            running_loss = 0
            t_start = datetime.now()
            # print(len(trainloader))
            for _, one_input in tqdm(enumerate(trainloader), total=len(trainloader)):
                # for key in one_input:
                #     print(key)
                X = one_input['input_ids_0']
                # print(X)
                # print(X.shape)
                Y = one_input['label']
                if not fix_length:
                    decision_position = torch.sum(one_input['mask_0'], dim=1).long() - 1
                else:
                    decision_position = torch.zeros(1)
                # print(decision_position.shape)
                # print(decision_position)
                X = X.cuda()
                Y = Y.cuda()
                decision_position = decision_position.cuda()

                optimizer.zero_grad()
                pred = net(X, decision_position)

                # print(pred.shape)

                output = loss(pred, Y)
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
                    for _, one_input in enumerate(valloader):
                        X = one_input['input_ids_0']
                        Y = one_input['label']
                        if not fix_length:
                            decision_position = torch.sum(one_input['mask_0'], dim=1).long() - 1
                        else:
                            decision_position = torch.zeros(1)

                        X = X.cuda()
                        Y = Y.cuda()
                        decision_position = decision_position.cuda()

                        optimizer.zero_grad()
                        pred = net(X, decision_position)

                        val_loss += loss(pred, Y).item()
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
                    with torch.no_grad():
                        # Testing loop
                        for _, one_input in enumerate(testloader):
                            X = one_input['input_ids_0']
                            Y = one_input['label']
                            if not fix_length:
                                decision_position = torch.sum(one_input['mask_0'], dim=1).long() - 1
                            else:
                                decision_position = torch.zeros(1)

                            X = X.cuda()
                            Y = Y.cuda()
                            decision_position = decision_position.cuda()

                            optimizer.zero_grad()
                            pred = net(X, decision_position)

                            test_loss += loss(pred, Y).item()
                            _, predicted = pred.max(1)
                            total_test += Y.size(0)

                            correct_test += predicted.eq(Y).sum().item()
                        loginf("Test loss: {}".format(test_loss / len(testloader)))
                        accuracy_test = 100. * correct_test / total_test
                        loginf("Test accuracy: {}".format(accuracy_test))
                        loginf('_' * 100)

                        # saving_epoch = epoch
                        # torch.save(net.state_dict(), '{}_epoch{}_acc{}_test{}.pt'.format(problem+str(length), saving_epoch, saving_best, accuracy_test))
                    net.train()
