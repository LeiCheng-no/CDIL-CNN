import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime


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


def retrieval(one_input, fix_length, device):
    X0 = one_input['input_ids_0']
    X1 = one_input['input_ids_1']
    Y = one_input['label']
    if not fix_length:
        decision_position0 = torch.sum(one_input['mask_0'], dim=1).long() - 1
        decision_position1 = torch.sum(one_input['mask_1'], dim=1).long() - 1
    else:
        decision_position0 = torch.zeros(1)
        decision_position1 = torch.zeros(1)
    X0, X1, Y = X0.to(device), X1.to(device), Y.to(device)
    mask0 = decision_position0.to(device)
    mask1 = decision_position1.to(device)
    return X0, mask0, X1, mask1, Y


def other(one_input, fix_length, device):
    X = one_input['input_ids_0']
    Y = one_input['label']
    if not fix_length:
        decision_position = torch.sum(one_input['mask_0'], dim=1).long() - 1
    else:
        decision_position = torch.zeros(1)
    X, Y, mask = X.to(device), Y.to(device), decision_position.to(device)
    return X, mask, Y


def net_eval(task, fix_length, val_test, n, eval_loader, device, net, loss, loginf):
    eval_loss = 0
    eval_num = 0
    eval_correct = 0
    eval_start = datetime.now()
    for one_input in tqdm(eval_loader, total=len(eval_loader)):
        if task == 'retrieval_4000':
            X0, mask0, X1, mask1, Y = retrieval(one_input, fix_length, device)
            pred = net(X0, mask0, X1, mask1)
        else:
            X, mask, Y = other(one_input, fix_length, device)
            pred = net(X, mask)
        eval_loss += loss(pred, Y).item()
        eval_num += len(Y)
        _, predicted = pred.max(1)
        eval_correct += predicted.eq(Y).sum().item()
    eval_loss_mean = eval_loss / eval_num
    eval_acc = eval_correct / eval_num * 100
    eval_end = datetime.now()
    eval_time = (eval_end - eval_start).total_seconds()
    loginf('{} num: {} — {} loss: {} — {} accuracy: {} — Time: {}'.format(val_test, eval_num, val_test, eval_loss_mean, val_test, eval_acc, eval_time))
    loginf('_' * n)
    return eval_loss_mean, eval_acc


def TrainModel(
        task,
        fix_length,
        net,
        device,
        trainloader,
        valloader,
        testloader,
        n_epochs,
        optimizer,
        loss,
        loginf,
        wandb,
        file_name
):
    saving_best = 0

    for epoch in range(n_epochs):
        # train
        net.train()

        train_loss = 0
        train_num = 0
        t_start = datetime.now()
        for one_input in tqdm(trainloader, total=len(trainloader)):
            optimizer.zero_grad()
            if task == 'retrieval_4000':
                X0, mask0, X1, mask1, Y = retrieval(one_input, fix_length, device)
                pred = net(X0, mask0, X1, mask1)
            else:
                X, mask, Y = other(one_input, fix_length, device)
                pred = net(X, mask)
            batch_loss = loss(pred, Y)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            train_num += len(Y)
        train_loss_mean = train_loss / train_num
        t_end = datetime.now()
        epoch_time = (t_end - t_start).total_seconds()
        loginf('Epoch: {}'.format(epoch))
        loginf('Train num: {} — Train loss: {} — Time: {}'.format(train_num, train_loss_mean, epoch_time))

        # validation and test
        with torch.no_grad():
            net.eval()
            val_loss_mean, val_acc = net_eval(task, fix_length, 'Val', 80, valloader, device, net, loss, loginf)
            if val_acc >= saving_best:
                saving_best = val_acc
                torch.save(net.state_dict(), file_name)
                _, test_acc = net_eval(task, fix_length, 'Test', 120, testloader, device, net, loss, loginf)

        if wandb is not None:
            wandb.log({"train loss": train_loss_mean,
                       "val loss": val_loss_mean,
                       "val acc": val_acc,
                       })
    loginf('best test acc: {}'.format(test_acc))
    loginf('_' * 200)
