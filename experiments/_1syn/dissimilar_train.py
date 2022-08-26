import torch
from tqdm import tqdm
from datetime import datetime


def net_eval(val_test, eval_loader, device, net, loss, loginf):
    eval_loss = 0
    eval_num = 0
    eval_correct = 0
    eval_start = datetime.now()
    for eval_X, eval_Y in tqdm(eval_loader, total=len(eval_loader)):
        eval_X, eval_Y = eval_X.to(device), eval_Y.to(device)
        eval_pred = net(eval_X)
        eval_loss += loss(eval_pred.squeeze(), eval_Y).item()
        eval_num += len(eval_Y)
        _, predicted = eval_pred.max(1)
        eval_correct += predicted.eq(eval_Y).sum().item()
    eval_loss_mean = eval_loss / eval_num
    eval_acc = eval_correct / eval_num * 100
    eval_end = datetime.now()
    eval_time = (eval_end - eval_start).total_seconds()
    loginf('{} num: {} — {} loss: {} — {} accuracy: {} — Time: {}'.format(val_test, eval_num, val_test, eval_loss_mean, val_test, eval_acc, eval_time))
    return eval_loss_mean, eval_acc


def TrainModel(
        net,
        device,
        trainloader,
        valloader,
        testloader,
        dtestloader,
        n_epochs,
        optimizer,
        loss,
        loginf,
        wandb,
        file_name
):
    saving_best = 0
    best_test = 0
    best_dtest = 0

    for epoch in range(n_epochs):
        # train
        net.train()

        train_loss = 0
        train_num = 0
        train_correct = 0
        t_start = datetime.now()
        for X, Y in tqdm(trainloader, total=len(trainloader)):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = net(X)
            batch_loss = loss(pred.squeeze(), Y)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            train_num += len(Y)
            _, predicted = pred.max(1)
            train_correct += predicted.eq(Y).sum().item()
        train_loss_mean = train_loss / train_num
        train_acc = train_correct / train_num * 100
        t_end = datetime.now()
        epoch_time = (t_end - t_start).total_seconds()
        loginf('Epoch: {}'.format(epoch))
        loginf('Train num: {} — Train loss: {} — Train accuracy: {}  — Time: {}'.format(train_num, train_loss_mean, train_acc, epoch_time))

        # validation and test
        with torch.no_grad():
            net.eval()

            val_loss_mean, val_acc = net_eval('Val', valloader, device, net, loss, loginf)
            if val_acc >= saving_best:
                saving_best = val_acc
                torch.save(net.state_dict(), file_name + '.ph')

                test_loss_mean, test_acc = net_eval('similar', testloader, device, net, loss, loginf)
                if test_acc >= best_test:
                    best_test = test_acc

                dtest_loss_mean, dtest_acc = net_eval('dissimilar', dtestloader, device, net, loss, loginf)
                if dtest_acc >= best_dtest:
                    best_dtest = dtest_acc

            if wandb is not None:
                wandb.log({
                    # "train loss": train_loss_mean,
                    # "val loss": val_loss_mean,
                    # "test loss": test_loss_mean,
                    # "dval loss": dval_loss_mean,
                    # "dtest loss": dtest_loss_mean,
                    # "acc train": train_acc,
                    # "acc val": val_acc,
                    "acc test": test_acc,
                    # "acc dval": dval_acc,
                    "acc dtest": dtest_acc,
                })
        loginf('_' * 120)
    loginf('best test acc similar: {} — best test acc dissimilar: {}'.format(best_test, best_dtest))
    loginf('final test acc similar: {} — final test acc dissimilar: {}'.format(test_acc, dtest_acc))
    loginf('_' * 200)
