import os
import sys
import wandb
import torch
import logging
import argparse
import numpy as np
from torch.utils.data import DataLoader

from time_config import config
from noise_train import TrainModel

sys.path.append('../')
from Models.net_conv import CONV
from Models.net_conv_rf import receptive_field
from Models.net_rnn import RNN
from Models.net_xformer import Xformer
from Models.utils import seed_everything, DatasetCreator

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--task', type=str, default='RightWhaleCalls')
parser.add_argument('--model', type=str, default='CDIL')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


# Config
use_wandb = False
INPUT_SIZE = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TASK = args.task
MODEL = args.model
SEED = args.seed

seed_everything(args.seed)
if use_wandb:
    wandb.init(project=TASK + '_noise', name=MODEL + str(SEED), entity='leic-no')
    WANDB = wandb
else:
    WANDB = None

cfg_training = config[TASK]['training']
cfg_model = config[TASK]['models']

BATCH = cfg_training['batch_size']


# Data
features_train = np.loadtxt(f'./noise_datasets/{TASK}_train_data.csv', delimiter=',')
labels_train = np.loadtxt(f'./noise_datasets/{TASK}_train_label.csv', delimiter=',')
features_train = torch.tensor(features_train).view(len(labels_train), -1, 1).float()
labels_train = torch.tensor(labels_train).long()
trainset = DatasetCreator(features_train, labels_train)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True, drop_last=False)

features_val = np.loadtxt(f'./noise_datasets/{TASK}_val_data.csv', delimiter=',')
labels_val = np.loadtxt(f'./noise_datasets/{TASK}_val_label.csv', delimiter=',')
features_val = torch.tensor(features_val).view(len(labels_val), -1, 1).float()
labels_val = torch.tensor(labels_val).long()
valset = DatasetCreator(features_val, labels_val)
valloader = DataLoader(valset, batch_size=BATCH, shuffle=False, drop_last=False)

features_test = np.loadtxt(f'./noise_datasets/{TASK}_test_data.csv', delimiter=',')
labels_test = np.loadtxt(f'./noise_datasets/{TASK}_test_label.csv', delimiter=',')
features_test = torch.tensor(features_test).view(len(labels_test), -1, 1).float()
labels_test = torch.tensor(labels_test).long()
testset = DatasetCreator(features_test, labels_test)
testloader = DataLoader(testset, batch_size=BATCH, shuffle=False, drop_last=False)

features_dtest = np.loadtxt(f'./noise_datasets/{TASK}_dtest_data.csv', delimiter=',')
labels_dtest = np.loadtxt(f'./noise_datasets/{TASK}_dtest_label.csv', delimiter=',')
features_dtest = torch.tensor(features_dtest).view(len(labels_dtest), -1, 1).float()
labels_dtest = torch.tensor(labels_dtest).long()
dtestset = DatasetCreator(features_dtest, labels_dtest)
dtestloader = DataLoader(dtestset, batch_size=BATCH, shuffle=False, drop_last=False)

CLASS = len(torch.unique(labels_train))
SEQ_LEN = features_train.shape[1]
LAYER = int(np.log2(SEQ_LEN))


# Model
if MODEL == 'CDIL' or MODEL == 'DIL' or MODEL == 'TCN' or MODEL == 'CNN':
    NHID = cfg_model['cnn_hidden']
    KERNEL_SIZE = cfg_model['cnn_ks']
    net = CONV(TASK, MODEL, INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == 'Deformable':
    NHID = cfg_model['cnn_hidden']
    KERNEL_SIZE = cfg_model['cnn_ks']
    net = CONV(TASK, 'CNN', INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE, True)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == 'LSTM' or MODEL == 'GRU':
    LAYER = cfg_model['rnn_layer']
    NHID = cfg_model['rnn_hidden']
    net = RNN(TASK, MODEL, INPUT_SIZE, CLASS, NHID, LAYER)
elif MODEL == 'Transformer' or MODEL == 'Linformer' or MODEL == 'Performer':
    DIM = cfg_model['former_dim']
    DEPTH = cfg_model['former_depth']
    HEADS = cfg_model['former_head']
    net = Xformer(MODEL, INPUT_SIZE, CLASS, DIM, SEQ_LEN, DEPTH, HEADS)
else:
    print('no this model.')
    sys.exit()

net = net.to(device)
para_num = sum(p.numel() for p in net.parameters() if p.requires_grad)


# Log
if MODEL == 'Transformer' or MODEL == 'Linformer' or MODEL == 'Performer':
    file_name = TASK + '_P' + str(para_num) + '_' + MODEL + '_S' + str(SEED) + '_H' + str(DIM)
else:
    file_name = TASK + '_P' + str(para_num) + '_' + MODEL + '_S' + str(SEED) + '_L' + str(LAYER) + '_H' + str(NHID)

os.makedirs('noise_log', exist_ok=True)
os.makedirs('noise_model', exist_ok=True)
log_file_name = './noise_log/' + file_name + '.txt'
model_name = './noise_model/' + file_name + '.ph'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

loginf(torch.cuda.get_device_name(device))
loginf(file_name)


# Optimize
optimizer = torch.optim.Adam(net.parameters())
loss = torch.nn.CrossEntropyLoss(reduction='sum')


# train
TrainModel(
    net=net,
    device=device,
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    dtestloader=dtestloader,
    n_epochs=cfg_training['epoch'],
    optimizer=optimizer,
    loss=loss,
    loginf=loginf,
    wandb=WANDB,
    file_name=model_name
)
