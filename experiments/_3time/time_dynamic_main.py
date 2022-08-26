import os
import sys
import wandb
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from time_config import config
from time_train import TrainModel
from dynamic import dc_activations

sys.path.append('../')
from Models.net_conv import CONV
from Models.utils import seed_everything, DatasetCreator

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--task', type=str, default='RightWhaleCalls')
# parser.add_argument('--task', type=str, default='FruitFlies')
# parser.add_argument('--task', type=str, default='MosquitoSound')
parser.add_argument('--model', type=str, default='CNN')
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
    wandb.init(project=TASK, name=MODEL + '_dynamic_' + str(SEED), entity="leic-no")
    WANDB = wandb
else:
    WANDB = None

cfg_training = config[TASK]['training']
cfg_model = config[TASK]['models']

BATCH = cfg_training['batch_size']


# Data
data_train = pd.read_csv(f'./time_datasets/{TASK}_train.csv')
labels_train_np = np.array(data_train['label'])
features_train_np = np.array(data_train.drop(['label'], axis=1))
labels_train = torch.tensor(labels_train_np)

data_val = pd.read_csv(f'./time_datasets/{TASK}_val.csv')
labels_val_np = np.array(data_val['label'])
features_val_np = np.array(data_val.drop(['label'], axis=1))
labels_val = torch.tensor(labels_val_np)
features_val = torch.tensor(features_val_np).view(len(labels_val), -1, 1).float()

data_test = pd.read_csv(f'./time_datasets/{TASK}_test.csv')
labels_test_np = np.array(data_test['label'])
features_test_np = np.array(data_test.drop(['label'], axis=1))
labels_test = torch.tensor(labels_test_np)

CLASS = len(torch.unique(labels_train))
SEQ_LEN = features_val.shape[1]
LAYER = int(np.log2(SEQ_LEN))


# Model
if MODEL == 'DIL' or MODEL == 'CNN':
    NHID = cfg_model['cnn_hidden']
    KERNEL_SIZE = cfg_model['cnn_ks']
    net = CONV(TASK, MODEL, INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE)
else:
    print('no this model.')
    sys.exit()

para_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
file_name = TASK + '_P' + str(para_num) + '_' + MODEL + '_S' + str(SEED) + '_L' + str(LAYER) + '_H' + str(NHID)
net.load_state_dict(torch.load('./time_model/' + file_name + '.ph'))


# Dynamic model
dnet = CONV(TASK, MODEL, INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE, dynamic=True)
dnet = dnet.to(device)
para_num = sum(p.numel() for p in dnet.parameters() if p.requires_grad)


# Dynamic log
file_name = TASK + '_P' + str(para_num) + '_' + MODEL + '_dynamic_S' + str(SEED) + '_L' + str(LAYER) + '_H' + str(NHID)
os.makedirs('time_log', exist_ok=True)
log_file_name = './time_log/' + file_name + '.txt'
model_name = './time_model/' + file_name + '.ph'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

loginf(torch.cuda.get_device_name(device))
loginf(file_name)


# Dynamic data
cpu_device = torch.device("cpu")
para = net.conv.conv_net[0].conv.weight_v
convolutional_filters = np.array(para.to(cpu_device).detach().numpy(), dtype=np.float64)

dc_activations_train = dc_activations(features_train_np, convolutional_filters)
trainset = DatasetCreator(dc_activations_train, labels_train)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True, drop_last=False)

dc_activations_val = dc_activations(features_val_np, convolutional_filters)
valset = DatasetCreator(dc_activations_val, labels_val)
valloader = DataLoader(valset, batch_size=BATCH, shuffle=False, drop_last=False)

dc_activations_test = dc_activations(features_test_np, convolutional_filters)
testset = DatasetCreator(dc_activations_test, labels_test)
testloader = DataLoader(testset, batch_size=BATCH, shuffle=False, drop_last=False)


# Optimize
optimizer = torch.optim.Adam(dnet.parameters())
loss = torch.nn.CrossEntropyLoss(reduction='sum')


# train
TrainModel(
    net=dnet,
    device=device,
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    n_epochs=cfg_training['epoch'],
    optimizer=optimizer,
    loss=loss,
    loginf=loginf,
    wandb=WANDB,
    file_name=model_name
)
