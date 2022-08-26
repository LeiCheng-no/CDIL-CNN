import os
import sys
import wandb
import torch
import logging
import argparse
import numpy as np
from torch.utils.data import DataLoader

from syn_config import config
from dissimilar_train import TrainModel

sys.path.append('../')
from Models.net_conv import CONV
from Models.net_conv_rf import receptive_field
from Models.net_rnn import RNN
from Models.net_xformer import Xformer
from Models.utils import seed_everything, DatasetCreator

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--length', type=int, default=2048)
parser.add_argument('--model', type=str, default='CDIL')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


# Config
use_wandb = False
TASK = 'dissimilar'
INPUT_SIZE = 2
CLASS = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

SEQ_LEN = args.length
MODEL = args.model
SEED = args.seed

seed_everything(args.seed)
if use_wandb:
    wandb.init(project=TASK, name=str(SEQ_LEN) + MODEL + str(SEED), entity="leic-no")
    WANDB = wandb
else:
    WANDB = None

cfg_training = config[TASK]['training']
cfg_model = config[TASK]['models']

BATCH = cfg_training['batch_size']


# Model
if MODEL == 'CDIL' or MODEL == 'DIL' or MODEL == 'TCN' or MODEL == 'CNN':
    LAYER = int(np.log2(SEQ_LEN)) - 1
    NHID = cfg_model['cnn_hidden']
    KERNEL_SIZE = cfg_model['cnn_ks']
    net = CONV(TASK, MODEL, INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == 'Deformable':
    LAYER = int(np.log2(SEQ_LEN)) - 1
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
    net = Xformer(MODEL, INPUT_SIZE, CLASS, DIM, SEQ_LEN, DEPTH, HEADS, use_pos=False)
else:
    print('no this model.')
    sys.exit()

net = net.to(device)
para_num = sum(p.numel() for p in net.parameters() if p.requires_grad)


# Log
if MODEL == 'Transformer' or MODEL == 'Linformer' or MODEL == 'Performer':
    file_name = TASK + str(SEQ_LEN) + '_P' + str(para_num) + '_' + MODEL + '_S' + str(SEED) + '_H' + str(DIM)
else:
    file_name = TASK + str(SEQ_LEN) + '_P' + str(para_num) + '_' + MODEL + '_S' + str(SEED) + '_L' + str(LAYER) + '_H' + str(NHID)

os.makedirs('dissimilar_log', exist_ok=True)
os.makedirs('dissimilar_model', exist_ok=True)
log_file_name = './dissimilar_log/' + file_name + '.txt'
model_name = './dissimilar_model/' + file_name
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

loginf(torch.cuda.get_device_name(device))
loginf(file_name)


# Optimize
optimizer = torch.optim.Adam(net.parameters())
loss = torch.nn.CrossEntropyLoss(reduction='sum')


# Data
data_train, labels_train = torch.load(f'./dissimilar_datasets/{TASK}_{SEQ_LEN}_train.pt')
train_set = DatasetCreator(data_train, labels_train)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, drop_last=False)

data_val, labels_val = torch.load(f'./dissimilar_datasets/{TASK}_{SEQ_LEN}_val.pt')
val_set = DatasetCreator(data_val, labels_val)
val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, drop_last=False)

data_test, labels_test = torch.load(f'./dissimilar_datasets/{TASK}_{SEQ_LEN}_test.pt')
test_set = DatasetCreator(data_test, labels_test)
test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False, drop_last=False)

data_dtest, labels_dtest = torch.load(f'./dissimilar_datasets/{TASK}_{SEQ_LEN}_dtest.pt')
dtest_set = DatasetCreator(data_dtest, labels_dtest)
dtest_loader = DataLoader(dtest_set, batch_size=BATCH, shuffle=False, drop_last=False)


# train
TrainModel(
    net=net,
    device=device,
    trainloader=train_loader,
    valloader=val_loader,
    testloader=test_loader,
    dtestloader=dtest_loader,
    n_epochs=cfg_training['epoch'],
    optimizer=optimizer,
    loss=loss,
    loginf=loginf,
    wandb=WANDB,
    file_name=model_name
)
