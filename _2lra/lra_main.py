import os
import sys
import wandb
import torch
import logging
import argparse
from torch.utils.data import DataLoader

from lra_config import config
from lra_train import NetDual, TrainModel

sys.path.append('../')
from Models.net_conv import CONV
from Models.net_conv_rf import receptive_field
from Models.net_rnn import RNN
from Models.utils import seed_everything, LRADataset

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--task', type=str, default='image')
# parser.add_argument('--task', type=str, default='pathfinder32')
# parser.add_argument('--task', type=str, default='text_4000')
# parser.add_argument('--task', type=str, default='retrieval_4000')
parser.add_argument('--model', type=str, default='CDIL')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


# Config
use_wandb = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TASK = args.task
MODEL = args.model
SEED = args.seed

seed_everything(args.seed)
if use_wandb:
    wandb.init(project=TASK, name=MODEL + str(SEED), entity="leic-no")
    WANDB = wandb
else:
    WANDB = None

cfg_training = config[TASK]['training']
cfg_model = config[TASK]['models']

BATCH = cfg_training['batch_size']
CLASS = cfg_model['n_class']
SEQ_LEN = cfg_model['n_length']
FIX_length = cfg_model['fix_length']
USE_EMBED = cfg_model['use_embedding']
CHAR_COCAB = cfg_model['vocab_size']
INPUT_SIZE = cfg_model['dim']


# Model
if MODEL == 'CDIL' or MODEL == 'DIL' or MODEL == 'TCN' or MODEL == 'CNN':
    LAYER = cfg_model['cnn_layer']
    NHID = cfg_model['cnn_hidden']
    KERNEL_SIZE = cfg_model['cnn_ks']
    net_part = CONV(TASK, MODEL, INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE, False, False, USE_EMBED, CHAR_COCAB, FIX_length)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == 'Deformable':
    LAYER = cfg_model['cnn_layer']
    NHID = cfg_model['cnn_hidden']
    KERNEL_SIZE = cfg_model['cnn_ks']
    net_part = CONV(TASK, 'CNN', INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE, True, False, USE_EMBED, CHAR_COCAB, FIX_length)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == 'LSTM' or MODEL == 'GRU':
    LAYER = cfg_model['rnn_layer']
    NHID = cfg_model['rnn_hidden']
    net_part = RNN(TASK, MODEL, INPUT_SIZE, CLASS, NHID, LAYER, USE_EMBED, CHAR_COCAB, FIX_length)
else:
    print('no this model.')
    sys.exit()

if TASK == 'retrieval_4000':
    net = NetDual(net_part, NHID, CLASS)
else:
    net = net_part

net = net.to(device)
para_num = sum(p.numel() for p in net.parameters() if p.requires_grad)


# Log
file_name = TASK + '_P' + str(para_num) + '_' + MODEL + '_S' + str(SEED) + '_L' + str(LAYER) + '_H' + str(NHID)

os.makedirs('lra_log', exist_ok=True)
os.makedirs('lra_model', exist_ok=True)
log_file_name = './lra_log/' + file_name + '.txt'
model_name = './lra_model/' + file_name + '.ph'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

loginf(torch.cuda.get_device_name(device))
loginf(file_name)


# Optimize
optimizer = torch.optim.Adam(net.parameters())
loss = torch.nn.CrossEntropyLoss(reduction='sum')


# Data
trainloader = DataLoader(LRADataset(f'./lra_datasets/{TASK}.train.pickle', True), batch_size=BATCH, shuffle=True, drop_last=False)
valloader = DataLoader(LRADataset(f'./lra_datasets/{TASK}.dev.pickle', True), batch_size=BATCH, shuffle=False, drop_last=False)
testloader = DataLoader(LRADataset(f'./lra_datasets/{TASK}.test.pickle', False), batch_size=BATCH, shuffle=False, drop_last=False)


# train
TrainModel(
    task=TASK,
    fix_length=FIX_length,
    net=net,
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
