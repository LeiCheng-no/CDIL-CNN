from net_lra_conv import ConvNet
from net_lra_conv_rf import receptive_field

from net_lra_rnn import RNN

from lra_config import config
from lra_utils import count_params, seed_everything, Net, NetDual, TrainModel
from dataset import LRADataset

import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import logging
import argparse
parser = argparse.ArgumentParser(description='experiment')
# dataset
parser.add_argument('--problem', type=str, default='image')
# parser.add_argument('--problem', type=str, default='pathfinder32')
# parser.add_argument('--problem', type=str, default='text_4000')
# parser.add_argument('--problem', type=str, default='retrieval_4000')
# model
parser.add_argument('--model', type=str, default='CDIL')
parser.add_argument('--hidden_cnn', type=int, default=50)
parser.add_argument('--layer_rnn', type=int, default=1)
parser.add_argument('--hidden_rnn', type=int, default=128)
args = parser.parse_args()

problem = args.problem

# Feel free to change the random seed
seed_everything(42)

# Parse config
cfg_model = config[problem]['models']
cfg_training = config[problem]['training']


# Setting device
torch.cuda.set_device(cfg_training["device_id"])


# Initialize
EPOCH = cfg_training['num_train_steps']
BATCH = cfg_training['batch_size']
MODEL = args.model
INPUT_SIZE = cfg_model["dim"]
CLASS = cfg_model["n_class"]
SEQ_LEN = cfg_model['n_length']
use_EMBED = cfg_model["use_embedding"]
CHAR_COCAB = cfg_model["vocab_size"]
FIX_length = cfg_model["fix_length"]

KERNEL_SIZE = 3

if MODEL == 'CDIL' or MODEL == 'TCN':
    LAYER = cfg_model["layer_cnn"]
    NHID = args.hidden_cnn
    net_part = ConvNet(MODEL, INPUT_SIZE, [NHID] * LAYER, KERNEL_SIZE, use_EMBED, CHAR_COCAB, FIX_length)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == 'LSTM' or MODEL == 'GRU':
    LAYER = args.layer_rnn
    NHID = args.hidden_rnn
    net_part = RNN(MODEL, INPUT_SIZE, args.hidden_rnn, args.layer_rnn, use_EMBED, CHAR_COCAB, FIX_length)
else:
    print('no this model.')
    sys.exit()

if problem == 'retrieval_4000':
    net = NetDual(net_part, NHID, CLASS)
else:
    net = Net(net_part, NHID, CLASS)

if cfg_model["use_cuda"]:
    net = net.cuda()

para_num = count_params(net)
file_name = problem + '_P' + str(para_num) + '_' + MODEL + '_L' + str(LAYER) + '_EM' + str(INPUT_SIZE) + '_H' + str(NHID) + '_E' + str(EPOCH)

log_file_name = './lra_log/' + file_name + '.txt'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

loginf(torch.cuda.get_device_name(cfg_training["device_id"]))
loginf(file_name)

optimizer = optim.Adam(
        net.parameters(),
        lr=cfg_training['learning_rate']
    )

loss = nn.CrossEntropyLoss()

# Prepare loaders
trainloader = DataLoader(LRADataset(f"./lra_datasets/{problem}.train.pickle", True), batch_size=BATCH, drop_last=True, shuffle=True)
valloader = DataLoader(LRADataset(f"./lra_datasets/{problem}.dev.pickle", True), batch_size=BATCH, drop_last=True)
testloader = DataLoader(LRADataset(f"./lra_datasets/{problem}.test.pickle", False), batch_size=BATCH, drop_last=True)


TrainModel(
    net=net,
    fix_length=FIX_length,
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    n_epochs=EPOCH,
    eval_freq=cfg_training['eval_frequency'],
    optimizer=optimizer,
    loss=loss,
    problem=problem,
    length=SEQ_LEN,
    saving_best=0,
    loginf=loginf,
)
