import os
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from torch import nn, optim
from sklearn.model_selection import train_test_split

from net_time_conv import ConvNet
from net_time_conv_rf import receptive_field
from net_time_rnn import RNN
from net_time_xformers import TransformerHead, LinformerHead, PerformerHead

from time_utils import DatasetCreator, count_params, seed_everything, TrainModel


parser = argparse.ArgumentParser(description='experiment')
# dataset
parser.add_argument('--problem', type=str, default='RightWhaleCalls')
# parser.add_argument('--problem', type=str, default='FruitFlies')
# parser.add_argument('--problem', type=str, default='MosquitoSound')
parser.add_argument('--epoch', type=int, default=100)
# model
parser.add_argument('--model', type=str, default='CDIL')
parser.add_argument('--hidden_cnn', type=int, default=32)
parser.add_argument('--layer_rnn', type=int, default=1)
parser.add_argument('--hidden_rnn', type=int, default=128)
parser.add_argument('--depth_former', type=int, default=4)
parser.add_argument('--head_former', type=int, default=4)
parser.add_argument('--dim_former', type=int, default=32)
args = parser.parse_args()

problem = args.problem
MODEL = args.model
DATASET = problem
EPOCH = args.epoch
DEPTH = args.depth_former
HEAD = args.head_former
BATCH_SIZE = 64
DIM = args.dim_former

# Feel free to change the random seed
seed_everything(42)

# Read the data
data_train = pd.read_csv(f'./time_datasets/{problem}_train.csv')

labels = np.array(data_train['label'])
labels = torch.tensor(labels)
features = np.array(data_train.drop(['label'], axis=1))
features = torch.tensor(features).view(len(labels), -1, 1).float()

CLASS = len(np.unique(labels))
SEQ_LEN = features.shape[1]

features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.3, random_state=1)

data_test = pd.read_csv(f'./time_datasets/{problem}_test.csv')
labels_test = np.array(data_test['label'])
labels_test = torch.tensor(labels_test)
features_test = np.array(data_test.drop(['label'], axis=1))
features_test = torch.tensor(features_test).view(len(labels_test), -1, 1).float()


# Setting device
torch.cuda.set_device(0)

# Initialize 
INPUT_SIZE = 1
use_CUDA = True

KERNEL_SIZE = 3

if MODEL == 'CDIL' or MODEL == 'TCN' or MODEL == 'CNN':
    LAYER = int(np.log2(SEQ_LEN))
    NHID = args.hidden_cnn
    net = ConvNet(MODEL, INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == 'LSTM' or MODEL == 'GRU':
    LAYER = args.layer_rnn
    NHID = args.hidden_rnn
    net = RNN(MODEL, INPUT_SIZE, CLASS, NHID, LAYER)
elif MODEL == "Transformer":
    net = TransformerHead(dim=DIM, heads=HEAD, depth=DEPTH, n_vec=SEQ_LEN, n_class=CLASS)
elif MODEL == "Linformer":
    net = LinformerHead(dim=DIM, heads=HEAD, depth=DEPTH, n_vec=SEQ_LEN, n_class=CLASS)
elif MODEL == "Performer":
    net = PerformerHead(dim=DIM, heads=HEAD, depth=DEPTH, n_vec=SEQ_LEN, n_class=CLASS)
else:
    net = None
# print(net)

para_num = count_params(net)
if MODEL == 'CDIL' or MODEL == 'TCN' or MODEL == 'CNN' or MODEL == 'LSTM' or MODEL == 'GRU':
    file_name = DATASET + '_P' + str(para_num) + '_' + MODEL + '_L' + str(LAYER) + '_H' + str(NHID) + '_E' + str(EPOCH)
else:
    file_name = DATASET + '_P' + str(para_num) + '_' + MODEL + '_D' + str(DIM) + '_H' + str(HEAD) + '_D' + str(DEPTH) +'_E' + str(EPOCH)

os.makedirs("time_log", exist_ok=True)
log_file_name = './time_log/' + file_name + '.txt'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
loginf = logging.info

loginf(torch.cuda.get_device_name(0))
loginf(file_name)

optimizer = optim.Adam(
        net.parameters(),
        lr=0.001
    )

loss = nn.CrossEntropyLoss()

if use_CUDA:
    net = net.cuda()

# Prepare the training loader
trainset = DatasetCreator(
    data=features_train,
    labels=labels_train
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=1
)

# Prepare the validation loader
valset = DatasetCreator(
    data=features_val,
    labels=labels_val
)

valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    num_workers=1
)

# Prepare the testing loader
testset = DatasetCreator(
    data=features_test,
    labels=labels_test
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    num_workers=1
)


TrainModel(
    net=net,
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    n_epochs=EPOCH,
    eval_freq=1,
    optimizer=optimizer,
    loss=loss,
    saving_best=0,
    loginf=loginf,
)

