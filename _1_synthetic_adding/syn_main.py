from net_syn_conv import ConvNet
from net_syn_conv_rf import receptive_field
from net_syn_rnn import RNN
from net_syn_xformers import Transformers

from syn_config import config
from syn_utils import DatasetCreator, count_params, seed_everything, TrainModel

import torch
from torch import nn, optim
import torch_geometric

import numpy as np

import logging
import argparse
parser = argparse.ArgumentParser(description="experiment")
parser.add_argument("--length", type=int, default=128)
parser.add_argument("--model", type=str, default="CDIL")
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Feel free to change the random seed
seed_everything(args.seed)

problem = "adding"  
DATASIZE = 2000

SEQ_LEN = args.length
MODEL = args.model
EPOCH = args.epoch
DATASET = problem + str(SEQ_LEN)

# Parse config
cfg_dataset = config[problem]["dataset"]
cfg_training = config[problem]["training"]
cfg_model = config[problem]["models"][MODEL]

# Setting device
torch.cuda.set_device(cfg_training["device_id"])

# Initialize 
use_CUDA = cfg_dataset["use_cuda"]
INPUT_SIZE = cfg_dataset["in_dim"]
CLASS = cfg_dataset["n_class"]

if MODEL == "CDIL" or MODEL == "TCN":
    LAYER = int(np.log2(SEQ_LEN)) - 1
    NHID = cfg_model["hidden"]
    KERNEL_SIZE = cfg_model["kernel_size"]
    net = ConvNet(MODEL, INPUT_SIZE, CLASS, [NHID] * LAYER, KERNEL_SIZE)
    receptive_field(seq_length=SEQ_LEN, model=MODEL, kernel_size=KERNEL_SIZE, layer=LAYER)
elif MODEL == "LSTM" or MODEL == "GRU":
    LAYER = cfg_model["layer"]
    NHID = cfg_model["hidden"]
    net = RNN(MODEL, INPUT_SIZE, CLASS, NHID, LAYER)
elif MODEL == "Transformer" or MODEL == "Linformer" or MODEL == "Performer":
    DIM = cfg_model["dim"]
    DEPTH = cfg_model["depth"]
    HEADS = cfg_model["heads"]
    net = Transformers(MODEL, INPUT_SIZE, CLASS, SEQ_LEN, DIM, DEPTH, HEADS)
else:
    net = None

if use_CUDA:
    net = net.cuda()

# print(net)
para_num = count_params(net)
if MODEL == "CDIL" or MODEL == "TCN" or MODEL == "LSTM" or MODEL == "GRU":
    file_name = DATASET + "_S" + str(DATASIZE) + "_P" + str(para_num) + "_" + MODEL + "_L" + str(LAYER) + "_H" + str(NHID) + "_E" + str(EPOCH)
else:
    file_name = DATASET + "_S" + str(DATASIZE) + "_P" + str(para_num) + "_" + MODEL + '_D' + str(DIM) + '_H' + str(HEADS) + '_L' + str(DEPTH) + '_E' + str(EPOCH)


log_file_name = "./syn_log/" + file_name + ".txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
loginf = logging.info

loginf(torch.cuda.get_device_name(cfg_training["device_id"]))
loginf(file_name)

optimizer = optim.Adam(
        net.parameters(),
        lr=cfg_training["learning_rate"]
    )

loss = nn.MSELoss()


# Read the data
data = torch.load(f"./syn_datasets/{problem}{DATASIZE}_{SEQ_LEN}_train.pt")
labels = torch.load(f"./syn_datasets/{problem}{DATASIZE}_{SEQ_LEN}_train_target.pt")

data_val = torch.load(f"./syn_datasets/{problem}{DATASIZE}_{SEQ_LEN}_val.pt")
labels_val = torch.load(f"./syn_datasets/{problem}{DATASIZE}_{SEQ_LEN}_val_target.pt")

data_test = torch.load(f"./syn_datasets/{problem}{DATASIZE}_{SEQ_LEN}_test.pt")
labels_test = torch.load(f"./syn_datasets/{problem}{DATASIZE}_{SEQ_LEN}_test_target.pt")


# Prepare the training loader
trainset = DatasetCreator(
    data=data,
    labels=labels
)

trainloader = torch_geometric.data.DataLoader(
    trainset,
    batch_size=cfg_training["batch_size"],
    shuffle=True,
    drop_last=True,
    num_workers=1
)

# Prepare the validation loader
valset = DatasetCreator(
    data=data_val,
    labels=labels_val
)

valloader = torch_geometric.data.DataLoader(
    valset,
    batch_size=cfg_training["batch_size"],
    shuffle=False,
    drop_last=True,
    num_workers=1
)

# Prepare the testing loader
testset = DatasetCreator(
    data=data_test,
    labels=labels_test
)

testloader = torch_geometric.data.DataLoader(
    testset,
    batch_size=cfg_training["batch_size"],
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
    eval_freq=cfg_training["eval_frequency"],
    optimizer=optimizer,
    loss=loss,
    problem=problem,
    saving_best=0,
    loginf=loginf,
)
