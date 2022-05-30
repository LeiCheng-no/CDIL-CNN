import os
import torch
import random
import pickle
import numpy as np
from torch.utils.data import Dataset


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class DatasetCreator(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        X = self.data[index]
        Y = self.labels[index]
        return X, Y

    def __len__(self):
        return len(self.labels)


# Refer to https://github.com/mlpen/Nystromformer/blob/main/LRA/code/dataset.py
class LRADataset(Dataset):
    def __init__(self, file_path, endless):

        self.endless = endless
        with open(file_path, 'rb') as f:
            self.examples = pickle.load(f)
            random.shuffle(self.examples)
            self.curr_idx = 0

        print(f'Loaded {file_path}... size={len(self.examples)}', flush=True)

    def __len__(self):
        return len(self.examples)

    def create_inst(self, inst):
        output = {}
        output['input_ids_0'] = torch.tensor(inst['input_ids_0'], dtype=torch.long)
        output['mask_0'] = (output['input_ids_0'] != 0).float()
        if 'input_ids_1' in inst:
            output['input_ids_1'] = torch.tensor(inst['input_ids_1'], dtype=torch.long)
            output['mask_1'] = (output['input_ids_1'] != 0).float()
        output['label'] = torch.tensor(inst['label'], dtype=torch.long)
        return output

    def __getitem__(self, i):
        if not self.endless:
            return self.create_inst(self.examples[i])

        if self.curr_idx >= len(self.examples):
            random.shuffle(self.examples)
            self.curr_idx = 0
        inst = self.examples[self.curr_idx]
        self.curr_idx += 1

        return self.create_inst(inst)
