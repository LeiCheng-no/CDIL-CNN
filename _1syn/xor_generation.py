import os
import torch
import random
import argparse
from tqdm import tqdm


def xor(num, seq_len):
    full = []
    labels = []
    for _ in tqdm(range(num)):

        x = torch.rand(seq_len)
        y = torch.zeros(seq_len)
        pos1, pos2 = list(random.sample(range(seq_len), 2))
        y[pos1] = 1
        y[pos2] = 1
        X1, X2 = torch.rand(2)
        x[pos1] = X1
        x[pos2] = X2
        one = torch.vstack([x, y]).T
        full.append(one)

        if 0 <= X1 < 0.5:
            if 0 <= X2 < 0.5:
                label = 0
            else:
                label = 1
        else:
            if 0 <= X2 < 0.5:
                label = 1
            else:
                label = 0

        labels.append(label)
    print(sum(labels)/num)
    sequences = torch.vstack(full).reshape(num, seq_len, 2)
    labels = torch.tensor(labels)
    return sequences, labels


def main(seq_len):
    datasets = [
        ['train', 10000],
        ['val', 10000],
        ['test', 10000]
    ]
    print(f'Generation sequences for length={seq_len}')
    for dataset in datasets:
        name, n_seq = dataset
        sequences, labels = xor(n_seq, seq_len)
        torch.save((sequences, labels), f'./xor_datasets/xor_{seq_len}_{name}.pt')


if __name__ == '__main__':
    os.makedirs('./xor_datasets/', exist_ok=True)

    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('--length', type=int, default=16)
    args = parser.parse_args()

    main(args.length)
