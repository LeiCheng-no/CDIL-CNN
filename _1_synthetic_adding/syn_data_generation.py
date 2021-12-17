# Adding Problem

import random
import torch
from tqdm import tqdm


def adding(sequences, n_data):
    full = []
    labels = []
    for _ in tqdm(range(sequences)):
        x = (-1 - 1) * torch.rand(n_data) + 1
        y = torch.zeros(n_data)
        pos_1 = pos_2 = -1
        while pos_1 == pos_2:
            samples = list(random.sample(range(n_data), 2))
            samples.sort()
            pos_1, pos_2 = samples

        y[pos_1] = y[pos_2] = 1
        data = torch.vstack([x, y]).T
        full.append(data)
        label = 0.5 + (x[pos_1] + x[pos_2]) / 4
        labels.append(label)

    data = torch.vstack(full).reshape(sequences, n_data, 2)
    labels = torch.tensor(labels)
    return data, labels


def main():
    size = 2000
    n_sequences = {
        'train': size,
        'val': size,
        'test': size
    }
    # seq_lenths = [2**7, 2**8]
    seq_lenths = [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14]
    for seq_len in seq_lenths:
        print(f"Generation sequences for length={seq_len}")
        for ds, n_seq in n_sequences.items():
            data, labels = adding(n_seq, seq_len)
            torch.save(data, f'./syn_datasets/adding{size}_{seq_len}_{ds}.pt')
            torch.save(labels, f'./syn_datasets/adding{size}_{seq_len}_{ds}_target.pt')


if __name__ == "__main__":
    main()
