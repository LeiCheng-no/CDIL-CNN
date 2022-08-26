import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--task', type=str, default='RightWhaleCalls')
args = parser.parse_args()


def new_dataset(name, features, labels, end=True, length=2000):
    new_features = []
    for feature, label in tqdm(zip(features, labels), total=len(labels)):
        mu = np.mean(feature)
        sigma = np.std(feature)
        noise = np.random.normal(mu, sigma, length)
        if end:
            new_feature = np.concatenate((feature, noise))
        else:
            new_feature = np.concatenate((noise, feature))
        new_features.append(new_feature)
    new_features = np.array(new_features)
    np.savetxt(name + '_data.csv', new_features, delimiter=',', fmt='%f')
    np.savetxt(name + '_label.csv', labels, delimiter=',', fmt='%d')
    return new_features


TASK = args.task

data_train = pd.read_csv(f'./time_datasets/{TASK}_train.csv')
labels_train = np.array(data_train['label'])
features_train = np.array(data_train.drop(['label'], axis=1))

data_val = pd.read_csv(f'./time_datasets/{TASK}_val.csv')
labels_val = np.array(data_val['label'])
features_val = np.array(data_val.drop(['label'], axis=1))

data_test = pd.read_csv(f'./time_datasets/{TASK}_test.csv')
labels_test = np.array(data_test['label'])
features_test = np.array(data_test.drop(['label'], axis=1))


path = './noise_datasets/'
os.makedirs(path, exist_ok=True)
new_dataset(path + TASK + '_train', features_train, labels_train, end=True)
new_dataset(path + TASK + '_val', features_val, labels_val, end=True)
new_dataset(path + TASK + '_test', features_test, labels_test, end=True)
new_dataset(path + TASK + '_dtest', features_test, labels_test, end=False)
