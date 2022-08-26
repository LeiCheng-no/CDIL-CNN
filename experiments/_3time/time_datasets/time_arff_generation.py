import sys
import argparse
import pandas as pd
from scipy.io import arff

parser = argparse.ArgumentParser(description='experiment')
parser.add_argument('--task', type=str, default='RightWhaleCalls')
# parser.add_argument('--task', type=str, default='FruitFlies')
# parser.add_argument('--task', type=str, default='MosquitoSound')
args = parser.parse_args()

name = args.task

if name == 'RightWhaleCalls':
    path = 'RightWhaleCalls/WhaleCalls'
    label_name = 'class'
    labels = ['RightWhale', 'NoWhale']
elif name == 'FruitFlies':
    path = 'FruitFlies'
    label_name = 'target'
    labels = ['zaprionus', 'suzukii', 'melanogaster']
elif name == 'MosquitoSound':
    path = 'MosquitoSound'
    label_name = 'target'
    labels = ['Aedes_aegypti', 'Culex_quinquefasciatus', 'Anopheles_gambiae', 'Culex_pipiens', 'Aedes_albopictus', 'Anopheles_arabiensis']
else:
    print('no this task.')
    sys.exit()


data_train, meta_train = arff.loadarff(path + '/' + name + '_TRAIN.arff')
df_train = pd.DataFrame(data_train)
print(df_train[label_name].value_counts())

df = df_train.rename(columns={label_name: 'label'})
df['label'] = df['label'].str.decode("utf-8")
for i, one_label in enumerate(labels):
    df.loc[df.label == one_label, 'label'] = int(i)
print(df['label'].value_counts())
# print(df['label'].value_counts(normalize=True))

print(len(df))
df_train_7 = df.sample(frac=0.7, random_state=1)
print(len(df_train_7))
df_val_3 = df[~df.index.isin(df_train_7.index)]
print(len(df_val_3))
print(len(df_train_7) + len(df_val_3))

df_train_7.to_csv(name + '_train.csv', index=False)
df_val_3.to_csv(name + '_val.csv', index=False)


data_test, meta_test = arff.loadarff(path + '/' + name + '_TEST.arff')
df_test = pd.DataFrame(data_test)

df = df_test.rename(columns={label_name: 'label'})
df['label'] = df['label'].str.decode("utf-8")
for i, one_label in enumerate(labels):
    df.loc[df.label == one_label, 'label'] = int(i)
print(df['label'].value_counts())
# print(df['label'].value_counts(normalize=True))

print(len(df))
df.to_csv(name + '_test.csv', index=False)
