import pandas as pd
from scipy.io import arff

# dir = 'FruitFlies'
# name = 'FruitFlies'
# label_name = 'target'
# labels = ['zaprionus', 'suzukii', 'melanogaster']

# dir = 'RightWhaleCalls/WhaleCalls'
# name = 'RightWhaleCalls'
# label_name = 'class'
# labels = ['RightWhale', 'NoWhale']

dir = 'MosquitoSound'
name = 'MosquitoSound'
label_name = 'target'
labels = ['Aedes_aegypti', 'Culex_quinquefasciatus', 'Anopheles_gambiae', 'Culex_pipiens', 'Aedes_albopictus', 'Anopheles_arabiensis']


data_train, meta_train = arff.loadarff('./time_datasets/' + dir + '/' + name + '_TRAIN.arff')
df_train = pd.DataFrame(data_train)
print(df_train)
print(df_train[label_name].value_counts())


df = df_train.rename(columns={label_name: 'label'})
df['label'] = df['label'].str.decode("utf-8")
for i, one_label in enumerate(labels):
    df.loc[df.label == one_label, 'label'] = int(i)
print(df['label'].value_counts())
# print(df['label'].value_counts(normalize=True))

df.to_csv('./time_datasets/' + name + '_train.csv', index=False)


data_test, meta_test = arff.loadarff('./time_datasets/' + dir + '/' + name + '_TEST.arff')
df_test = pd.DataFrame(data_test)

df = df_test.rename(columns={label_name: 'label'})
df['label'] = df['label'].str.decode("utf-8")
for i, one_label in enumerate(labels):
    df.loc[df.label == one_label, 'label'] = int(i)
print(df['label'].value_counts())
# print(df['label'].value_counts(normalize=True))

df.to_csv('./time_datasets/' + name + '_test.csv', index=False)


# value_counts
# df_train = pd.read_csv('./time_datasets/' + name + '_train.csv')
# print(df_train['label'].value_counts(normalize=True))
# df_test = pd.read_csv('./time_datasets/' + name + '_test.csv')
# print(df_test['label'].value_counts(normalize=True))
