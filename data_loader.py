import os, sys
p = os.path.abspath('..')
sys.path.insert(1,p)

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

class FDS_Dataset(Dataset):
    def __init__(self, data, labels):

        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(np.array(labels)).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx]
        return data, label

def Make_Dataset(sample_num, ratio):

    # Load Basic dataset
    file_list = os.listdir('../Data/train')
    df = pd.DataFrame()

    print()
    print(' ############################ ')
    print()
    print(' #### Collecting Dataset #### ')
    print()
    print(' ############################ ')
    print()

    for each in tqdm(file_list[:-2]):

        tmp_df = pd.read_pickle('../Data/train/' + each)

        df = pd.concat([df,tmp_df])

    df = df.fillna(-1)

    # Remove duplicated part
    dp_list = df['dt'].duplicated()
    df = df[~dp_list]

    # Drop dt and custNo
    df = df.drop(['dt','custNo'],axis=1)
    df = df.sample(sample_num)

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Train, Valid Split
    X_train, X_valid, y_train, y_valid = train_test_split(scaled_data, np.array([0]*len(scaled_data)), test_size=ratio)

    return X_train, X_valid, y_train, y_valid

def get_loader(sample_num = 500000, ratio=.2,batch_size=256):

    print()
    print(' ##### Make Loader ##### ')

    X_train, X_valid, y_train, y_valid = Make_Dataset(sample_num, ratio)

    train_loader = DataLoader(
        dataset=FDS_Dataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        dataset=FDS_Dataset(X_valid, y_valid),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, valid_loader


if __name__ == '__main__':

    train_loader, valid_loader = get_loader()

    # for checking
    for each in train_loader:
        print(each[0].shape)
        print(each[1].shape)
