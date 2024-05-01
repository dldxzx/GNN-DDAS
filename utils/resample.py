import sys
sys.path.append("/home/zengxin/fpk/pycharm_project/GNN-DDAS")
from sklearn.utils import resample
import pandas as pd
from torch_geometric.data import Dataset
from utils.graph_dataset import SMILESDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from os.path import join
import os
def resampled(temp_train_root=None,raw_train_root=None,ratio=0.0):
    temp_train_root = temp_train_root
    df = pd.read_csv(raw_train_root)
    class_num = df['label'].value_counts()
    positive_num = class_num.min()
    temp_data = []
    for label, group in df.groupby('label'):
        if label == 1:
            resampled_group = resample(group, replace=False, n_samples=positive_num, random_state=42)
        else:
            resampled_group = resample(group, replace=False, n_samples=positive_num*ratio, random_state=42)
        temp_data.append(resampled_group)
    temp_df = pd.concat(temp_data)
    # temp_root = os.path.join(temp_train_root,'raw')
    # os.makedirs(temp_root, exist_ok=True)
    temp_df.to_csv(join(temp_train_root,'raw/resampled.csv'),index=False)
    temp_data = SMILESDataset(root=temp_train_root, raw_dataset='resampled.csv', processed_data='resampled_train.pt')
    return temp_data
