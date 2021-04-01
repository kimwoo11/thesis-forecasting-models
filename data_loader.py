from collections import defaultdict
from config import *
from utils import load_dataset, series_preparation, Data
from torch.utils.data import Dataset

import torch
import numpy as np


class CaseUpc(Dataset):
    def __init__(self, path_to_dataset, n_in, n_out, features=None, targets=None):
        print("Loading CaseUPC Dataset")
        df = load_dataset(path_to_dataset)
        self.cases = df.CASE_UPC_CD.unique()
        self.upc_to_ts = defaultdict(Single)  # upc to time series mapping
        self.targets = ['ShipmentCases'] if targets is None else targets
        self.n_in = n_in  # number of input time steps
        self.n_out = n_out  # forecasting time horizon
        self.features = features
        self.X = None  # (num_samples, n_in, num_features)
        self.y = None  # (num_samples, n_out, num_targets)
        self.agg = None

        for case in self.cases:
            ds = df[df.CASE_UPC_CD == case][features].dropna()
            if ds.shape[0] > 100:
                X, y, agg, scaler = series_preparation(ds, targets, n_in, n_out)
                self.upc_to_ts[case].X = X
                self.upc_to_ts[case].y = y
                self.upc_to_ts[case].agg = agg
                self.upc_to_ts[case].scaler = scaler

                if self.X is None:
                    self.X = X
                else:
                    self.X = np.concatenate((self.X, X), axis=0)

                if self.y is None:
                    self.y = y
                else:
                    self.y = np.concatenate((self.y, y), axis=0)

                if self.agg is None:
                    self.agg = agg
                else:
                    self.agg = self.agg.append(agg, ignore_index=True)
        print("Successfully loaded the CaseUPC dataset")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': self.X[idx, :, :], 'label': self.y[idx, :, :]}

        return sample


class Category(Dataset):
    def __init__(self, path_to_dataset, n_in, n_out, features=None, targets=None):
        print("Loading Category Dataset")
        df = load_dataset(path_to_dataset)
        self.categories = df.CategoryDesc.unique()
        self.category_to_ts = defaultdict(Single)  # upc to time series mapping
        self.targets = ['ShipmentCases'] if targets is None else targets
        self.n_in = n_in  # number of input time steps
        self.n_out = n_out  # forecasting time horizon
        self.features = features
        self.X = None  # (num_samples, n_in, num_features)
        self.y = None  # (num_samples, n_out, num_targets)
        self.agg = None

        for cat in self.categories:
            ds = df[df.CategoryDesc == cat][features].dropna().groupby('WeekNumber').sum()
            if ds.shape[0] > 100:
                X, y, agg, scaler = series_preparation(ds, targets, n_in, n_out)
                self.category_to_ts[cat].X = X
                self.category_to_ts[cat].y = y
                self.category_to_ts[cat].agg = agg
                self.category_to_ts[cat].scaler = scaler

                if self.X is None:
                    self.X = X
                else:
                    self.X = np.concatenate((self.X, X), axis=0)

                if self.y is None:
                    self.y = y
                else:
                    self.y = np.concatenate((self.y, y), axis=0)

                if self.agg is None:
                    self.agg = agg
                else:
                    self.agg = self.agg.append(agg, ignore_index=True)
        print("Successfully loaded the Category dataset")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': self.X[idx, :, :], 'label': self.y[idx, :, :]}

        return sample


class Single(Dataset):
    def __init__(self, X=None, y=None, agg=None, scaler=None):
        self.X = X
        self.y = y
        self.agg = agg
        self.scaler = scaler

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': self.X[idx, :, :], 'label': self.y[idx, :, :]}

        return sample


if __name__ == "__main__":
    dataset = Category("data/UnileverShipmentPOS.csv", 2, 2, FEATURES, TARGETS)
    bs = 1
    test_len = int(len(dataset) * 0.15)
    test_set = torch.utils.data.Subset(dataset, list(range(0, test_len)))
    tv_set = torch.utils.data.Subset(dataset, list(range(test_len, len(dataset))))

    train_len = int(len(tv_set) * 0.8)
    val_len = len(tv_set) - train_len
    train_set, val_set = torch.utils.data.random_split(tv_set, [train_len, val_len])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=0)

    train_sample = train_set[24]
    val_sample = val_set[42]
    test_sample = test_set[76]
    print("Data Loading Successful")
