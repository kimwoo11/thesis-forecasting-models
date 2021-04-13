from collections import defaultdict
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset

import torch
import numpy as np


class CaseUpc(Dataset):
    def __init__(self, path_to_dataset, n_in, n_out, features=None, targets=None):
        print("Loading CaseUPC Dataset")
        df = load_csv(path_to_dataset)
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
            if ds.shape[0] < 100:
                continue
            X, y, agg, scaler = series_preparation(ds, targets, n_in, n_out)
            self.upc_to_ts[case].X = X
            self.upc_to_ts[case].y = y
            self.upc_to_ts[case].agg = agg
            self.upc_to_ts[case].scaler = scaler
            self.upc_to_ts[case].ds = ds

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


class CaseUpcTV(Dataset):
    def __init__(self, path_to_dataset, n_in, n_out, features=None, targets=None):
        print("Loading CaseUPC Dataset")
        df = load_csv(path_to_dataset)
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
            if ds.shape[0] < 200:
                continue
            X, y, agg, scaler = series_preparation(ds, targets, n_in, n_out)
            X = X[:-n_in]
            y = y[:-n_in]
            agg = agg.iloc[:-n_in]
            ds = ds.iloc[:-n_in]
            self.upc_to_ts[case].X = X
            self.upc_to_ts[case].y = y
            self.upc_to_ts[case].agg = agg
            self.upc_to_ts[case].scaler = scaler
            self.upc_to_ts[case].ds = ds

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


class CaseUpcTest(Dataset):
    def __init__(self, path_to_dataset, n_in, n_out, features=None, targets=None):
        print("Loading CaseUPC Dataset")
        df = load_csv(path_to_dataset)
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
            if ds.shape[0] < 200:
                continue
            X, y, agg, scaler = series_preparation(ds, targets, n_in, n_out)
            X = np.expand_dims(X[-1], axis=0)
            y = np.expand_dims(y[-1], axis=0)
            agg = agg.iloc[-1]

            self.upc_to_ts[case].X = X
            self.upc_to_ts[case].y = y
            self.upc_to_ts[case].agg = agg
            self.upc_to_ts[case].scaler = scaler
            self.upc_to_ts[case].ds = ds

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
        df = load_csv(path_to_dataset)
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
            if ds.shape[0] < 100:
                continue
            X, y, agg, scaler = series_preparation(ds, targets, n_in, n_out)
            self.category_to_ts[cat].X = X
            self.category_to_ts[cat].y = y
            self.category_to_ts[cat].agg = agg
            self.category_to_ts[cat].scaler = scaler
            self.category_to_ts[cat].ds = ds

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


class CategoryTV(Dataset):
    def __init__(self, path_to_dataset, n_in, n_out, features=None, targets=None):
        print("Loading Category Dataset")
        df = load_csv(path_to_dataset)
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
            if ds.shape[0] < 200:
                continue
            X, y, agg, scaler = series_preparation(ds, targets, n_in, n_out)
            X = X[:-n_in]
            y = y[:-n_in]
            agg = agg.iloc[:-n_in]
            ds = ds.iloc[:-n_in]

            self.category_to_ts[cat].X = X
            self.category_to_ts[cat].y = y
            self.category_to_ts[cat].agg = agg
            self.category_to_ts[cat].scaler = scaler
            self.category_to_ts[cat].ds = ds

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


class CategoryTest(Dataset):
    def __init__(self, path_to_dataset, n_in, n_out, features=None, targets=None):
        print("Loading Category Dataset")
        df = load_csv(path_to_dataset)
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
            if ds.shape[0] < 200:
                continue
            X, y, agg, scaler = series_preparation(ds, targets, n_in, n_out)
            X = np.expand_dims(X[-1], axis=0)
            y = np.expand_dims(y[-1], axis=0)
            agg = agg.iloc[-1]

            self.category_to_ts[cat].X = X
            self.category_to_ts[cat].y = y
            self.category_to_ts[cat].agg = agg
            self.category_to_ts[cat].scaler = scaler
            self.category_to_ts[cat].ds = ds

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
    def __init__(self, X=None, y=None, agg=None, scaler=None, ds=None):
        self.X = X
        self.y = y
        self.agg = agg
        self.ds = ds
        self.scaler = scaler

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': self.X[idx, :, :], 'label': self.y[idx, :, :]}

        return sample


def load_csv(path_to_dataset):
    df = pd.read_csv(path_to_dataset, sep='|', engine='python')
    return format_unilever_data(df)


def series_preparation(df, targets, n_in=1, n_out=1, dropnan=True, normalize=True):
    """ Takes in a dataframe and prepares it for a time series forecasting model
    Args:
        df: Dataframe that contains the features and columns and is indexed by
            datetime.
        targets: A list of targets that the model is predicting
        n_in: Number of lagged timesteps that the model takes in
        n_out: Prediction horizion; number of time steps model predicts
        dropnan: If True, drops NaN values
        normalize: If True, normalizes data across features
    Returns:
        X: numpy array of size (num_samples, n_in, num_features)
        y: numpy array of size (num_samples, n_out, num_targets)
        agg: aggregated X and y.
            df of size (num_samples, n_in*num_features+n_out*num_targets)
    """
    scaler = None
    if normalize:
        values = df.values
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # created normalized dataframe
        scaled_df = pd.DataFrame(scaled)
        scaled_df.index = df.index
        scaled_df.columns = df.columns
        df = scaled_df

    col_names = list(df.columns)
    cols, names = list(), list()

    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('{}(t-{})'.format(col_name, i)) for col_name in col_names]

    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        target_df = df[targets]
        cols.append(target_df.shift(-i))
        if i == 0:
            names += [('{}(t)'.format(col_name)) for col_name in targets]
        else:
            names += [('{}(t+{})'.format(col_name, i)) for col_name in targets]

    # create dataframe
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)

    # separate X and y
    num_targets = len(targets)
    X = agg.iloc[:, :-(num_targets * n_out)]
    y = agg.iloc[:, -(num_targets * n_out):]

    # reshape X and y
    num_samples = agg.shape[0]
    num_feats = len(col_names)
    X = np.reshape(np.ravel(X.values), (num_samples, n_in, num_feats))
    y = np.reshape(np.ravel(y.values), (num_samples, n_out, num_targets))

    return X, y, agg, scaler


def format_unilever_data(df):
    """ Formats Unilever data to contain a time axis
    Args:
        df: dataframe from UnileverShipmentPos.csv
    """
    # parse WeekNumber to create time axis
    # - since the given format is not sufficient to create a datetime, we add
    #   '-1' and "%w" so that the parser can default to the Monday of the given week
    dt = []
    for d in df.WeekNumber:
        dt.append(datetime.strptime(str(d) + '-1', "%Y%W-%w"))

    # assign datetime to df & set week number as index
    df.WeekNumber = dt

    # Pick out columnns & Sort data by time
    df = df.sort_values('WeekNumber')
    df = df.set_index(df.WeekNumber)

    return df
