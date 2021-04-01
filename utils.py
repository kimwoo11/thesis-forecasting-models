from collections import defaultdict

import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from torch import nn


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y):
        loss = torch.sqrt(self.mse(y_pred, y) + self.eps)
        return loss


class Data:
    def __init__(self, X=None, y=None, agg=None, scaler=None):
        self.X = X
        self.y = y
        self.agg = agg
        self.scaler = scaler


def load_dataset(path_to_dataset):
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


def train_test_split(X, y, test_frac):
    """Creates train/test split
    Args:
        X: np array with shape (num_samples, n_in, num_features)
        y: np array with shape (num_samples, n_out, num_targets)
        test_frac: float indicating percentage of test data
    """
    num_samples = X.shape[0]
    split = int(num_samples * test_frac)

    X_test = X[:split]
    y_test = y[:split]
    X_train = X[split:]
    y_train = y[split:]

    return X_train, y_train, X_test, y_test


def normalize_df(df):
    values = df.values
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # created normalized dataframe
    scaled_df = pd.DataFrame(scaled)
    scaled_df.index = df.index
    scaled_df.columns = df.columns
    return scaled_df
