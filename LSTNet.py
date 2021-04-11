import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

from attrdict import AttrDict
from config import *
from datasets import CaseUpc, Category, CaseUpcTV, CategoryTV
from train_utils import train, load_data


# import matplotlib.pyplot as plt

# from data_loader import *
# from utils import RMSELoss

# Import when using ray tune
# import torch
# from functools import partial
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler


class LSTNet(nn.Module):
    def __init__(self, args):
        super(LSTNet, self).__init__()
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_features = args.num_features
        self.rnn_hid_size = args.rnn_hid_size
        self.cnn_hid_size = args.cnn_hid_size
        self.skip_hid_size = args.skip_hid_size
        self.kernel_size = args.kernel_size
        self.skip = args.skip
        self.pt = (self.input_size - self.kernel_size) // self.skip
        self.highway_size = args.highway_size
        self.conv1 = nn.Conv2d(1, self.cnn_hid_size, kernel_size=(self.kernel_size, self.num_features))
        self.GRU1 = nn.GRU(self.cnn_hid_size, self.rnn_hid_size)
        self.dropout = nn.Dropout(p=args.dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.cnn_hid_size, self.skip_hid_size)
            self.linear1 = nn.Linear(self.rnn_hid_size + self.skip * self.skip_hid_size, self.num_features)
        else:
            self.linear1 = nn.Linear(self.rnn_hid_size, self.num_features)
        if self.highway_size > 0:
            self.highway = nn.Linear(self.highway_size, 1)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        if args.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.input_size, self.num_features)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        temp, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.cnn_hid_size, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.cnn_hid_size)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.skip_hid_size)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.highway_size > 0:
            z = x[:, -self.highway_size:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.highway_size)
            z = self.highway(z)
            z = z.view(-1, self.num_features)
            res = res + z

        if self.output:
            res = self.output(res)
        return res


def run(path_to_data):
    features = FEATURES
    targets = FEATURES

    # CaseUpc
    args = {
        'input_size': 50,
        'output_size': 1,
        'num_features': len(features),
        'rnn_hid_size': 200,
        'cnn_hid_size': 200,
        'skip_hid_size': 4,
        'kernel_size': 4,
        'skip': 4,
        'highway_size': 4,
        'dropout': 0.1,
        'output_fun': 'sigmoid',
        'lr': 0.0001,  # for training
        'wd': 0.0005,  # for training
        'epochs': 100,  # for training
        'bs': 1000,
    }

    args = AttrDict(args)
    name = "LSTNet_upc"
    train_loader, val_loader = load_data(CaseUpcTV, path_to_data, args.input_size, args.output_size, features, targets,
                                         args.bs)
    model = LSTNet(args)
    train(args, model, train_loader, val_loader, name)

    # Category
    args = {
        'input_size': 50,
        'output_size': 1,
        'num_features': len(features),
        'rnn_hid_size': 200,
        'cnn_hid_size': 200,
        'skip_hid_size': 4,
        'kernel_size': 4,
        'skip': 4,
        'highway_size': 4,
        'dropout': 0.1,
        'output_fun': 'sigmoid',
        'lr': 0.0001,
        'wd': 0.0005,
        'bs': 100,
        'epochs': 500
    }

    args = AttrDict(args)
    name = "LSTNet_category"

    train_loader, val_loader = load_data(CategoryTV, path_to_data, args.input_size, args.output_size, features,
                                         targets, args.bs)
    model = LSTNet(args)
    train(args, model, train_loader, val_loader, name)


def run_single(path_to_data):
    features = FEATURES
    targets = FEATURES

    # CaseUpc
    args = {
        'input_size': 50,
        'output_size': 1,
        'num_features': len(features),
        'rnn_hid_size': 200,
        'cnn_hid_size': 200,
        'skip_hid_size': 4,
        'kernel_size': 4,
        'skip': 4,
        'highway_size': 4,
        'dropout': 0.1,
        'output_fun': 'sigmoid',
        'lr': 0.0001,  # for training
        'wd': 0.0005,  # for training
        'epochs': 200,  # for training
        'bs': 16,
    }

    args = AttrDict(args)
    dataset = CaseUpc(path_to_data, args.input_size, args.output_size, features, targets)

    top_cases = np.load("data/top_cases.npy")
    for case in top_cases:
        data = dataset.upc_to_ts[case]

        train_loader, val_loader = load_data(data, path_to_data, args.input_size, args.output_size, features,
                                             targets, args.bs, single=True)
        model = LSTNet(args)
        name = "LSTNet_upc_{}".format(case)
        train(args, model, train_loader, val_loader, name)

    # Category
    args = {
        'input_size': 50,
        'output_size': 1,
        'num_features': len(features),
        'rnn_hid_size': 200,
        'cnn_hid_size': 200,
        'skip_hid_size': 4,
        'kernel_size': 4,
        'skip': 4,
        'highway_size': 4,
        'dropout': 0.1,
        'output_fun': 'sigmoid',
        'lr': 0.0001,
        'wd': 0.0005,
        'bs': 16,
        'epochs': 200
    }

    args = AttrDict(args)
    dataset = Category(path_to_data, args.input_size, args.output_size, features, targets)

    top_categories = np.load("data/top_categories.npy")

    for cat in top_categories:
        data = dataset.category_to_ts[cat]

        train_loader, val_loader = load_data(data, path_to_data, args.input_size, args.output_size, features,
                                             targets, args.bs, single=True)
        model = LSTNet(args)
        name = "LSTNet_category_{}".format(cat)
        train(args, model, train_loader, val_loader, name)


if __name__ == "__main__":
    run("data/UnileverShipmentPOS.csv")
