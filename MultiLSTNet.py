import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from attrdict import AttrDict
from config import *
from data_loader import CaseUpc, Category
from train_utils import train, load_data


class MultiLSTNet(nn.Module):
    def __init__(self, args):
        super(MultiLSTNet, self).__init__()
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_features = args.num_features
        self.num_targets = args.num_targets
        self.rnn_hid_size = args.rnn_hid_size
        self.cnn_hid_size = args.cnn_hid_size
        self.kernel_size = args.kernel_size
        self.highway_size = args.highway_size
        self.conv1 = nn.Conv2d(1, self.cnn_hid_size, kernel_size=(self.kernel_size, self.num_features))
        self.LSTM1 = nn.LSTM(self.cnn_hid_size, self.rnn_hid_size)
        self.dropout = nn.Dropout(p=args.dropout)
        self.LSTM2 = nn.LSTM(self.rnn_hid_size, self.num_targets)

        if self.highway_size > 0:
            self.highway = nn.Linear(self.highway_size, self.output_size)
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        if args.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN encoder
        c = x.view(-1, 1, self.input_size, self.num_features)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # lstm encoder
        r = c.permute(2, 0, 1).contiguous()
        _, (r, _) = self.LSTM1(r)
        r = self.dropout(r)
        r = r.repeat(self.output_size, 1, 1)

        # lstm decoder
        res, _ = self.LSTM2(r)
        res = res.permute(1, 0, 2).contiguous()

        # highway
        if self.highway_size > 0:
            z = x[:, -self.highway_size:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.highway_size)
            z = self.highway(z)
            z = z.view(-1, self.output_size, self.num_targets)
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
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'rnn_hid_size': 100,
        'cnn_hid_size': 100,
        'kernel_size': 4,
        'highway_size': 0,  # set to zero for single target
        'dropout': 0.1,
        'output_fun': 'sigmoid',
        'lr': 0.0001,
        'wd': 0.0005,
        'bs': 1000,
        'epochs': 200
    }

    args = AttrDict(args)
    name = "MultiLSTNet_upc"
    train_loader, val_loader = load_data(CaseUpc, path_to_data, args.input_size, args.output_size, features, targets,
                                         args.bs)
    model = MultiLSTNet(args)
    train(args, model, train_loader, val_loader, name)

    # Category
    args = {
        'input_size': 20,
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'rnn_hid_size': 100,
        'cnn_hid_size': 100,
        'kernel_size': 4,
        'highway_size': 0,  # set to zero for single target
        'dropout': 0.1,
        'output_fun': 'sigmoid',
        'lr': 0.001,
        'wd': 0.0005,
        'bs': 16,
        'epochs': 500
    }

    args = AttrDict(args)
    name = "MultiLSTNet_category"

    train_loader, val_loader = load_data(Category, path_to_data, args.input_size, args.output_size, features,
                                         targets, args.bs)
    model = MultiLSTNet(args)
    train(args, model, train_loader, val_loader, name)


if __name__ == "__main__":
    run("data/UnileverShipmentPOS.csv")
