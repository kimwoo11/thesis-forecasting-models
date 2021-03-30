import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class LSTNet(nn.Module):
    def __init__(self, data, args):
        super(LSTNet, self).__init__()
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_features = args.num_features
        self.rnn_hid_size = args.rnn_hid_size
        self.cnn_hid_size = args.cnn_hid_size
        self.skip_hid_size = args.skip_hid_size
        self.kernel_size = args.kernel_size
        self.skip = args.skip
        self.pt = (self.input_size - self.kernel_size) / self.skip
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
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.input_size, self.num_features)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
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
