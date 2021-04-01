import torch
import torch.nn as nn
import torchsummary
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import math


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TConvBlock(nn.Module):
    # Temporal Convolution block that accepts an input of Lxc_in with a dilation factor of d and performs
    # causal convolution on the input with a kernel size of K to return an output size Lxc_out

    # Note that the look-back length is not necessarily L but is actually the nearest value K*d^i < L for some int i
    def __init__(self, L, c_in, c_out, K, d):
        super(TConvBlock, self).__init__()
        layers = []
        n = math.floor(math.log(L / K, d))
        for i in range(n):
            if i == 0:
                layers += [TConvLayer(c_in, c_out, K, stride=1, dilation=d, padding=(K - 1) * d)]
            else:
                layers += [TConvLayer(c_out, c_out, K, stride=1, dilation=d, padding=(K - 1) * d)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SimpleTCN(nn.Module):
    def __init__(self, lookback, forward_step, num_features):
        super(SimpleTCN, self).__init__()
        L = lookback
        P = forward_step
        K = 8
        d = 2
        self.L = L
        self.P = P
        self.tconv1 = TConvBlock(L, num_features, 32, K, d)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(L, 32, 32, K, d)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(L, 32, 64, K, d)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(L, 64, 64, K, d)
        self.bn4 = torch.nn.BatchNorm1d(64)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(L, 64, 128, K, d)
        self.bn5 = torch.nn.BatchNorm1d(128)
        self.relu5 = torch.nn.ReLU()
        self.tconv6 = TConvBlock(L, 128, 1, K, d)

    def forward(self, input):
        # Assume X: batch by length by channel size
        # print(input.shape)
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = self.relu4(self.bn4(self.tconv4(x3)))
        x5 = self.relu5(self.bn5(self.tconv5(x4)))
        x6 = self.tconv6(x5)
        # print(x.shape)
        return x6[:, :, self.L-self.P:]

