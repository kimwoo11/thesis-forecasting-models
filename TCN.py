import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from attrdict import AttrDict
import math
from utils import *
from config import *
from data_loader import *


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
    def __init__(self, args):
        super(SimpleTCN, self).__init__()
        L = args.input_size     # look back
        P = args.output_size    # prediction horizon
        self.num_targets = args.num_targets
        K = 8
        d = 2
        self.L = L
        self.P = P
        self.tconv1 = TConvBlock(L, args.num_features, 32, K, d)
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
        self.tconv6 = TConvBlock(L, 128, self.num_targets, K, d)

    def forward(self, input):
        # Assume X: batch by length by channel size
        # print(input.shape)
        input = torch.transpose(input, 1, 2)
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = self.relu4(self.bn4(self.tconv4(x3)))
        x5 = self.relu5(self.bn5(self.tconv5(x4)))
        x6 = self.tconv6(x5)
        # print(x.shape)
        x7 = x6[:, :, self.L-self.P:]
        # print(x7.size())
        return torch.transpose(x7, 1, 2)

def load_data(data_class, device, path_to_data, input_size, output_size, features, targets, bs):
    dataset = data_class(path_to_data, input_size, output_size, features, targets)
    test_len = int(len(dataset) * 0.20)
    test_set = torch.utils.data.Subset(dataset, list(range(0, test_len)))
    tv_set = torch.utils.data.Subset(dataset, list(range(test_len, len(dataset))))

    train_len = int(len(tv_set) * 0.9)
    val_len = len(tv_set) - train_len

    train_set, val_set = torch.utils.data.random_split(tv_set, [train_len, val_len])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader


def train_tcn(args, features, targets, data_class, path_to_data, name="TCN", grid_search=False):
    print(name)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if not grid_search:
            print("Using GPU")
    else:
        device = torch.device("cpu")
        if not grid_search:
            print("Using CPU")

    model = SimpleTCN(args).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr,
                                 weight_decay=args.wd)

    loss = RMSELoss()

    train_loss = []
    val_loss = []
    best_loss = 0
    best_epoch = 0

    train_loader, val_loader, test_loader = load_data(data_class, device, path_to_data, args.input_size,
                                                      args.output_size,
                                                      features, targets, args.bs)
    train_len = len(train_loader)
    val_len = len(val_loader)

    for epoch in range(args.epochs):
        model.train(True)
        epoch_train_loss = 0

        # Training
        for data in train_loader:
            x = data["input"].type(torch.FloatTensor).to(device)  # Load Input data
            label = data["label"].type(torch.FloatTensor).to(device)  # Load labels
            optimizer.zero_grad()  # Reset gradients
            pred = model(x)  # Forward Pass
            batch_loss = loss(pred, label)  # Compute loss
            epoch_train_loss += batch_loss.item() / train_len

            batch_loss.backward()  # Backpropagation
            optimizer.step()  # Optimization

        train_loss.append(epoch_train_loss)

        # Validation
        model.train(False)
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                x = data["input"].type(torch.FloatTensor).to(device)  # Load Input data
                label = data["label"].type(torch.FloatTensor).to(device)  # Load labels
                optimizer.zero_grad()  # Reset gradients
                pred = model(x)  # Forward Pass
                batch_loss = loss(pred, label)  # Compute loss
                epoch_val_loss += batch_loss.item() / val_len
            val_loss.append(epoch_val_loss)

            if best_epoch == 0 or epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), "{}.pth".format(name))

        print("Epoch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f}".format(epoch, train_loss[epoch],
                                                                                val_loss[epoch]))

    print("Training Complete")
    print("Best Validation Error ({}) at epoch {}".format(best_loss, best_epoch))

    # Plot Final Training Errors
    fig, ax = plt.subplots(figsize=(15,9))
    plt.grid(True)
    ax.plot(train_loss, linewidth=2, label="Training Loss")
    ax.plot(val_loss, linewidth=2, label="Validation Loss")
    ax.set_title("{} Training & Validation Losses".format(name))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.savefig("figures/{}.png".format(name))
    fig.show()
    print("Finished Training!")
    return train_loader, val_loader, test_loader


def test_loss(model, test_loader):
    loss_f = RMSELoss()
    mean_test_loss = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for data in test_loader:
            X = data["input"].type(torch.FloatTensor)  # Load Input data
            label = data["label"].type(torch.FloatTensor)  # Load labels
            pred = model(X)
            mean_test_loss += loss_f(pred, label).item() / test_len

    return mean_test_loss


def main(args, features, targets, data_class, path_to_data, name="TCN"):
    args = AttrDict(args)
    _, _, test_loader = train_tcn(args, features, targets, data_class, path_to_data, name=name)

    best_model = SimpleTCN(args)
    best_model.load_state_dict(torch.load("{}.pth".format(name), map_location=torch.device("cpu")))
    best_model.train(False)
    best_model.eval()

    test_l = test_loss(best_model, test_loader)
    print("Best Model's Test Loss: {}".format(test_l))
    return best_model


def plot_multi_step(model, X, y, name, targets, forecast_steps=1):
    preds = list()
    with torch.no_grad():
        for i in range(0, len(X), forecast_steps):
            curr_x = torch.unsqueeze(torch.tensor(X[i]).type(torch.FloatTensor), 0)  # curr input to model
            pred = model(curr_x)
            preds.append(np.squeeze(pred.cpu().numpy()))

    preds = np.array(preds)
    nrows=len(targets)
    if nrows > 1:
        fig, ax = plt.subplots(nrows=nrows, figsize=(12,12))
        fig.suptitle("{} 12 Step Forecasts on Sample Time Series".format(name), fontsize=16)
        for i in range(nrows):
            ax[i].plot(preds[-1, :, i], label="Forecast")
            ax[i].plot(y[-1, :, i], label="Target")
            ax[i].legend()
            ax[i].set_title("{} Forecast vs Target".format(targets[i]))
            ax[i].grid(True)
    else:
        fig, ax = plt.subplots(nrows=nrows, figsize=(12,9))
        fig.suptitle("{} 12 Step Forecasts on Sample Time Series".format(name), fontsize=16)
        ax.plot(preds[-1, :], label="Forecast")
        ax.plot(y[-1, :], label="Target")
        ax.legend()
        ax.set_title("{} Forecast vs Target".format(targets[0]))
        ax.grid(True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("figures/{}_{}_step.png".format(name, forecast_steps))


def best_multi_step(model, ts, keys):
    """ Returns the best performing time series and its loss, as well as the total mean loss of the given dataset
    Args:
        model: trained model to evaluated
        ts: a time series dataset dictionary; either CaseUpc.case_to_ts, or Category.cat_to_ts
        keys: case_to_ts.keys(), or cat_to_ts.keys()
        forecast_steps: number of forecast steps we want to evaluate the model on
    """
    loss_f = RMSELoss()
    best_loss = float('inf')
    best_X, best_y = None, None
    total_loss = list() # across all datasets
    for key in keys:
        curr_loss = 0
        with torch.no_grad():
            curr_dataset = ts[key]
            curr_loader = torch.utils.data.DataLoader(curr_dataset, shuffle=True, num_workers=0)
            data_len = len(curr_dataset)
            for data in curr_loader:
                X = data["input"].type(torch.FloatTensor)
                label = data["label"].type(torch.FloatTensor)  # Load labels
                pred = model(X)
                curr_loss += loss_f(pred, label).item() / data_len

        total_loss.append(curr_loss)
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_X, best_y = curr_dataset.X, curr_dataset.y

    return best_X, best_y, best_loss, np.mean(total_loss)
