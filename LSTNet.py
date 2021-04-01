import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt

from data_loader import *
from utils import RMSELoss
from attrdict import AttrDict

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


def train_lstnet(args, features, targets, data_class, path_to_data, name="LSTNet", grid_search=False):
    print(name)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if not grid_search:
            print("Using GPU")
    else:
        device = torch.device("cpu")
        if not grid_search:
            print("Using CPU")

    model = LSTNet(args).to(device)
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
            label = torch.squeeze(data["label"].type(torch.FloatTensor).to(device))  # Load labels

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
                label = torch.squeeze(data["label"].type(torch.FloatTensor).to(device))  # Load labels
                optimizer.zero_grad()  # Reset gradients
                pred = model(x)  # Forward Pass
                batch_loss = loss(pred, label)  # Compute loss
                epoch_val_loss += batch_loss.item() / val_len
            val_loss.append(epoch_val_loss)
            if not grid_search:
                if best_epoch == 0 or epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), "{}.pth".format(name))
        if not grid_search:
            print("Epoch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f}".format(epoch, train_loss[epoch],
                                                                                val_loss[epoch]))
        # else:
        #     tune.report(loss=val_loss[epoch])

    if not grid_search:
        print("Training Complete")
        print("Best Validation Error ({}) at epoch {}".format(best_loss, best_epoch))

        # Plot Final Training Errors
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.plot(train_loss, linewidth=2, label="Training Loss")
        ax.plot(val_loss, linewidth=2, label="Validation Loss")
        ax.set_title("{} Training & Validation Losses".format(name))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True)
        fig.savefig("figures/{}.png".format(name))
        fig.show()
        print("Finished Training!")
    return train_loader, val_loader, test_loader


# def run_tune(path_to_data, features, targets, max_num_epochs):
#     num_features = len(features)
#     args = {
#         'input_size': tune.choice([4, 8, 12, 16, 20, 50, 100]),
#         'output_size': 1,
#         'num_features': num_features,
#         'rnn_hid_size': tune.choice([50, 100, 200]),
#         'cnn_hid_size': tune.choice([50, 100, 200]),
#         'skip_hid_size': tune.choice([5, 20, 50, 100]),
#         'kernel_size': 4,
#         'skip': tune.choice([4, 8, 12]),
#         'highway_size': 4,
#         'dropout': 0.2,
#         'output_fun': 'sigmoid',
#         'lr': tune.choice([0.00001, 0.0001, 0.001, 0.01]),
#         'wd': tune.choice([0.00005, 0.0005, 0.005]),
#         'bs': tune.choice([16, 50, 100, 200, 300, 500]),
#         'epochs': max_num_epochs
#     }
#     args = AttrDict(args)
#
#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=max_num_epochs)
#     reporter = CLIReporter(
#         # parameter_columns=["input_size", "rnn_hid_size", "cnn_hid_size", "skip_hid_size", "skip", "lr", "wd", "bs"],
#         metric_columns=["loss"])
#     result = tune.run(
#         partial(train_lstnet, features=features, targets=targets, data_class=CaseUpc, path_to_data=path_to_data,
#                 name="LSTNet", grid_search=True),
#         resources_per_trial={"cpu": 1, "gpu": 1},
#         config=args,
#         num_samples=1,
#         scheduler=scheduler,
#         progress_reporter=reporter)
#
#     return result


def test_loss(model, test_loader):
    loss_f = RMSELoss()
    mean_test_loss = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for data in test_loader:
            X = data["input"].type(torch.FloatTensor)  # Load Input data
            label = torch.squeeze(data["label"].type(torch.FloatTensor))  # Load labels
            pred = model(X)
            mean_test_loss += loss_f(pred, label).item() / test_len

    return mean_test_loss


def plot_multi_step(model, X, y, output_size, name, forecast_steps=1):
    preds = list()
    targets = y[:, 0]
    with torch.no_grad():
        for i in range(0, len(X), forecast_steps):
            curr_x = torch.unsqueeze(torch.tensor(X[i]).type(torch.FloatTensor), 0)  # curr input to model
            for j in range(forecast_steps):
                if i+j >= len(X):
                    break
                pred = model(curr_x)
                curr_x = torch.cat((curr_x[:, 1:], torch.unsqueeze(pred, 0)), 1)
                preds.append(np.squeeze(pred.cpu().numpy()))

    preds = np.array(preds)
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.suptitle("{} {} Step Forecasts on Sample Time Series".format(name, forecast_steps), fontsize=16)
    ax.plot(preds[-output_size:, 0], label="preds")
    ax.plot(targets[-output_size:, 0], label="targets")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/{}_{}_step.png".format(name, forecast_steps))


def best_multi_step(model, ts, keys, forecast_steps=1):
    """ Returns the best performing time series and its loss, as well as the total mean loss of the given dataset
    Args:
        model: trained model to evaluated
        ts: a time series dataset; either CaseUpc.case_to_ts, or Category.cat_to_ts
        keys: case_to_ts.keys(), or cat_to_ts.keys()
        forecast_steps: number of forecast steps we want to evaluate the model on
    """
    loss_f = RMSELoss()
    best_loss = float('inf')
    best_X, best_y = None, None
    total_loss = list()
    for key in keys:
        curr_loss = 0
        X, y = ts[key].X, ts[key].y
        with torch.no_grad():
            for i in range(0, len(X), forecast_steps):
                curr_x = torch.unsqueeze(torch.tensor(X[i]).type(torch.FloatTensor), 0)
                for j in range(forecast_steps):
                    if i+j >= len(X):
                        break
                    pred = model(curr_x)
                    label = torch.tensor(y[i+j]).type(torch.FloatTensor)
                    curr_x = torch.cat((curr_x[:, 1:], torch.unsqueeze(pred, 0)), 1)
                    curr_loss += loss_f(pred, label).item() / len(X)
        total_loss.append(curr_loss)
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_X, best_y = X, y
    return best_X, best_y, best_loss, np.mean(total_loss)


def main(args, features, targets, data_class, path_to_data, name="LSTNet"):
    args = AttrDict(args)
    _, _, test_loader = train_lstnet(args, features, targets, data_class, path_to_data, name=name)

    best_model = LSTNet(args)
    best_model.load_state_dict(torch.load("{}.pth".format(name), map_location=torch.device("cpu")))
    best_model.train(False)
    best_model.eval()

    test_l = test_loss(best_model, test_loader)
    print("Best Model's Test Loss: {}".format(test_l))
    return best_model


if __name__ == "__main__":
    # Cases
    args = {
        'input_size': 50,  # decreasing didn't help
        'output_size': 1,
        'num_features': len(FEATURES),
        'rnn_hid_size': 200,  # increasing helped
        'cnn_hid_size': 200,
        'skip_hid_size': 4,
        'kernel_size': 4,
        'skip': 4,
        'highway_size': 4,
        'dropout': 0.1,
        'output_fun': 'sigmoid',
        'lr': 0.0001,
        'wd': 0.0005,
        'bs': 1000,
        'epochs': 100
    }

    args = AttrDict(args)
    name = "LSTNet_cases"

    best_case_model = main(args, FEATURES, FEATURES, CaseUpc, "UnileverShipmentPOS.csv", name)

    # Categories
    args = {
        'input_size': 50,  # decreasing didn't help
        'output_size': 1,
        'num_features': len(FEATURES),
        'rnn_hid_size': 200,  # increasing helped
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
    name = "LSTNet_categories"

    best_category_model = main(args, FEATURES, FEATURES, Category, "UnileverShipmentPOS.csv", name)