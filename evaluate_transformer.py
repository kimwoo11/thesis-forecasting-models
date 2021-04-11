import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from Transformer import *
from config import *
from datasets import CaseUpcTest, CategoryTest
from train_utils import RMSELoss
from attrdict import AttrDict


class EvaluateTransformer(object):
    """Evaluator for multi-step forecasting models
    """
    def __init__(self, model, test_loader, name):
        self.results = {}
        self.model = model
        self.test_loader = test_loader
        self.best_y = None
        self.best_pred = None
        self.name = name
        self.worst_y = None
        self.worst_pred = None
        self.accuracies = None

    def run(self):
        loss_f = RMSELoss()
        losses = []
        accuracies = []
        biases = []

        with torch.no_grad():
            for data in self.test_loader:
                X = data["input"].type(torch.FloatTensor)
                target = data["label"].type(torch.FloatTensor)
                output_window = target.size(1)
                zeros = torch.zeros((X.size(0), output_window, X.size(2)))
                X = torch.cat((X, zeros), 1)       # concat zeros for model
                target = torch.squeeze(target, 0)  # remove batch dim
                pred = self.model(X)
                pred = torch.squeeze(pred, 0)
                pred = pred[-output_window:]
                batch_loss = loss_f(pred, target).item()
                batch_acc = self.calc_accuracy(pred, target).item()  # r2_score(target, pred)
                batch_bias = self.calc_bias(pred, target).item()
                losses.append(batch_loss)
                accuracies.append(batch_acc)
                biases.append(batch_bias)
                curr_best_acc = max(accuracies)
                curr_worst_acc = min(accuracies)
                if batch_acc >= curr_best_acc:
                    self.best_y = target
                    self.best_pred = pred
                if batch_acc <= curr_worst_acc:
                    self.worst_y = target
                    self.worst_pred = pred

        losses = np.array(losses)
        accuracies = np.array(accuracies)
        biases = np.array(biases)
        self.results['avg_loss'] = np.mean(losses)
        self.results['median_acc'] = np.median(accuracies)
        self.results['mean_acc'] = np.mean(accuracies)
        self.results['median_bias'] = np.median(biases)
        self.results['mean_bias'] = np.mean(biases)
        self.results['best_loss'] = np.min(losses)
        self.results['best_acc'] = np.max(accuracies)
        self.results['best_bias'] = np.min(biases)
        self.accuracies = np.array(accuracies)

    def calc_accuracy(self, pred, target):
        target[target == 0] = 0.01
        diff = abs(target[:, 0] - pred[:, 0]) / (target[:, 0])
        return torch.mean(1 - diff)

    def calc_bias(self, pred, target):
        return torch.mean(abs(target[:, 0] - pred[:, 0]) / pred[:, 0])

    def plot(self):
        """Plots highest accuracy forecast vs target
        """
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.plot(self.best_y[:, 0], color='blue', label="Target")
        ax.plot(self.best_pred[:, 0], color='red', label="Prediction")
        ax.legend()
        ax.set_title("ShipmentCases Forecast vs Target for {}".format(self.name))
        ax.grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig("{}_{}_step.png".format(self.name, self.best_y.size(0)))

    def plot_full(self, data):
        """Plot full time series
        """
        df = data.ds
        scaler = data.scaler
        values = df.values
        # normalize features
        scaled = scaler.fit_transform(values)
        # created normalized dataframe
        scaled_df = pd.DataFrame(scaled)
        scaled_df.index = df.index
        scaled_df.columns = df.columns
        df = scaled_df['ShipmentCases'][-150:]

        fig, ax = plt.subplots(figsize=(15, 9))
        t = df.index
        ax.plot(t[-12:], self.best_y[:, 0], color='blue', label="Target")
        ax.plot(t[-12:], self.best_pred[:, 0], color='red', label="Prediction")
        ax.plot(t[:-11], df.iloc[:-11], 'k')
        ax.legend()
        ax.set_title("ShipmentCases Forecast vs Target for {}".format(self.name))
        ax.grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig("{}_{}_step.png".format(self.name, self.best_y.size(0)))

    def plot_hist(self):
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.hist(self.accuracies, bins=30, color='purple')
        ax.set_title("Accuracy Histogram for {}".format(self.name))
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("{}_accuracy_histogram.png".format(self.name))

    def save(self):
        row = list(self.results.values())
        np.savetxt("results/{}_evaluation_results.csv".format(self.name), row, delimiter=',')


def load_model(model_object, name, args):
    model = model_object(args)
    model.load_state_dict(torch.load("models/{}.pth".format(name), map_location=torch.device("cpu")))
    model.train(False)
    model.eval()
    return model


if __name__ == "__main__":
    path_to_data = "data/UnileverShipmentPOS.csv"
    # Transformer_upc
    name = "Transformer_upc"
    features = FEATURES
    targets = TARGETS
    args = {
        'input_size': 50,  # decreasing didn't help
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.0001,
        'wd': 0.0005,
        'bs': 32,
        'epochs': 100
    }
    args = AttrDict(args)
    test_set = CaseUpcTest(path_to_data, args.input_size, args.output_size, features, targets)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    Transformer_upc = load_model(Transformer, name, args)
    Transformer_upc_eval = EvaluateTransformer(Transformer_upc, test_loader, name)
    Transformer_upc_eval.run()
    Transformer_upc_eval.plot()
    Transformer_upc_eval.save()
    Transformer_upc_eval.plot_hist()
    print("{} results: ".format(name), Transformer_upc_eval.results)

    # Transformer_category
    name = "Transformer_category"
    features = FEATURES
    targets = TARGETS
    args = {
        'input_size': 50,  # decreasing didn't help
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.0001,
        'wd': 0.0005,
        'bs': 32,
        'epochs': 200
    }
    args = AttrDict(args)
    test_set = CategoryTest(path_to_data, args.input_size, args.output_size, features, targets)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    Transformer_category = load_model(Transformer, name, args)
    Transformer_category_eval = EvaluateTransformer(Transformer_category, test_loader, name)
    Transformer_category_eval.run()
    Transformer_category_eval.plot()
    Transformer_category_eval.save()
    Transformer_category_eval.plot_hist()
    print("{} results: ".format(name), Transformer_category_eval.results)

    # Transformer_singles
    features = FEATURES
    targets = TARGETS

    args = {
        'input_size': 20,
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.0001,
        'wd': 0.0005,
        'bs': 16,
        'epochs': 200
    }

    args = AttrDict(args)
    dataset = CaseUpc('data/UnileverShipmentPOS.csv', args.input_size, args.output_size, features, targets)

    top_cases = np.load("data/top_cases.npy")
    for case in top_cases:
        data = dataset.upc_to_ts[case]
        test_set = torch.utils.data.Subset(data, list(range(len(data)-1, len(data))))
        model = Transformer(args)
        name = "Transformer_upc_{}".format(case)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        Transformer_upc = load_model(Transformer, name, args)
        Transformer_upc_eval = EvaluateTransformer(Transformer_upc, test_loader, name)
        Transformer_upc_eval.run()
        Transformer_upc_eval.plot_full(data)
        Transformer_upc_eval.save()
        print("{} results: ".format(name), Transformer_upc_eval.results)
