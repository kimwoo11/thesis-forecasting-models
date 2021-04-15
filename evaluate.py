import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from TCN import *
from MultiLSTNet import *
from LSTNet import *
from datasets import CaseUpcTest, CategoryTest
from train_utils import RMSELoss


class EvaluateMultiStep(object):
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
                target = torch.squeeze(data["label"].type(torch.FloatTensor), 0)
                pred = self.model(X)
                pred = torch.squeeze(pred, 0)
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
        self.results['median_loss'] = np.median(losses)
        self.results['median_acc'] = np.median(accuracies)
        self.results['median_bias'] = np.median(biases)
        self.results['avg_loss'] = np.mean(losses)
        self.results['mean_acc'] = np.mean(accuracies)
        self.results['mean_bias'] = np.mean(biases)
        self.results['best_loss'] = np.min(losses)
        self.results['best_acc'] = np.max(accuracies)
        self.results['best_bias'] = np.min(biases)
        self.accuracies = np.array(accuracies)

    def run_rolling(self):
        loss_f = RMSELoss()
        losses = []
        accuracies = []
        biases = []

        with torch.no_grad():
            for data in self.test_loader:
                X = data["input"].type(torch.FloatTensor)  # (1, 20, 4)
                target = torch.squeeze(data["label"].type(torch.FloatTensor), 0)
                preds = torch.tensor([])
                for _ in range(target.size(0)):
                    pred = self.model(X)  # (1, 4)
                    pred = torch.unsqueeze(pred, 0)
                    X = torch.cat((X[:, 1:], pred), 1)
                    preds = torch.cat((preds, pred), 1)
                preds = torch.squeeze(preds, 0)
                batch_loss = loss_f(preds, target).item()
                batch_acc = self.calc_accuracy(preds, target).item()  # r2_score(target, pred)
                batch_bias = self.calc_bias(preds, target).item()
                losses.append(batch_loss)
                accuracies.append(batch_acc)
                biases.append(batch_bias)
                curr_best_acc = max(accuracies)
                curr_worst_acc = min(accuracies)
                if batch_acc >= curr_best_acc:
                    self.best_y = target
                    self.best_pred = preds
                if batch_acc <= curr_worst_acc:
                    self.worst_y = target
                    self.worst_pred = preds

        losses = np.array(losses)
        accuracies = np.array(accuracies)
        biases = np.array(biases)
        self.results['median_loss'] = np.median(losses)
        self.results['median_acc'] = np.median(accuracies)
        self.results['median_bias'] = np.median(biases)
        self.results['avg_loss'] = np.mean(losses)
        self.results['mean_acc'] = np.mean(accuracies)
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
        ax.plot(self.best_y[:, 0], color="blue", label="Target")
        ax.plot(self.best_pred[:, 0], color="red", label="Prediction")
        ax.legend()
        ax.set_title("ShipmentCases Forecast vs Target for {}".format(self.name))
        ax.grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig("{}_{}_step.png".format(self.name, self.best_y.size(0)))
        plt.close(fig)

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
        ax.plot(t[:-12], df.iloc[:-12], 'k')
        ax.plot([t[-13], t[-12]], [df.iloc[-13], self.best_y[0, 0]], color='blue')  # connect to previous sequence
        ax.plot([t[-13], t[-12]], [df.iloc[-13], self.best_pred[0, 0]], color='red')
        ax.legend()
        ax.set_title("ShipmentCases Forecast vs Target for {}".format(self.name))
        ax.grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig("{}_{}_step.png".format(self.name, self.best_y.size(0)))
        plt.close(fig)

    def plot_hist(self):
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.hist(self.accuracies, bins=30, color="purple")
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


def evaluate(path_to_data):
    # LSTNet_upc
    name = "LSTNet_upc"
    features = FEATURES
    targets = FEATURES

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
        'bs': 1000,
    }
    args = AttrDict(args)
    test_set = CaseUpcTest(path_to_data, args.input_size, 12, features, targets)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    LSTNet_upc = load_model(LSTNet, name, args)
    LSTNet_upc_eval = EvaluateMultiStep(LSTNet_upc, test_loader, name)
    LSTNet_upc_eval.run_rolling()
    LSTNet_upc_eval.plot()
    LSTNet_upc_eval.save()
    LSTNet_upc_eval.plot_hist()
    print("{} results: ".format(name), LSTNet_upc_eval.results)

    # LSTnet_category
    name = "LSTNet_category"
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
        'bs': 100,
    }

    args = AttrDict(args)
    test_set = CategoryTest(path_to_data, args.input_size, 12, features, targets)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    LSTNet_category = load_model(LSTNet, name, args)
    LSTNet_category_eval = EvaluateMultiStep(LSTNet_category, test_loader, name)
    LSTNet_category_eval.run_rolling()
    LSTNet_category_eval.plot()
    LSTNet_category_eval.save()
    LSTNet_category_eval.plot_hist()
    print("{} results: ".format(name), LSTNet_category_eval.results)

    # TCN_upc
    name = "TCN_upc"
    features = FEATURES
    targets = TARGETS
    args = {
        'input_size': 20,
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.001,
        'wd': 0.0005,
        'bs': 1000,
        'epochs': 250
    }
    args = AttrDict(args)
    test_set = CaseUpcTest(path_to_data, args.input_size, args.output_size, features, targets)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    TCN_upc = load_model(TCN, name, args)
    TCN_upc_eval = EvaluateMultiStep(TCN_upc, test_loader, name)
    TCN_upc_eval.run()
    TCN_upc_eval.plot()
    TCN_upc_eval.save()
    TCN_upc_eval.plot_hist()
    print("{} results: ".format(name), TCN_upc_eval.results)

    # TCN_category
    name = "TCN_category"
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
        'epochs': 400
    }
    args = AttrDict(args)
    test_set = CategoryTest(path_to_data, args.input_size, args.output_size, features, targets)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    TCN_category = load_model(TCN, name, args)
    TCN_category_eval = EvaluateMultiStep(TCN_category, test_loader, name)
    TCN_category_eval.run()
    TCN_category_eval.plot()
    TCN_category_eval.save()
    TCN_category_eval.plot_hist()
    print("{} results: ".format(name), TCN_category_eval.results)

    # MultiLSTNet_upc
    name = "MultiLSTNet_upc"
    features = FEATURES
    targets = FEATURES

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
    test_set = CaseUpcTest(path_to_data, args.input_size, args.output_size, features, targets)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    MultiLSTNet_upc = load_model(MultiLSTNet, name, args)
    MultiLSTNet_upc_eval = EvaluateMultiStep(MultiLSTNet_upc, test_loader, name)
    MultiLSTNet_upc_eval.run()
    MultiLSTNet_upc_eval.plot()
    MultiLSTNet_upc_eval.save()
    MultiLSTNet_upc_eval.plot_hist()
    print("{} results: ".format(name), MultiLSTNet_upc_eval.results)

    # MultiLSTNet_category
    name = "MultiLSTNet_category"
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
    test_set = CategoryTest(path_to_data, args.input_size, args.output_size, features, targets)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    MultiLSTNet_category = load_model(MultiLSTNet, name, args)
    MultiLSTNet_category_eval = EvaluateMultiStep(MultiLSTNet_category, test_loader, name)
    MultiLSTNet_category_eval.run()
    MultiLSTNet_category_eval.plot()
    MultiLSTNet_category_eval.save()
    MultiLSTNet_category_eval.plot_hist()
    print("{} results: ".format(name), MultiLSTNet_category_eval.results)


def evaluate_singles(path_to_data):
    # LSTNet_upc_single
    features = FEATURES
    targets = FEATURES

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
        'bs': 1000,
    }
    args = AttrDict(args)
    dataset = CaseUpcTest(path_to_data, args.input_size, 12, features, targets)

    top_cases = np.load("data/top_cases.npy")
    for case in top_cases:
        test_set = dataset.upc_to_ts[case]
        name = "LSTNet_upc_{}".format(case)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        LSTNet_upc = load_model(LSTNet, name, args)
        LSTNet_upc_eval = EvaluateMultiStep(LSTNet_upc, test_loader, name)
        LSTNet_upc_eval.run_rolling()
        LSTNet_upc_eval.plot_full(test_set)
        LSTNet_upc_eval.save()
        print("{} results: ".format(name), LSTNet_upc_eval.results)

    # LSTnet_category_single
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
        'bs': 100,
    }

    args = AttrDict(args)
    dataset = CategoryTest(path_to_data, args.input_size, 12, features, targets)

    top_categories = np.load("data/top_categories.npy")
    for cat in top_categories:
        test_set = dataset.category_to_ts[cat]
        name = "LSTNet_category_{}".format(cat)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        LSTNet_category = load_model(LSTNet, name, args)
        LSTNet_category_eval = EvaluateMultiStep(LSTNet_category, test_loader, name)
        LSTNet_category_eval.run_rolling()
        LSTNet_category_eval.plot_full(test_set)
        LSTNet_category_eval.save()
        print("{} results: ".format(name), LSTNet_category_eval.results)

    # MultiLSTNet_upc
    features = FEATURES
    targets = FEATURES

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
    dataset = CaseUpcTest(path_to_data, args.input_size, args.output_size, features, targets)

    top_cases = np.load("data/top_cases.npy")
    for case in top_cases:
        test_set = dataset.upc_to_ts[case]
        name = "MultiLSTNet_upc_{}".format(case)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        MultiLSTNet_upc = load_model(MultiLSTNet, name, args)
        MultiLSTNet_upc_eval = EvaluateMultiStep(MultiLSTNet_upc, test_loader, name)
        MultiLSTNet_upc_eval.run()
        MultiLSTNet_upc_eval.plot_full(test_set)
        MultiLSTNet_upc_eval.save()
        print("{} results: ".format(name), MultiLSTNet_upc_eval.results)

    # MultiLSTNet_category
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
    dataset = CategoryTest(path_to_data, args.input_size, args.output_size, features, targets)

    top_categories = np.load("data/top_categories.npy")
    for cat in top_categories:
        test_set = dataset.category_to_ts[cat]
        name = "MultiLSTNet_category_{}".format(cat)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        MultiLSTNet_category = load_model(MultiLSTNet, name, args)
        MultiLSTNet_category_eval = EvaluateMultiStep(MultiLSTNet_category, test_loader, name)
        MultiLSTNet_category_eval.run()
        MultiLSTNet_category_eval.plot_full(test_set)
        MultiLSTNet_category_eval.save()
        print("{} results: ".format(name), MultiLSTNet_category_eval.results)

    # TCN_upc
    features = FEATURES
    targets = TARGETS
    args = {
        'input_size': 20,
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.001,
        'wd': 0.0005,
        'bs': 1000,
        'epochs': 200
    }
    args = AttrDict(args)
    dataset = CaseUpcTest(path_to_data, args.input_size, args.output_size, features, targets)

    top_cases = np.load("data/top_cases.npy")
    for case in top_cases:
        test_set = dataset.upc_to_ts[case]
        name = "TCN_upc_{}".format(case)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        TCN_upc = load_model(TCN, name, args)
        TCN_upc_eval = EvaluateMultiStep(TCN_upc, test_loader, name)
        TCN_upc_eval.run()
        TCN_upc_eval.plot_full(test_set)
        TCN_upc_eval.save()
        print("{} results: ".format(name), TCN_upc_eval.results)

    # TCN_category
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
    dataset = CategoryTest(path_to_data, args.input_size, args.output_size, features, targets)

    top_categories = np.load("data/top_categories.npy")
    for cat in top_categories:
        test_set = dataset.category_to_ts[cat]
        name = "TCN_category_{}".format(cat)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        TCN_category = load_model(TCN, name, args)
        TCN_category_eval = EvaluateMultiStep(TCN_category, test_loader, name)
        TCN_category_eval.run()
        TCN_category_eval.plot_full(test_set)
        TCN_category_eval.save()
        print("{} results: ".format(name), TCN_category_eval.results)


def evaluate_full_on_singles(path_to_data):
    # LSTNet_upc_single
    features = FEATURES
    targets = FEATURES

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
        'bs': 1000,
    }
    args = AttrDict(args)
    dataset = CaseUpcTest(path_to_data, args.input_size, 12, features, targets)

    top_cases = np.load("data/top_cases.npy")
    for case in top_cases:
        test_set = dataset.upc_to_ts[case]
        name = "LSTNet_upc_on_{}".format(case)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        LSTNet_upc = load_model(LSTNet, "LSTNet_upc", args)
        LSTNet_upc_eval = EvaluateMultiStep(LSTNet_upc, test_loader, name)
        LSTNet_upc_eval.run_rolling()
        LSTNet_upc_eval.plot_full(test_set)
        LSTNet_upc_eval.save()
        print("{} results: ".format(name), LSTNet_upc_eval.results)

    # LSTnet_category_single
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
        'bs': 100,
    }

    args = AttrDict(args)
    dataset = CategoryTest(path_to_data, args.input_size, 12, features, targets)

    top_categories = np.load("data/top_categories.npy")
    for cat in top_categories:
        test_set = dataset.category_to_ts[cat]
        name = "LSTNet_category_on_{}".format(cat)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        LSTNet_category = load_model(LSTNet, "LSTNet_category", args)
        LSTNet_category_eval = EvaluateMultiStep(LSTNet_category, test_loader, name)
        LSTNet_category_eval.run_rolling()
        LSTNet_category_eval.plot_full(test_set)
        LSTNet_category_eval.save()
        print("{} results: ".format(name), LSTNet_category_eval.results)

    # MultiLSTNet_upc
    features = FEATURES
    targets = FEATURES

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
    dataset = CaseUpcTest(path_to_data, args.input_size, args.output_size, features, targets)

    top_cases = np.load("data/top_cases.npy")
    for case in top_cases:
        test_set = dataset.upc_to_ts[case]
        name = "MultiLSTNet_upc_on_{}".format(case)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        MultiLSTNet_upc = load_model(MultiLSTNet, "MultiLSTNet_upc", args)
        MultiLSTNet_upc_eval = EvaluateMultiStep(MultiLSTNet_upc, test_loader, name)
        MultiLSTNet_upc_eval.run()
        MultiLSTNet_upc_eval.plot_full(test_set)
        MultiLSTNet_upc_eval.save()
        print("{} results: ".format(name), MultiLSTNet_upc_eval.results)

    # MultiLSTNet_category
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
    dataset = CategoryTest(path_to_data, args.input_size, args.output_size, features, targets)

    top_categories = np.load("data/top_categories.npy")
    for cat in top_categories:
        test_set = dataset.category_to_ts[cat]
        name = "MultiLSTNet_category_on_{}".format(cat)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        MultiLSTNet_category = load_model(MultiLSTNet, "MultiLSTNet_category", args)
        MultiLSTNet_category_eval = EvaluateMultiStep(MultiLSTNet_category, test_loader, name)
        MultiLSTNet_category_eval.run()
        MultiLSTNet_category_eval.plot_full(test_set)
        MultiLSTNet_category_eval.save()
        print("{} results: ".format(name), MultiLSTNet_category_eval.results)

    # TCN_upc
    features = FEATURES
    targets = TARGETS
    args = {
        'input_size': 20,
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.001,
        'wd': 0.0005,
        'bs': 1000,
        'epochs': 200
    }
    args = AttrDict(args)
    dataset = CaseUpcTest(path_to_data, args.input_size, args.output_size, features, targets)

    top_cases = np.load("data/top_cases.npy")
    for case in top_cases:
        test_set = dataset.upc_to_ts[case]
        name = "TCN_upc_on_{}".format(case)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        TCN_upc = load_model(TCN, "TCN_upc", args)
        TCN_upc_eval = EvaluateMultiStep(TCN_upc, test_loader, name)
        TCN_upc_eval.run()
        TCN_upc_eval.plot_full(test_set)
        TCN_upc_eval.save()
        print("{} results: ".format(name), TCN_upc_eval.results)

    # TCN_category
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
    dataset = CategoryTest(path_to_data, args.input_size, args.output_size, features, targets)

    top_categories = np.load("data/top_categories.npy")
    for cat in top_categories:
        test_set = dataset.category_to_ts[cat]
        name = "TCN_category_on_{}".format(cat)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
        TCN_category = load_model(TCN, "TCN_upc", args)
        TCN_category_eval = EvaluateMultiStep(TCN_category, test_loader, name)
        TCN_category_eval.run()
        TCN_category_eval.plot_full(test_set)
        TCN_category_eval.save()
        print("{} results: ".format(name), TCN_category_eval.results)


if __name__ == "__main__":
    path_to_data = 'data/UnileverShipmentPOS.csv'
    evaluate(path_to_data)
    evaluate_singles(path_to_data)
    evaluate_full_on_singles(path_to_data)
