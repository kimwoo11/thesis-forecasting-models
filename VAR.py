from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from datasets import load_csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def is_stationary(dataset):
    non_stationary = False
    for i in range(len(dataset.columns)):
        res = adfuller(dataset[dataset.columns[i]])
        if res[1] > 0.05:
            # print("{} - Series is not stationary".format(dataset.columns[i]))
            non_stationary = True
        # else:
        # print("{} - Series is stationary".format(dataset.columns[i]))
    return non_stationary


def invert_transformation(df_train, df_forecast):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 1st Diff
        df_fc[str(col) + '_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()
    return df_fc


def plot_features(dataset, features):
    """
    :param dataset: 2d dataframe (num_samples, num_features)
    """
    fig, axes = plt.subplots(nrows=len(features), figsize=(15, 9))
    for i, ax in enumerate(axes.flatten()):
        data = dataset[dataset.columns[i]]
        ax.plot(data, color='red', linewidth=1)
        # Decorations
        ax.set_title(dataset.columns[i])
    plt.tight_layout()
    plt.show()


def calc_rmse(pred, df_test):
    return np.sqrt(mean_squared_error(pred, df_test))


def calc_accuracy(pred, df_test):
    df_test[df_test == 0] = 0.01
    diff = abs(df_test - pred) / df_test
    return np.mean(1 - diff)


def calc_bias(pred, df_test):
    return np.mean(abs(df_test - pred) / pred)


def evaluate(ts, n_in, n_out):
    df_train = ts[:-n_out]
    df_test = ts[-n_out:]

    # first order differencing (no 2nd order)
    num_diff = 0
    while is_stationary(df_train) and num_diff < 1:
        # print("Non-stationary series exists; applying differencing...")
        df_train = df_train.diff().dropna()
        num_diff += 1

    # print("All series are stationary")
    try:
        model = VAR(df_train)
        fitted_model = model.fit(maxlags=n_in, ic='aic')
    except:
        return None, None

    lag_order = fitted_model.k_ar
    fc = fitted_model.forecast(df_train.values[-lag_order:], n_out)
    df_pred = pd.DataFrame(fc, index=ts.index[-n_out:], columns=ts.columns)

    if num_diff > 0:
        df_pred = invert_transformation(df_train, df_pred)

    return calc_accuracy(df_pred['ShipmentCases'], df_test['ShipmentCases']), \
           calc_bias(df_pred['ShipmentCases'], df_test['ShipmentCases']), \
           calc_rmse(df_pred['ShipmentCases'], df_test['ShipmentCases']), df_pred


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


if __name__ == "__main__":
    df = load_csv("data/UnileverShipmentPOS.csv")
    n_in = 20
    n_out = 12
    results = []

    features = ['ShipmentCases', 'POSCases']
    cases = df.CASE_UPC_CD.unique()

    accs = []
    biases = []
    rmses = []


    for case in cases:
        ts = df[df.CASE_UPC_CD == case][features].dropna()
        if ts.shape[0] > 200:
            ts = normalize_df(ts)
            acc, bias, rmse, df_pred = evaluate(ts, n_in, n_out)
            if acc is not None:
                accs.append(acc)
                biases.append(bias)
                rmses.append(rmse)

    # plot accuracy histogram
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.hist(accs, bins=40)
    ax.grid(True)
    ax.set_title("Accuracy Histogram for CaseUPC")
    plt.tight_layout()
    plt.savefig("figures/VAR_upc_accuracy_histogram.png")

    print("VAR_upc")
    res = [np.median(rmses), np.median(accs), np.median(biases)]
    print(np.median(rmses))
    print(np.median(accs))
    print(np.median(biases))
    np.savetxt("results/{}_upc_evaluation_results.csv".format("VAR"), res, delimiter=',')



    categories = df.CategoryDesc.unique()

    accs = []
    biases = []
    rmses = []

    for cat in categories:
        ts = df[df.CategoryDesc == cat][features].dropna().groupby('WeekNumber').sum()
        if ts.shape[0] > 200:
            ts = normalize_df(ts)
            acc, bias, rmse, df_pred = evaluate(ts, n_in, n_out)
            accs.append(acc)
            biases.append(bias)
            rmses.append(rmse)

    # plot accuracy histogram
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.hist(accs, bins=40)
    ax.set_title("Accuracy Histogram for Category")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("figures/VAR_category_accuracy_histogram.png")

    print("VAR_category")
    res = [np.median(rmses), np.median(accs), np.median(biases)]
    print(np.median(rmses))
    print(np.median(accs))
    print(np.median(biases))
    np.savetxt("results/{}_category_evaluation_results.csv".format("VAR"), res, delimiter=',')


    top_cases = np.load("data/top_cases.npy")
    top_categories = np.load("data/top_categories.npy")

    for case in top_cases:
        name = "VAR_upc_{}".format(case)
        ts = df[df.CASE_UPC_CD == case][features].dropna()
        ts = normalize_df(ts)
        acc, bias, rmse, df_pred = evaluate(ts, n_in, n_out)
        if acc is None:
            print("ABORT!!")
        res = [rmse, acc, bias]
        np.savetxt("results/{}_upc_{}_evaluation_results.csv".format("VAR", case), res, delimiter=',')

        fig, ax = plt.subplots(figsize=(15, 9))
        col = "ShipmentCases"

        t = ts.index
        ax.plot(t[-n_out:], ts[col].iloc[-n_out:], color='blue', label="Target")
        ax.plot(t[-n_out:], df_pred[col], color='red', label="Prediction")
        ax.plot(t[:-n_out], ts[col].iloc[:-n_out], 'k')
        ax.plot([t[-13], t[-12]], [ts[col].iloc[-13], ts[col][-12]], color='blue')  # connect to previous sequence
        ax.plot([t[-13], t[-12]], [ts[col].iloc[-13], df_pred[col][0]], color='red')
        ax.legend()

        ax.grid(True)
        ax.set_title("ShipmentCases Forecast vs Target for {}".format(name))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig("{}_{}_step.png".format(name, 12))
        plt.close(fig)

    top_categories = np.load("data/top_categories.npy")

    for cat in top_categories:
        name = "VAR_category_{}".format(cat)
        ts = df[df.CategoryDesc == cat][features].dropna().groupby('WeekNumber').sum()
        ts = normalize_df(ts)
        acc, bias, rmse, df_pred = evaluate(ts, n_in, n_out)
        if acc is None:
            print("ABORT!!")
        res = [rmse, acc, bias]
        np.savetxt("results/{}_category_{}_evaluation_results.csv".format("VAR", cat), res, delimiter=',')

        fig, ax = plt.subplots(figsize=(15, 9))
        col = "ShipmentCases"

        t = ts.index
        ax.plot(t[-n_out:], ts[col][-n_out:], color='blue', label="Target")
        ax.plot(t[-n_out:], df_pred[col], color='red', label="Prediction")
        ax.plot(t[:-n_out], ts[col].iloc[:-n_out], 'k')
        ax.plot([t[-13], t[-12]], [ts[col].iloc[-13], ts[col].iloc[-12]], color='blue')  # connect to previous sequence
        ax.plot([t[-13], t[-12]], [ts[col].iloc[-13], df_pred[col][0]], color='red')
        ax.legend()

        ax.grid(True)
        ax.set_title("ShipmentCases Forecast vs Target for {}".format(name))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig("{}_{}_step.png".format(name, 12))
        plt.close(fig)