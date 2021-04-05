from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from data_loader import load_csv

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

    return calc_accuracy(df_pred['ShipmentCases'], df_test['ShipmentCases']), df_pred


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

    features = ['ShipmentCases', 'POSCases']
    cases = df.CASE_UPC_CD.unique()

    accs = []
    best_case = None
    best_acc = 0
    best_pred = None
    for case in cases:
        ts = df[df.CASE_UPC_CD == case][features].dropna()
        if ts.shape[0] > 200:
            ts = normalize_df(ts)
            acc, df_pred = evaluate(ts, n_in, n_out)
            if acc is not None:
                accs.append(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_case = case
                    best_pred = df_pred
    print("Best Accuracy: ", best_acc)
    best_ts = df[df.CASE_UPC_CD == best_case][features].dropna()
    best_ts = normalize_df(best_ts)

    fig, ax = plt.subplots(figsize=(15, 9))
    col = "ShipmentCases"
    best_pred[col].plot(legend=True, label="Predictions").autoscale(axis='x', tight=True)
    best_ts[col][-n_out:].plot(legend=True, ax=ax, label="Targets")
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
    ax.grid(True)

    plt.suptitle("Highest Accuracy Prediction for Case UPC {}".format(best_case), fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # [left, bottom, right, top]
    plt.savefig("figures/VAR_cases_12_step.png")

    # plot accuracy histogram
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.hist(accs, bins=40)
    ax.grid(True)
    ax.set_title("Accuracy Histogram for CaseUPC")
    plt.tight_layout()
    plt.savefig("figures/VAR_upc_accuracy_histogram.png")

    print("Median accuracy across all UPCs: ", np.median(accs))

    categories = df.CategoryDesc.unique()

    accs = []
    best_case = None
    best_acc = 0
    best_pred = None
    for cat in categories:
        ts = df[df.CategoryDesc == cat][features].dropna().groupby('WeekNumber').sum()
        if ts.shape[0] > 200:
            ts = normalize_df(ts)
            acc, df_pred = evaluate(ts, n_in, n_out)
            accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_case = cat
                best_pred = df_pred

    print("Best Accuracy: ", best_acc)
    best_ts = df[df.CategoryDesc == best_case][features].dropna().groupby('WeekNumber').sum()
    best_ts = normalize_df(best_ts)

    fig, ax = plt.subplots(figsize=(15, 9))
    col = "ShipmentCases"
    best_pred[col].plot(legend=True, label="Predictions").autoscale(axis='x', tight=True)
    best_ts[col][-n_out:].plot(legend=True, label="Targets")
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
    ax.grid(True)

    plt.suptitle("Highest Accuracy Prediction For Category {}".format(best_case), fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # [left, bottom, right, top]
    plt.savefig("figures/VAR_categories_12_step.png")

    # plot accuracy histogram
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.hist(accs, bins=40)
    ax.set_title("Accuracy Histogram for Category")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("figures/VAR_category_accuracy_histogram.png")

    print("Median accuracy across all categories: ", np.median(accs))

"""    
Best Accuracy: 0.900303317923492
Median accuracy across all UPCs: 0.03443253810424375
Best Accuracy: 0.6750380114543121
Median accuracy across all categories: 0.020290658609166245
"""
