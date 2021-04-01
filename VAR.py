from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from utils import load_dataset, normalize_df

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

    return calc_rmse(df_pred['ShipmentCases'], df_test['ShipmentCases']), df_pred


if __name__ == "__main__":
    df = load_dataset("data/UnileverShipmentPOS.csv")
    n_in = 20
    n_out = 12

    features = ['ShipmentCases', 'POSCases']
    cases = df.CASE_UPC_CD.unique()

    errors = []
    best_case = None
    best_error = float('inf')
    best_pred = None
    for case in cases:
        ts = df[df.CASE_UPC_CD == case][features].dropna()
        if ts.shape[0] > 100:
            ts = normalize_df(ts)
            error, df_pred = evaluate(ts, n_in, n_out)
            if error is not None:
                errors.append(error)
                if error < best_error:
                    best_error = error
                    best_case = case
                    best_pred = df_pred

    best_ts = df[df.CASE_UPC_CD == best_case][features].dropna()
    best_ts = normalize_df(best_ts)

    fig, axes = plt.subplots(nrows=int(len(best_ts.columns)), figsize=(15, 9))
    for i, (col, ax) in enumerate(zip(best_ts.columns, axes.flatten())):
        best_pred[col].plot(legend=True, ax=ax, label="Predictions").autoscale(axis='x', tight=True)
        best_ts[col][-n_out:].plot(legend=True, ax=ax, label="Targets")
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
        ax.grid(True)

    plt.suptitle("Lowest Error Prediction for Case UPC {}".format(best_case), fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # [left, bottom, right, top]
    plt.savefig("figures/VAR_cases_12_step.png")

    print("MSE Mean across all UPCs: ", np.mean(errors))  # 0.17677832878635388

    categories = df.CategoryDesc.unique()

    errors = []
    best_case = None
    best_error = float('inf')
    best_pred = None
    for cat in categories:
        ts = df[df.CategoryDesc == cat][features].dropna().groupby('WeekNumber').sum()
        if ts.shape[0] > 100:
            ts = normalize_df(ts)
            error, df_pred = evaluate(ts, n_in, n_out)
            errors.append(error)
            if error < best_error:
                best_error = error
                best_case = cat
                best_pred = df_pred

    best_ts = df[df.CategoryDesc == best_case][features].dropna().groupby('WeekNumber').sum()
    best_ts = normalize_df(best_ts)

    fig, axes = plt.subplots(nrows=int(len(best_ts.columns)), figsize=(15, 9))
    for i, (col, ax) in enumerate(zip(best_ts.columns, axes.flatten())):
        best_pred[col].plot(legend=True, ax=ax, label="Predictions").autoscale(axis='x', tight=True)
        best_ts[col][-n_out:].plot(legend=True, ax=ax, label="Targets")
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
        ax.grid(True)

    plt.suptitle("Lowest Error Prediction For Category {}".format(best_case), fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # [left, bottom, right, top]
    plt.savefig("figures/VAR_categories_12_step.png")

    print("MSE Mean across all categories: ", np.mean(errors))  # 0.32031955752873664

    # With ShipmentPricePerUnit:
    #   UPCs: 0.2760834587103174
    #   Categories: 0.3412113133204744

