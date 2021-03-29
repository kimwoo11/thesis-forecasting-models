import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from data_util import load_npz, train_test_split
from config import *


def data_reshape_rf(x):
    num_samples, num_timesteps, num_features = x.shape[0], x.shape[1], x.shape[2]
    x = np.reshape(np.ravel(x), (num_samples, num_timesteps * num_features))
    return pd.DataFrame(x)


def grid_search_random_forest(X_train, y_train, random_grid, iterations, cv, verbose):
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=iterations, cv=cv, verbose=verbose)
    rf_random.fit(X_train1, y_train1)

    params = rf_random.best_params_
    regr = RandomForestRegressor(**params)
    regr.fit(X_train, y_train)

    return regr


def plot_feat_importance(regr, X_train, fig_title, filename):
    feat_importance = pd.Series(regr.feature_importances_, index=X_train.columns)
    feat_importance = feat_importance.nlargest(55)

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.barh(feat_importance.index, feat_importance.values)
    ax.set_title(fig_title)
    fig.tight_layout()
    plt.savefig("figures/{}".format(filename))


def evaluate(y_pred, y_test):
    y_test.reset_index(drop=True, inplace=True)
    c = y_test.shape[1]
    y_pred = pd.DataFrame(y_pred)

    if c == 1:
        plt.plot(y_pred)
        plt.plot(y_test)

    elif c > 1:

        for i in range(c):
            rmse = mean_squared_error(y_test.iloc[:, i], y_pred.iloc[:, i], squared=False)
            print("Total RMSE for step", str(i + 1), ":", rmse)

        fig, axs = plt.subplots(c, 1, figsize=(10, 20))
        fig.tight_layout(pad=2.0)
        for i in range(c):
            axs[i].plot(y_pred.iloc[:, i])
            axs[i].plot(y_test.iloc[:, i])
            axs[i].set_title(("Step:" + str(i + 1)))
            axs[i].legend(['Prediction', 'Actual'], loc="upper left")

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Total RMSE:", rmse)


if __name__ == "__main__":
    # load datasets
    case_upc_dataset, case_upc_dataset_concat, category_dataset, category_dataset_concat = \
        load_npz('data/unilever_datasets.npz')

    # parameters
    num_targets = len(TARGETS)
    # number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=100, num=4)]

    # number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=4)]
    max_depth.append(None)

    # minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # use the random grid to search for best hyper-parameters
    # edit scoring and feature selection
    iterations = 200
    cv = 2

    cases = case_upc_dataset.keys()

    """ Scenario 1: Sample Instance of Case UPC Dataset """
    sample_case = case_upc_dataset[cases[1]]

    X_train1, y_train1, X_test1, y_test1 = train_test_split(sample_case['X'], sample_case['y'], 0.2)

    X_train1 = data_reshape_rf(X_train1)
    y_train1 = data_reshape_rf(y_train1)
    X_test1 = data_reshape_rf(X_test1)
    y_test1 = data_reshape_rf(y_test1)

    columns = sample_case['agg'].columns
    X_train1.columns = columns[:-(OUTPUT_SIZE * num_targets)]
    X_test1.columns = columns[:-(OUTPUT_SIZE * num_targets)]
    y_train1.columns = columns[-(OUTPUT_SIZE * num_targets):]
    y_test1.columns = columns[-(OUTPUT_SIZE * num_targets):]

    regr1 = grid_search_random_forest(X_train1, y_train1, random_grid, iterations, cv, 3)


