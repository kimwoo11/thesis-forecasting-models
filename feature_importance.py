import matplotlib.pyplot as plt
import numpy as np
import sklearn.feature_selection as fs

from config import *
from sklearn.preprocessing import MinMaxScaler
from data_util import load_npz


def plot_mi_scores(dataset, n, fig_name, dataset_name):
    fig, ax = plt.subplots(nrows=n, ncols=n, figsize=(20, 10))

    fig.suptitle(fig_name, fontsize=16)

    all_scores = list()

    for i, c in enumerate(list(dataset.keys())[:n**2]):
        agg = dataset[c].agg
        val = agg.values

        # normalize features
        scale = MinMaxScaler(feature_range=(0, 1))
        val = scale.fit_transform(val)

        X = val[:, :len(FEATURES)]
        y = val[:, len(FEATURES)]

        # apply feature selection
        mi = fs.mutual_info_regression(X, y)
        mi /= np.max(mi)
        all_scores.append(mi)

        ax[i // n][i % n].barh(FEATURES, mi)
        ax[i // n][i % n].set_title("{}: {}".format(dataset_name, c))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # [left, bottom, right, top]
    plt.savefig("figures/feature_importance_{}.png".format(dataset_name))
    return all_scores


def plot_average_mi_scores(all_scores, name):
    # MI Scores across all Case UPCs
    all_scores = np.array(all_scores)
    all_scores = all_scores[~np.isnan(all_scores).any(axis=1)]

    avg_scores = np.mean(all_scores, axis=0)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.barh(FEATURES, avg_scores)
    ax.set_title("Averaged MI Scores across all {}".format(name))
    fig.tight_layout()
    plt.savefig("figures/feature_importance_{}_average".format(name))


if __name__ == "__main__":
    case_upc_dataset, _, category_dataset, _ = load_npz('data/unilever_datasets.npz')

    case_all_scores = plot_mi_scores(case_upc_dataset, 5, "MI Scores of 25 Case UPCs", "Case_UPC")
    category_all_scores = plot_mi_scores(category_dataset, 4, "MI Scores of 16 Categories", "Categories")
    plot_average_mi_scores(case_all_scores, "Case_UPC")
    plot_average_mi_scores(category_all_scores, "Categories")


