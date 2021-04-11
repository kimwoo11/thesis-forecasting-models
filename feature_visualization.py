import matplotlib.pyplot as plt
from datasets import load_csv

if __name__ == "__main__":
    df = load_csv("data/UnileverShipmentPOS.csv")

    # Data setup
    ALL_FEATURES = ['ShipmentCases', 'ShipmentUnits', 'ShipmentNSV', 'DispatchedQty', 'FinalCustomerExpectedOrderQty',
                    'POSUnits', 'POSSales', 'POSCases', 'POSPricePerUnit', 'ShipmentPricePerUnit',
                    'ChangeLossesQty', 'QtyOnHand']

    # == Change categories/cases to inspect ==
    cases = df.CASE_UPC_CD.unique()
    # categories = df.CategoryDesc.unique()
    top_categories = ['FC-MEALMAKERS', '6T-HAIR CARE', '6S-DEO-MALE TOILETRIES', '6P-PERSONAL WASH',
                      'FG-TOTAL CONDIMENTS']

    sample_df1 = df[df['CASE_UPC_CD'] == cases[0]]
    sample_df1 = sample_df1[ALL_FEATURES].dropna()

    sample_df2 = df[df['CASE_UPC_CD'] == cases[1]]
    sample_df2 = sample_df2[ALL_FEATURES].dropna()

    sample_df3 = df[df['CategoryDesc'] == top_categories[2]][ALL_FEATURES].dropna()
    sample_df3 = sample_df3.groupby('WeekNumber').sum()

    sample_df4 = df[df['CategoryDesc'] == top_categories[3]][ALL_FEATURES].dropna()
    sample_df4 = sample_df4.groupby('WeekNumber').sum()

    sample_df_list = [sample_df1, sample_df2, sample_df3, sample_df4]

    # Visualization Setup
    nrows = len(ALL_FEATURES)

    fig, ax = plt.subplots(nrows=nrows, ncols=len(sample_df_list), figsize=(25, 15))
    pad = 25
    fig.suptitle("Feature Plots", fontsize=16)
    for i in range(len(sample_df_list)):
        for j in range(nrows):
            vals = sample_df_list[i].values
            t = sample_df_list[i].index
            y = vals[:, j]
            ax[j][i].plot(t, y)
            ax[j][i].set_title(ALL_FEATURES[j])
            if j == 0:  # first row
                if i < 2:
                    ax[j][i].annotate("Case UPC: {}".format(cases[i]), xy=(0.5, 1),
                                      xytext=(0, pad), xycoords='axes fraction',
                                      textcoords='offset points', size='x-large',
                                      ha='center', va='baseline')
                else:
                    ax[j][i].annotate("Category: {}".format(top_categories[i]), xy=(0.5, 1),
                                      xytext=(0, pad), xycoords='axes fraction',
                                      textcoords='offset points', size='x-large',
                                      ha='center', va='baseline')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # [left, bottom, right, top]
    fig.savefig("figures/feature_visualizations.png")
