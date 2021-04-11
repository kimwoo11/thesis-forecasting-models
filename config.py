# from datasets import *

INPUT_SIZE = 20
OUTPUT_SIZE = 12
FEATURES = ['ShipmentCases', 'POSCases', 'POSPricePerUnit', 'ShipmentPricePerUnit']
TARGETS = ['ShipmentCases']


# if __name__ == "__main__":
#     df = load_csv("data/UnileverShipmentPOS.csv")
#     org_cases = df.CASE_UPC_CD.unique()
#
#     cases = []
#     for case in org_cases:
#         ds = df[df.CASE_UPC_CD == case][FEATURES].dropna()
#         if ds.shape[0] > 200:
#             cases.append(case)
#
#     num_cases = len(cases)
#     tv_cases = np.random.choice(cases, int(num_cases*0.9), replace=False)
#     test_cases = []
#     for c in cases:
#         if c not in tv_cases:
#             test_cases.append(c)
#
#     np.save("data/tv_cases.npy", tv_cases)
#     np.save("data/test_cases.npy", test_cases)
#
#     org_categories = df.CategoryDesc.unique()
#     categories = []
#     for cat in org_categories:
#         ds = df[df.CategoryDesc == cat][FEATURES].dropna().groupby('WeekNumber').sum()
#         if ds.shape[0] > 200:
#             categories.append(cat)
#
#     num_cat = len(categories)
#     tv_categories = np.random.choice(categories, int(num_cat*0.9), replace=False)
#     test_categories = []
#     for cat in categories:
#         if cat not in tv_categories:
#             test_categories.append(cat)
#
#     np.save("data/tv_categories.npy", tv_categories)
#     np.save("data/test_categories.npy", test_categories)

