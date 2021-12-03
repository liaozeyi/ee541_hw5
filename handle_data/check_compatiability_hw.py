import pandas as pd

Config = {}
Config['compatiability_path_hw'] = '../polyvore_outfits/test_pairwise_compat_hw.txt'

a = pd.read_csv(Config['compatiability_path_hw'],sep=' ',names=['item_1','item_2'])