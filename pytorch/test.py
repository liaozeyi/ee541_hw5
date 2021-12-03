import pandas as pd
from utils import Config

a = pd.read_csv(Config['compatiability_path_hw'],sep=' ',names=['item_1','item_2'])
print(a.head())