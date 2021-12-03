import numpy as np
import pandas as pd

pd_transfer_total = pd.read_csv('/Users/lzy/Desktop/semeter1/541/HW5/ee541-hw5-starter-master_2/np_total_valid.csv')
compatiability_df = pd.read_csv('/Users/lzy/Desktop/semeter1/541/HW5/ee541-hw5-starter-master_2/my_compatiablitiy_test.csv')
pd_transfer_total_copy = pd_transfer_total.copy()
item_1 = []
item_2 = []
label_list = []
compatiability_df_copy_dict = {}
for index in range(len(compatiability_df)):
    print('This is {}th iter'.format(index))

    compatiability_id_1 = compatiability_df.item_1[index]
    compatiability_id_2 = compatiability_df.item_2[index]
    try:
        index_1 = pd_transfer_total_copy[pd_transfer_total_copy['compatiability_id'].isin([compatiability_id_1])].index.tolist()[0]
        index_2 = pd_transfer_total_copy[pd_transfer_total_copy['compatiability_id'].isin([compatiability_id_2])].index.tolist()[0]

        item_1.append(pd_transfer_total.true_id[index_1])
        item_2.append(pd_transfer_total.true_id[index_2])
        label_list.append(compatiability_df.label[index])
    except:
        continue

compatiability_df_copy_dict['label'] = label_list
compatiability_df_copy_dict['item_1'] = item_1
compatiability_df_copy_dict['item_2'] = item_2

df = pd.DataFrame(compatiability_df_copy_dict)
df.to_csv('/Users/lzy/Desktop/semeter1/541/HW5/ee541-hw5-starter-master_2/my_compatiablitiy_test_changed.csv')