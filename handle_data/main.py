import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp
import json


# with open('/Users/lzy/Desktop/semeter1/541/HW5/polyvore_outfits/compatibility_train.txt','r') as f:
#     data = f.read()
name_list = []
#
# for i in range(20):
#     name = 'data'
#     name = name + '_' + str(i)
#     name_list.append(name)
#
# data_2 = pd.read_csv('/Users/lzy/Desktop/semeter1/541/HW5/polyvore_outfits/compatibility_train.txt',sep=' ',names=name_list)
# print(data_2)
# print('今天好好学习！')
#
# list_total = list(data_2.loc[0])
# a = [a_ for a_ in list_total if a_ == a_]
# print(a)
#
# data = {}
# data['label']= 0
# data['item_1'] = 0
# data['item_2'] = 0
#
# label = []
# item_1 = []
# item_2 = []
#
# for index in tqdm(range(106612)):
#     list_total = list(data_2.loc[index])
#     a = [a_ for a_ in list_total if a_ == a_]
#     length = len(a)-1
#     for i in range(1,length+1):
#         for j in range(length-2,length+1):
#             label.append(a[0])
#             item_1.append(a[i])
#             item_2.append(a[j])
#
# data['label']= label
# data['item_1'] = item_1
# data['item_2'] = item_2
#
# df = pd.DataFrame(data)
# df.to_csv('/Users/lzy/Desktop/semeter1/541/HW5/polyvore_outfits/my_compatiablitiy.csv')




np_total = np.zeros([1,2])
meta_file_train = open('/Users/lzy/Desktop/semeter1/541/HW5/polyvore_outfits/valid.json', 'r')
meta_json_train = json.load(meta_file_train)
for k in tqdm(meta_json_train):
    set_id = k['set_id']
    items = k['items']
    for item in items:
        index = item['index']
        item_id = item['item_id']

        compatiability_id = str(set_id) + '_' + str(index)
        true_id = item_id
        new_np = np.array([[true_id,compatiability_id]])
        np_total = np.concatenate((np_total,new_np),axis=0)

pd_data= pd.DataFrame(np_total,columns=['true_id','compatiability_id'])
pd_data.to_csv('/Users/lzy/Desktop/semeter1/541/HW5/ee541-hw5-starter-master_2/np_total_valid.csv',index=False)

#np.save('/Users/lzy/Desktop/semeter1/541/HW5/ee541-hw5-starter-master_2/np_total_valid.npy',np_total)

#
#
# meta_file = open('/Users/lzy/Desktop/semeter1/541/HW5/polyvore_outfits/polyvore_item_metadata.json', 'r')
# meta_json = json.load(meta_file)
# for k, v in tqdm(meta_json.items()):
#     if 'Style Number' in v['description']:
#         print(v['description'])
#         print('\n')