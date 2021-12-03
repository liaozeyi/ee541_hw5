import pandas as pd
import numpy as np
from tqdm import tqdm
import os.path as osp
import json
import random


with open('/Users/lzy/Desktop/semeter1/541/HW5/polyvore_outfits/compatibility_train.txt','r') as f:
    data = f.read()
name_list = []

for i in range(20):
    name = 'data'
    name = name + '_' + str(i)
    name_list.append(name)

data_2 = pd.read_csv('/Users/lzy/Desktop/semeter1/541/HW5/polyvore_outfits/compatibility_valid.txt',sep=' ',names=name_list)
####这个秒啊！！！！！！！
#####
#
##
#
#
#
#
#
#
#

print(data_2)
print('今天好好学习！')

list_total = list(data_2.loc[0])
a = [a_ for a_ in list_total if a_ == a_]
print(a)

data = {}
data['label']= 0
data['item_1'] = 0
data['item_2'] = 0

label = []
item_1 = []
item_2 = []

for index in tqdm(range(10000)):
    list_total = list(data_2.loc[index])
    a = [a_ for a_ in list_total if a_ == a_]
    length = len(a)-1
    try:
        index = random.sample(range(1,length+1),3)
    except:
        continue

    label.append(a[0])
    item_1.append(a[index[0]])
    item_2.append(a[index[1]])

    label.append(a[0])
    item_1.append(a[index[0]])
    item_2.append(a[index[2]])

    label.append(a[0])
    item_1.append(a[index[1]])
    item_2.append(a[index[2]])


data['label']= label
data['item_1'] = item_1
data['item_2'] = item_2

df = pd.DataFrame(data)
df.to_csv('/Users/lzy/Desktop/semeter1/541/HW5/polyvore_outfits/my_compatiablitiy_test.csv')

