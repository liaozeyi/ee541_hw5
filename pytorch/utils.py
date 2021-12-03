import numpy as np
import os
import sys
import os.path as osp
import argparse

Config ={}

Config['root_path'] = '../polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''


Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 20

Config['train_parameter'] = ['fc.weight','fc.bias']
Config['batch_size'] = 128

Config['weight_decay'] = 0.001
Config['learning_rate'] = 0.003
Config['num_workers'] = 1

Config['category_hw'] = '/home/ec2-user/lzy/polyvore_outfits/test_category_hw.txt'
Config['category_txt'] = '/home/ec2-user/lzy/category.txt'


Config['pic_path'] = '../pic/'
Config['model_path_compatiability'] = '../model_compatiability/model_compatiability.pth'
Config['model_path_best'] = '../best_model/best_model.pth'

Config['compatiability_path_train'] = '../data/my_compatiablitiy_train_changed_little.csv' #
Config['compatiability_path_valid'] = '../data/my_compatiablitiy_valid_changed_little.csv' #
Config['compatiability_path_hw'] = '../polyvore_outfits/test_pairwise_compat_hw.txt'
Config['compatiability_txt_hw'] = '../data/compatiability_txtx_hw.txt'
