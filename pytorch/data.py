import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

from utils import Config

import pandas as pd


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()



    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms



    def create_dataset(self):
        # map id to category
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']

        # create X, y pairs
        files = os.listdir(self.image_dir)
        X = []; y = []
        for x in files:
            if x[:-4] in id_to_category:
                X.append(x)
                y.append(int(id_to_category[x[:-4]]))

        y = LabelEncoder().fit_transform(y)
        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1


class Test_dataset_hw(Dataset):
    def __init__(self, path=Config['category_hw']):
        self.df = pd.read_csv(Config['category_hw'],header=None,names=['label'])
        self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        id = self.df.label[item]
        id_jpg = str(id) + '.jpg'
        return self.transforms(Image.open(Config['root_path'] + '/images/' + id_jpg)),id


class Compatiability_dataset_train(Dataset):
    def __init__(self,path= Config['compatiability_path_train']):
        self.df = pd.read_csv(path)
        self.transofroms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.df)


    def __getitem__(self, item):
        id_1 = int(self.df.item_1[item])
        id_2 = int(self.df.item_2[item])
        label = self.df.label[item]
        id_1_jpg = str(id_1) + '.jpg'
        id_2_jpg = str(id_2) + '.jpg'
        return label,self.transofroms(Image.open(Config['root_path'] + '/images/' + id_1_jpg)),self.transofroms(Image.open(Config['root_path'] +'/images/' +  id_2_jpg))



class Compatiability_dataset_test(Dataset):
    def __init__(self,path= Config['compatiability_path_valid']):
        self.df = pd.read_csv(path)
        self.transofroms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.df)


    def __getitem__(self, item):
        id_1 = int(self.df.item_1[item])
        id_2 = int(self.df.item_2[item])
        label = self.df.label[item]
        id_1_jpg = str(id_1) + '.jpg'
        id_2_jpg = str(id_2) + '.jpg'
        return label,self.transofroms(Image.open(Config['root_path'] +'/images/' + id_1_jpg)),self.transofroms(Image.open(Config['root_path'] +'/images/' + id_2_jpg))

class Compatiability_dataset_hw(Dataset):
    def __init__(self,path= Config['compatiability_path_hw']):
        self.df = pd.read_csv(path,sep=' ',names=['item_1','item_2'])
        self.transofroms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.df)


    def __getitem__(self, item):
        id_1 = int(self.df.item_1[item])
        id_2 = int(self.df.item_2[item])
        id_1_jpg = str(id_1) + '.jpg'
        id_2_jpg = str(id_2) + '.jpg'
        return self.transofroms(Image.open(Config['root_path'] +'/images/' + id_1_jpg)),self.transofroms(Image.open(Config['root_path'] +'/images/' + id_2_jpg))




# For category classification
class polyvore_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_train[item])
        return self.transform(Image.open(file_path)),self.y_train[item]




class polyvore_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')


    def __len__(self):
        return len(self.X_test)


    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_test[item])
        return self.transform(Image.open(file_path)), self.y_test[item]




def get_dataloader(debug, batch_size):
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes = dataset.create_dataset()

    if debug==True:
        train_set = polyvore_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train'])
        test_set = polyvore_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size)
                                 for x in ['train', 'test']}
    return dataloaders, classes, dataset_size


########################################################################
# For Pairwise Compatibility Classification

# class train_compatibility_dataset(Dataset):
#     def __init__(self, root=Config['compatibility_path'],transforms):
#         self.df = pd.read_csv(root)
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.df['label'])
#
#     def __getitem__(self, item):
#         return (self.df['label'][item], self.df['item_1'][item], self.df['item_2'][item])
#
#
