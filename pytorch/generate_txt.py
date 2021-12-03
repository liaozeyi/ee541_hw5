from utils import Config
from model import model
import torch
import pandas as pd
import os.path as osp
import os
import json
import tqdm
from torch.utils.data import Dataset, DataLoader
from data import Test_dataset_hw
import numpy
from write_func import write_func




def generate_txt():
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')


    pred_list = []
    id_list = []
    df = pd.read_csv(Config['category_hw'])

    test_dataset_hw = Test_dataset_hw()
    test_dataloader_hw = DataLoader(test_dataset_hw,batch_size=1024)

    check_point = torch.load(Config['model_path_best'],map_location=device)
    model.load_state_dict(check_point)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for iter,(inputs,id) in enumerate(test_dataloader_hw):
            print('This is the {}th iter'.format(iter))
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            pred_list = pred_list + pred.detach().cpu().numpy().tolist()
            id_list = id_list + id.tolist()
        write_func(id_list,pred_list)
