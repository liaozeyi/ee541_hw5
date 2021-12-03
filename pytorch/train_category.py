import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp
import os

from utils import Config
from model import model
from data import get_dataloader
from data import Compatiability_dataset_train,Compatiability_dataset_test,Compatiability_dataset_hw
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from write_func import write_func_1



feature_list = [0]

def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    test_acc_list = []
    train_acc_list = []
    test_loss_list = []
    train_loss_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred==labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            if phase == 'train':
                train_acc_list.append(epoch_acc.detach().cpu().numpy())
                train_loss_list.append(epoch_loss)
            if phase =='test':
                test_acc_list.append(epoch_acc.detach().cpu().numpy())
                test_loss_list.append(epoch_loss)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase=='test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth'))
        print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth')))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))

    plt.title('This is train acc')
    plt.plot(range(len(train_acc_list)),train_acc_list)
    plt.savefig(osp.join(Config['pic_path'],'train_acc.jpg'))
    plt.show()

    plt.title('This is test acc')
    plt.plot(range(len(test_acc_list)),test_acc_list)
    plt.savefig(osp.join(Config['pic_path'],'test_acc.jpg'))
    plt.show()

    plt.title('This is train loss')
    plt.plot(range(len(train_loss_list)),train_loss_list)
    plt.savefig(osp.join(Config['pic_path'],'train_loss.jpg'))
    plt.show()

    plt.title('This is test loss')
    plt.plot(range(len(test_loss_list)),test_loss_list)
    plt.savefig(osp.join(Config['pic_path'],'test_loss.jpg'))
    plt.show()


class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.linear_1 = nn.Linear(4096,1024)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(1024,2)

    def forward(self,x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        logits = x
        return logits


def hook_fn_forward(module, input, output):

    feature_list[0] = output


def compatiability_train(my_model,model,compatiability_dataloader_train,device,optimizer,criterion,batch_size):
    model.eval()
    model.to(device)
    my_model.train()
    my_model.to(device)

    train_acc_list = []
    train_loss_list = []

    train_loss = 0
    train_acc = 0
    for label, img_1, img_2 in tqdm(compatiability_dataloader_train):
        handle = model.avgpool.register_forward_hook(hook_fn_forward)

        optimizer.zero_grad()
        label = label.to(device)
        img_1 = img_1.to(device)
        img_2 = img_2.to(device)

        _ = model(img_1)
        feature_1 = torch.flatten(feature_list[0],1)
        feature_1 = torch.tensor(np.array(feature_1.cpu().detach()))
        feature_1 = feature_1.to(device)
        _ = model(img_2)
        feature_2 = torch.flatten(feature_list[0],1)
        feature_2 = torch.tensor(np.array(feature_2.cpu().detach()))
        feature_2 = feature_2.to(device)
        feature_total = torch.cat([feature_1,feature_2],dim=1)
        handle.remove()
        outputs = my_model(feature_total)
        _, pred = torch.max(outputs, 1)
        loss = criterion(outputs,label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()* label.size(0)
        train_acc += torch.sum(pred==label).item()
    print('The loss is{train_loss},the acc is {train_acc}'.format(train_loss = train_loss,train_acc = train_acc))
    return train_loss/batch_size,train_acc/batch_size

def compatiability_test(my_model, model, compatiability_dataloader_test, device, criterion,batch_size):
    model.eval()
    model.to(device)
    my_model.eval()
    my_model.to(device)

    test_acc = 0
    test_loss = 0

    for label,img_1,img_2 in tqdm(compatiability_dataloader_test):
        with torch.no_grad():
            handle = model.avgpool.register_forward_hook(hook_fn_forward)

            label = label.to(device)
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)

            _ = model(img_1)
            feature_1 = torch.flatten(feature_list[0],1)
            _ = model(img_2)
            feature_2 = torch.flatten(feature_list[0],1)
            feature_total = torch.cat([feature_1,feature_2],dim=1)
            handle.remove()
            outputs = my_model(feature_total)

            _, pred = torch.max(outputs, 1)
            loss = criterion(outputs,label)


            test_loss += loss.item() * label.size(0)
            test_acc += torch.sum(pred == label).item()
    print('The loss is{test_loss},the acc is {test_acc}'.format(test_loss = test_loss,test_acc = test_acc))
    return test_loss/batch_size, test_acc/batch_size



def compatiability_hw(model,my_model,compatiability_dataloader_hw):
    pred_list = []
    for img_1,img_2 in compatiability_dataloader_hw:
        with torch.no_grad():
            handle = model.avgpool.register_forward_hook(hook_fn_forward)
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)

            _ = model(img_1)
            feature_1 = torch.flatten(feature_list[0],1)
            _ = model(img_2)
            feature_2 = torch.flatten(feature_list[0],1)
            feature_total = torch.cat([feature_1,feature_2],dim=1)
            handle.remove()
            outputs = my_model(feature_total)

            _, pred = torch.max(outputs, 1)
            pred = pred.detach().cpu().numpy().tolist()
            pred_list += pred
    write_func_1(pred_list)





if __name__=='__main__':
    for_hw_1 = False
    for_hw_2 = True


    if for_hw_1 is True:
        dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
        generate_txt()

    elif for_hw_2 is True:
        batch_size = 128
        epcoh = 17 #需要调整
        #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
        #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

        acc_train = 0
        acc_test = 0
        loss_train = 0
        loss_test = 0

        best_acc = 0
        acc_train_list = []
        acc_test_list =[]
        loss_train_list = []
        loss_test_list = []
        device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 153)
        check_point = torch.load(Config['model_path_best'], map_location=device)
        model.load_state_dict(check_point)

        my_model = My_model()
        check_point = torch.load(Config['model_path_compatiability'], map_location=device)
        my_model.load_state_dict(check_point)



        #criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(params=(list(model.parameters()) + list(my_model.parameters())), lr=Config['learning_rate'])


        compatiability_dataset_hw = Compatiability_dataset_hw()
        compatiability_dataloader_hw = DataLoader(compatiability_dataset_hw, batch_size=batch_size)

        compatiability_hw(model,my_model,compatiability_dataloader_hw)

        compatiability_dataset_train = Compatiability_dataset_train()
        compatiability_dataloader_train = DataLoader(compatiability_dataset_train, batch_size=batch_size, shuffle=True)


        compatiability_dataset_test = Compatiability_dataset_test()
        compatiability_dataloader_test = DataLoader(compatiability_dataset_test, batch_size=batch_size)


        for epoch_num in range(epcoh):
            print('This is {}th epoch'.format(epoch_num))
            loss_train,acc_train = compatiability_train(my_model,model,compatiability_dataloader_train,device,optimizer,criterion,batch_size)
            loss_test,acc_test = compatiability_test(my_model,model,compatiability_dataloader_test,device,criterion,batch_size)
            acc_train_list.append(acc_train)
            acc_test_list.append(acc_test)
            loss_train_list.append(loss_train)
            loss_test_list.append(loss_test)

            if acc_test > best_acc:
                best_acc = acc_test
                best_pth = copy.deepcopy(my_model.state_dict())
        torch.save(my_model.state_dict(),Config['model_path_compatiability'],_use_new_zipfile_serialization=False)

        plt.title('This is train acc')
        plt.plot(range(len(acc_train_list)), acc_train_list)
        plt.savefig(osp.join(Config['pic_path'], 'train_acc.jpg'))
        plt.show()

        plt.title('This is test acc')
        plt.plot(range(len(acc_test_list)), acc_test_list)
        plt.savefig(osp.join(Config['pic_path'], 'test_acc.jpg'))
        plt.show()

        plt.title('This is train loss')
        plt.plot(range(len(loss_train_list)), loss_train_list)
        plt.savefig(osp.join(Config['pic_path'], 'train_loss.jpg'))
        plt.show()

        plt.title('This is test loss')
        plt.plot(range(len(loss_test_list)), loss_test_list)
        plt.savefig(osp.join(Config['pic_path'], 'test_loss.jpg'))
        plt.show()

    else:
        dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)

        # for name, value in model.named_parameters():
        #     value.requires_grad = False
        #
        #     if name in Config['train_parameter']:
        #         value.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate'])
        device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

        train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)