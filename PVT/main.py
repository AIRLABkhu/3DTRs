import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse

import os, sys
import shutil
from sklearn import metrics

prj_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(prj_dir)
from model.pvt import pvt
from data.ModelNetDataLoader import ModelNetDataLoader
from criterion import cal_loss

def train(opt, model, trainloader, testloader, criterion, optimizer, scheduler, device):
    best_test_acc = 0.0
    for epoch in range(opt.epoch):
        scheduler.step()
        # train
        running_loss = 0.0
        
        for input, label in trainloader:
            # label = torch.LongTensor(label[:,0].numpy())
            input, label = input.to(device), label.to(device).squeeze()
            input = input.permute(0,2,1)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(trainloader)
        
        print(f'[{epoch}] train loss: {epoch_loss:.3f}')
        
        # test
        test_loss = 0.0
        test_pred = []
        test_label = []
        model.eval()
        
        for input, label in testloader:
            #label = torch.LongTensor(label[:,0].numpy())
            input, label = input.to(device), label.to(device).squeeze()
            input = input.permute(0,2,1)
            output = model(input)
            loss = criterion(output, label)
            preds = output.max(dim=1)[1]
            test_label.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            test_loss += loss.item()
        test_label = np.concatenate(test_label)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_label, test_pred)
        test_loss = test_loss / len(testloader)
        
        print(f'[{epoch}] test loss: {test_loss:.6f}, test acc: {test_acc:.6f}')
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            model_path = opt.expname + os.sep + opt.model + f'model_epoch_{epoch}_{best_test_acc}.pth'
            torch.save(model.state_dict(), model_path)
            
                    
def eval(opt):
    # TODO
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PVT classification')
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batch')
    parser.add_argument('--epoch', type=int, default=200, help='number of epoch to train')
    parser.add_argument('--expname', type=str, default='train_test', help='name of experiment')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer: adam or sgd')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001, 0.01 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')
    parser.add_argument('--data_path', type=str, default='/material/data/modelnet40_normal_resampled', help='data path')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('--model', type=str, default='pvt', help='model name')
    opt = parser.parse_args()
    print(opt)
    
    # make directory
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)
        
    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DataLoader
    DEFAULT_ROOT = opt.data_path
    data = ModelNetDataLoader(ModelNetDataLoader.DEFAULT_ROOT, split='train')
    trainloader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)
    testloader = DataLoader(data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=False)
    
    # Model
    if opt.model == 'pvt':
        model = torch.nn.DataParallel(pvt(), device_ids=[0,1,2,3])
        model = model.cuda()
    else:
        raise NotImplementedError()
    # model = pvt().to(device)
    
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=1e-4)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(model)
    else:
        raise NotImplementedError()
       
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch, eta_min=opt.lr)
    criterion = cal_loss
    
    if opt.mode == 'train':
        train(opt, model, trainloader, testloader, cal_loss, optimizer, scheduler, device)
    elif opt.mode == 'test':
        eval(opt)
    else:
        raise NotImplementedError()