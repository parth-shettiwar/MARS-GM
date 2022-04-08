#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 30 Sep, 2019

@author: wangshuo
"""

import os
import time
import argparse
import sys
import pickle5 as pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader, dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from utils import collate_fn, collate_fn_Film
from model import GraphRec
from dataloader import GRDataset

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='FilmTrust', help='dataset Name')
parser.add_argument('--dataset_path', default='dataset/FilmTrust/', help='dataset directory path: datasets/Ciao/Epinions')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=30, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', action='store_true', help='test')
args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print('Loading data...')
    with open(args.dataset_path + 'dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)

    with open(args.dataset_path + 'list.pkl', 'rb') as f:
        u_items_list = pickle.load(f)
        u_users_list = pickle.load(f)
        u_users_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        i_items_list = None
        i_items_users_list = None
        if args.dataset == 'FilmTrust':
          i_items_list = pickle.load(f)
          i_items_users_list = pickle.load(f)

        (user_count, item_count, rate_count) = pickle.load(f)
        
    
    # train_data, test_data, valid_data = DatasetManager(train_set, valid_set, test_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
    train_data = GRDataset(train_set, u_items_list, u_users_list, u_users_items_list, i_users_list, dataset=args.dataset, i_items_list=i_items_list, i_items_users_list=i_items_users_list)
    valid_data = GRDataset(valid_set, u_items_list, u_users_list, u_users_items_list, i_users_list, dataset=args.dataset, i_items_list=i_items_list, i_items_users_list=i_items_users_list)
    test_data = GRDataset(test_set, u_items_list, u_users_list, u_users_items_list, i_users_list, dataset=args.dataset, i_items_list=i_items_list, i_items_users_list=i_items_users_list)
    # train_data = GRDataset(train_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
    # valid_data = GRDataset(valid_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
    # test_data = GRDataset(test_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
    
    if args.dataset == "FilmTrust":
      train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn_Film)
      valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn_Film)
      test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn_Film)
    else:
      train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
      valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
      test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    
    model = GraphRec(user_count+1, item_count+1, rate_count+1, args.embed_dim, args.dataset).to(device)

    if args.test:
        print('Load checkpoint and testing...')
        ckpt = torch.load('best_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        mae, rmse = validate(test_loader, model)
        print("Test: MAE: {:.4f}, RMSE: {:.4f}".format(mae, rmse))
        return

    optimizer = optim.RMSprop(model.parameters(), args.lr)
    criterion = nn.HuberLoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)
    losses = []
    maes = []
    rmses = []
    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch = epoch)
        curr_loss = trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 100)
        
        losses.append(curr_loss)
        mae, rmse = validate(valid_loader, model)
        maes.append(mae)
        rmses.append(rmse)

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')

        if epoch == 0:
            best_mae = mae
        elif mae < best_mae:
            best_mae = mae
            torch.save(ckpt_dict, 'best_checkpoint.pth.tar')

        print('Epoch {} validation: MAE: {:.4f}, RMSE: {:.4f}, Best MAE: {:.4f}'.format(epoch, mae, rmse, best_mae))

    print("Losses: ",losses)
    print("Maes: ", maes)
    print("Rmses: ", rmses)
    plt.plot(range(args.epoch), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")

    plt.savefig("training_losses.png")

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    total_iter = 0
    for i, (uids, iids, labels, u_items, u_users, u_users_items, i_users, i_items, i_items_users) in tqdm(enumerate(train_loader), total=len(train_loader)):
        uids = uids.to(device)
        iids = iids.to(device)
        labels = labels.to(device)
        u_items = u_items.to(device)
        u_users = u_users.to(device)
        u_users_items = u_users_items.to(device)
        i_users = i_users.to(device)
        if i_items is not None:
          i_items = i_items.to(device)

        if i_items_users is not None:
          i_items_users = i_items_users.to(device)
        optimizer.zero_grad()
        outputs = model(uids, iids, u_items, u_users, u_users_items, i_users, i_item_pad=i_items, i_item_user_pad=i_items_users, dataset=args.dataset)

        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        # scheduler.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        # if i % log_aggr == 0:
        #     print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
        #         % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
        #           len(uids) / (time.time() - start)))
        total_iter += 1
        start = time.time()

    return sum_epoch_loss/total_iter
    


def validate(valid_loader, model):
    model.eval()
    errors = []
    with torch.no_grad():
        for uids, iids, labels, u_items, u_users, u_users_items, i_users, i_items, i_items_users in tqdm(valid_loader):
            uids = uids.to(device)
            iids = iids.to(device)
            labels = labels.to(device)
            u_items = u_items.to(device)
            u_users = u_users.to(device)
            u_users_items = u_users_items.to(device)
            i_users = i_users.to(device)

            if i_items is not None:
                i_items = i_items.to(device)

            if i_items_users is not None:
                i_items_users = i_items_users.to(device)
            preds = model(uids, iids, u_items, u_users, u_users_items, i_users, i_item_pad=i_items, i_item_user_pad=i_items_users, dataset=args.dataset)
            error = torch.abs(preds.squeeze(1) - labels)
            errors.extend(error.data.cpu().numpy().tolist())
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    return mae, rmse


if __name__ == '__main__':
    SEED = 42069

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    main()
