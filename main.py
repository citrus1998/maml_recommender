import os, sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch import autograd

from tqdm import trange

from collections import OrderedDict

from dataset import MovieRatings
from model import MatrixFactorization


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_ratio', type=float, default=0.3, help='')
    parser.add_argument('--query_ratio', type=float, default=0.2, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--num_epochs', type=int, default=1, help='')
    parser.add_argument('--K', type=int, default=5) 
    parser.add_argument('--n_task_loop', type=int, default=10000) 
    parser.add_argument('--alpha', type=float, default=1e-2)
    #parser.add_argument('--beta', type=float, default=5e-1)
    parser.add_argument('--num_latent', type=int, default=32, help='')
    parser.add_argument('--l2_lambda', type=float, default=0.001)
    parser.add_argument('--plot', type=str, default='')
    args = parser.parse_args() 

    data = MovieRatings('datas/ml-latest-small/ratings.csv')

    path_name = 'results/ml-latest-small/'

    n_metatest = int(len(data) * args.task_ratio)
    n_metatrain = len(data) - n_metatest
    m_train, m_test = random_split(data, [n_metatrain, n_metatest])

    meta_train_query_size = int(len(m_train) * args.query_ratio)
    meta_train_support_size = len(m_train) - meta_train_query_size
    meta_train_support, meta_train_query = random_split(m_train, [meta_train_support_size, meta_train_query_size])
    
    meta_test_query_size = int(len(m_test) * args.query_ratio)
    meta_test_support_size = len(m_test) - meta_test_query_size
    meta_test_support, meta_test_query = random_split(m_test, [meta_test_support_size, meta_test_query_size])

    meta_train_dataloaders = {
        'support': DataLoader(meta_train_support, batch_size=args.batch_size, shuffle=True, num_workers=4),
        'query': DataLoader(meta_train_query, batch_size=args.batch_size, shuffle=False, num_workers=4)
    }

    meta_test_dataloaders = {
        'support': DataLoader(meta_test_support, batch_size=args.batch_size, shuffle=True, num_workers=4),
        'query': DataLoader(meta_test_query, batch_size=args.batch_size, shuffle=False, num_workers=4)
    }

    #dataset_sizes = {'support': meta_train_support_size, 'query': meta_train_query_size}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = MatrixFactorization()

    #alpha = torch.tensor([1, 1])
    params = OrderedDict([
        ('user_weight', torch.Tensor(int(data[:, 0].max().item())+1, args.num_latent).uniform_(-1., 1.).requires_grad_()),
        ('item_weight', torch.Tensor(int(data[:, 1].max().item())+1, args.num_latent).uniform_(-1., 1.).requires_grad_()),
    ])

    optimizer = optim.Adam(params.values(), lr=1e-3)

    for epoch in range(args.num_epochs):
        
        meta_train_datasets = {}
        meta_test_datasets = {}

        '''
            Sets Datasets for Meta-Train Phase
        '''

        for batch in meta_train_dataloaders['support']:
            meta_train_datasets['support_users'] = batch[:, 0].to(device)
            meta_train_datasets['support_items'] = batch[:, 1].to(device)
            meta_train_datasets['support_ratings'] = batch[:, 2].float().to(device)

        for batch in meta_test_dataloaders['query']:
            meta_train_datasets['query_users'] = batch[:, 0].to(device)
            meta_train_datasets['query_items'] = batch[:, 1].to(device)
            meta_train_datasets['query_ratings'] = batch[:, 2].float().to(device)
        
        '''
            Set Datasets for Meta-Test Phase
        '''

        for batch in meta_test_dataloaders['support']:
            meta_test_datasets['support_users'] = batch[:, 0].to(device)
            meta_test_datasets['support_items'] = batch[:, 1].to(device)
            meta_test_datasets['support_ratings'] = batch[:, 2].float().to(device)

        for batch in meta_test_dataloaders['query']:
            meta_test_datasets['query_users'] = batch[:, 0].to(device)
            meta_test_datasets['query_items'] = batch[:, 1].to(device)
            meta_test_datasets['query_ratings'] = batch[:, 2].float().to(device)

        '''
            Meta-Train Phase
        '''

        for itr in range(args.n_task_loop):
            users = autograd.Variable(meta_train_datasets['support_users'])
            items = autograd.Variable(meta_train_datasets['support_items'])
            ratings = autograd.Variable(meta_train_datasets['support_ratings'])

            optimizer.zero_grad()

            for k in range(args.K):
                pred_train_ratings, _, _ = model(users.long(), items.long(), params)
                support_loss = F.l1_loss(pred_train_ratings, ratings)
                        
                grads = torch.autograd.grad(support_loss, params.values(), create_graph=True)
                new_params = OrderedDict((name, param - args.alpha * grad) for ((name, param), grad) in zip(params.items(), grads))
                #new_params = OrderedDict((name, param - alpha * grad) for ((name, param), grad) in zip(params.items(), grads))

                if itr % 100 == 0:
                    print('Iteration %d -- Inner loop %d -- Loss: %.4f' % (itr, k, support_loss))
            
            pred_test_ratings, _, _ =  model(users.long(), items.long(), new_params)
            query_loss = F.l1_loss(pred_test_ratings, ratings)
            query_loss.backward(retain_graph=True)
            optimizer.step()
            
            if itr % 100 == 0: 
                print('Iteration %d -- Outer Loss: %.4f' % (itr, query_loss))
        
        '''
            Meta-Test Phase
        '''

        users = autograd.Variable(meta_test_datasets['support_users'])
        items = autograd.Variable(meta_test_datasets['support_items'])
        ratings = autograd.Variable(meta_test_datasets['support_ratings'])

        optimizer.zero_grad()

        for k in range(args.K):
            pred_train_ratings, _, _ = model(users.long(), items.long(), new_params)
            support_loss = F.l1_loss(pred_train_ratings, ratings)
                        
            grads = torch.autograd.grad(support_loss, new_params.values(), create_graph=True)
            new_params = OrderedDict((name, param - args.alpha * grad) for ((name, param), grad) in zip(new_params.items(), grads))
        
        with torch.no_grad():
            users = autograd.Variable(meta_test_datasets['query_users'])
            items = autograd.Variable(meta_test_datasets['query_items'])
            ratings = autograd.Variable(meta_test_datasets['query_ratings'])

            pred_test_ratings, _, _ =  model(users.long(), items.long(), new_params)
            query_loss = F.l1_loss(pred_test_ratings, ratings)
