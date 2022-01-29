'''
The main driving code 

    1. CML/FL Training

    2. Compute/Approximate Cosine Gradient Shapley

    3. Calculate and realize the fair gradient reward

'''

import os, sys, json
from os.path import join as oj
import copy
from copy import deepcopy as dcopy
import time, datetime, random, pickle
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.linalg import norm
from torchtext.data import Batch
import torch.nn.functional as F


from utils.Data_Prepper import Data_Prepper
from utils.arguments import mnist_args, cifar_cnn_args, mr_args, sst_args

from utils.utils import cwd, train_model, evaluate, cosine_similarity, mask_grad_update_by_order, \
    compute_grad_update, add_update_to_model, add_gradient_updates,\
    flatten, unflatten, compute_distance_percentage


import argparse

parser = argparse.ArgumentParser(description='Process which dataset to run')
parser.add_argument('-D', '--dataset', help='Pick the dataset to run.', type=str, required=True)
parser.add_argument('-N', '--n_agents', help='The number of agents.', type=int, default=5)

parser.add_argument('-nocuda', dest='cuda', help='Not to use cuda even if available.', action='store_false')
parser.add_argument('-cuda', dest='cuda', help='Use cuda if available.', action='store_true')


parser.add_argument('-split', '--split', dest='split', help='The type of data splits.', type=str, default='all', choices=['all', 'uni', 'cla', 'pow'])

cmd_args = parser.parse_args()

print(cmd_args)

N = cmd_args.n_agents

if torch.cuda.is_available() and cmd_args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if cmd_args.dataset == 'mnist':
    args = copy.deepcopy(mnist_args)

    if N > 0:
        agent_iterations = [[N, N*600]]
    else:
        agent_iterations = [[5,3000], [10, 6000], [20, 12000]]

    if cmd_args.split == 'uni':
        splits = ['uniform']
    
    elif cmd_args.split == 'pow':
        splits = ['powerlaw']
    
    elif cmd_args.split == 'cla':
        splits = ['classimbalance']
    
    elif cmd_args.split == 'all':
        splits = ['uniform', 'powerlaw', 'classimbalance',]
    
    args['iterations'] = 200
    args['E'] = 3
    args['lr'] = 1e-3
    args['num_classes'] = 10
    args['lr_decay'] = 0.955

elif cmd_args.dataset == 'cifar10':
    args = copy.deepcopy(cifar_cnn_args)    

    if N > 0:
        agent_iterations = [[N, N*2000]]
    else:
        agent_iterations = [[10, 20000]]

    if cmd_args.split == 'uni':
        splits = ['uniform']
    
    elif cmd_args.split == 'pow':
        splits = ['powerlaw']
    
    elif cmd_args.split == 'cla':
        splits = ['classimbalance']
    
    elif cmd_args.split == 'all':
        splits = ['uniform', 'powerlaw', 'classimbalance']

    args['iterations'] = 200
    args['E'] = 3
    args['num_classes'] = 10

elif cmd_args.dataset == 'sst':
    args = copy.deepcopy(sst_args)  
    agent_iterations = [[5, 8000]]
    splits = ['powerlaw']
    args['iterations'] = 200
    args['E'] = 3
    args['num_classes'] = 5
    
elif cmd_args.dataset == 'mr':
    args = copy.deepcopy(mr_args)   
    agent_iterations = [[5, 8000]]
    splits = ['powerlaw']
    args['iterations'] = 200
    args['E'] = 3
    args['num_classes'] = 2


E = args['E']

ts = time.time()
time_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M')

for N, sample_size_cap in agent_iterations:
    
    args.update(vars(cmd_args))


    args['n_agents'] = N
    args['sample_size_cap'] = sample_size_cap
    # args['momentum'] = 1.5 / N

    for beta in [0.5, 1, 1.2, 1.5, 2, 1e7]:
        args['beta'] = beta

        for split in splits:
            args['split'] = split

            optimizer_fn = args['optimizer_fn']
            loss_fn = args['loss_fn']

            print(args)
            print("Data Split information for the agents:")
            data_prepper = Data_Prepper(
                args['dataset'], train_batch_size=args['batch_size'], n_agents=N, sample_size_cap=args['sample_size_cap'], 
                train_val_split_ratio=args['train_val_split_ratio'], device=device, args_dict=args)

            # valid_loader = data_prepper.get_valid_loader()
            test_loader = data_prepper.get_test_loader()

            train_loaders = data_prepper.get_train_loaders(N, args['split'])
            shard_sizes = data_prepper.shard_sizes


            # shard sizes refer to the sizes of the local data of each agent
            shard_sizes = torch.tensor(shard_sizes).float()
            relative_shard_sizes = torch.div(shard_sizes, torch.sum(shard_sizes))         
            print("Shard sizes are: ", shard_sizes.tolist())

            if args['dataset'] in ['mr', 'sst']:
                server_model = args['model_fn'](args=data_prepper.args).to(device)
            else:
                server_model = args['model_fn']().to(device)

            D = sum([p.numel() for p in server_model.parameters()])
            init_backup = dcopy(server_model)

            # ---- init the agents ----
            agent_models, agent_optimizers, agent_schedulers = [], [], []

            for i in range(N):
                model = copy.deepcopy(server_model)
                # try:
                    # optimizer = optimizer_fn(model.parameters(), lr=args['lr'], momentum=args['momentum'])
                # except:

                optimizer = optimizer_fn(model.parameters(), lr=args['lr'])

                # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)                    
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args['lr_decay'])

                agent_models.append(model)
                agent_optimizers.append(optimizer)
                agent_schedulers.append(scheduler)


            # ---- book-keeping variables

            rs_dict, qs_dict = [], []
            rs = torch.zeros(N, device=device)
            past_phis = []

            # for performance analysis
            valid_perfs, local_perfs, fed_perfs = defaultdict(list), defaultdict(list), defaultdict(list)

            # for gradient/model parameter analysis
            dist_all_layer, dist_last_layer = defaultdict(list),  defaultdict(list)
            reward_all_layer, reward_last_layer=  defaultdict(list),  defaultdict(list)

        # ---- CML/FL begins ---- 
            for iteration in range(args['iterations']):

                gradients = []
                for i in range(N):
                    loader = train_loaders[i]
                    model = agent_models[i]
                    optimizer = agent_optimizers[i]
                    scheduler = agent_schedulers[i]

                    model.train()
                    model = model.to(device)

                    backup = copy.deepcopy(model)

                    model = train_model(model, loader, loss_fn, optimizer, device=device, E=E, scheduler=scheduler)

                    gradient = compute_grad_update(old_model=backup, new_model=model, device=device)
                    

                    # SUPPOSE DO NOT TOP UP WITH OWN GRADIENTS
                    model.load_state_dict(backup.state_dict())
                    # add_update_to_model(model, gradient, device=device)

                    # append the normalzied gradient
                    flattened = flatten(gradient)
                    norm_value = norm(flattened) + 1e-7 # to prevent division by zero
                         
                    gradient = unflatten(torch.multiply(torch.tensor(args['Gamma']), torch.div(flattened,  norm_value)), gradient)
                    gradients.append(gradient)

                        
            # ---- Server Aggregate ----

                aggregated_gradient =  [torch.zeros(param.shape).to(device) for param in server_model.parameters()]

                # aggregate and update server model

                if iteration == 0:
                    # first iteration use FedAvg
                    weights = torch.div(shard_sizes, torch.sum(shard_sizes))
                else:
                    weights = rs

                for gradient, weight in zip(gradients, weights):
                    add_gradient_updates(aggregated_gradient, gradient, weight=weight)

                add_update_to_model(server_model, aggregated_gradient)

                # update reputation and calculate reward gradients
                flat_aggre_grad = flatten(aggregated_gradient)

                # phis = torch.zeros(N, device=device)
                phis = torch.tensor([F.cosine_similarity(flatten(gradient), flat_aggre_grad, 0, 1e-10) for gradient in gradients], device=device)
                past_phis.append(phis)

                rs = args['alpha'] * rs + (1 - args['alpha']) * phis

                rs = torch.clamp(rs, min=1e-3) # make sure the rs do not go negative
                rs = torch.div(rs, rs.sum()) # normalize the weights to 1 
                
                # --- altruistic degree function
                q_ratios = torch.tanh(args['beta'] * rs) 
                q_ratios = torch.div(q_ratios, torch.max(q_ratios))
                
                qs_dict.append(q_ratios)
                rs_dict.append(rs)


                for i in range(N):

                    reward_gradient = mask_grad_update_by_order(aggregated_gradient, mask_percentile=q_ratios[i], mode='layer')

                    add_update_to_model(agent_models[i], reward_gradient)


                    ''' Analysis of rewarded gradients in terms cosine to the aggregated gradient '''
                    reward_all_layer[str(i)+'cos'].append(F.cosine_similarity(flatten(reward_gradient), flat_aggre_grad, 0, 1e-10).item()  )
                    reward_all_layer[str(i)+'l2'].append(norm(flatten(reward_gradient) - flat_aggre_grad).item())

                    reward_last_layer[str(i)+'cos'].append(F.cosine_similarity(flatten(reward_gradient[-2]), flatten(aggregated_gradient[-2]), 0, 1e-10).item()  )
                    reward_last_layer[str(i)+'l2'].append(norm(flatten(reward_gradient[-2])- flatten(aggregated_gradient[-2])).item())


                weights = torch.div(shard_sizes, torch.sum(shard_sizes)) if iteration == 0 else rs

                for i, model in enumerate(agent_models + [server_model]):

                    loss, accuracy = evaluate(model, test_loader, loss_fn=loss_fn, device=device)

                    valid_perfs[str(i)+'_loss'].append(loss.item())
                    valid_perfs[str(i)+'_accu'].append(accuracy.item())

                    fed_loss, fed_accu = 0, 0
                    for j, train_loader in enumerate(train_loaders):
                        loss, accuracy = evaluate(model, train_loader, loss_fn=loss_fn, device=device)

                        fed_loss += weights[j] * loss.item()
                        fed_accu += weights[j] * accuracy.item()
                        if j == i:
                            local_perfs[str(i)+'_loss'].append(loss.item())
                            local_perfs[str(i)+'_accu'].append(accuracy.item())
                    
                    fed_perfs[str(i)+'_loss'].append(fed_loss.item())
                    fed_perfs[str(i)+'_accu'].append(fed_accu.item())

                # ---- Record model distance to the server model ----
                for i, model in enumerate(agent_models + [init_backup]) :

                    percents, dists = compute_distance_percentage(model, server_model)

                    dist_all_layer[str(i)+'dist'].append(np.mean(dists))
                    dist_last_layer[str(i)+'dist'].append(dists[-1])
                    
                    dist_all_layer[str(i)+'perc'].append(np.mean(percents))
                    dist_last_layer[str(i)+'perc'].append(percents[-1])


            # Saving results, into csvs
            agent_str = '{}-{}'.format(args['split'][:3].upper(), 'A'+str(N), )

            folder = oj('RESULTS', args['dataset'], time_str, agent_str, 
                'beta-{}'.format(str(args['beta'])[:4]) )

            os.makedirs(folder, exist_ok=True)

            with cwd(folder):

                # distance to the full gradient: all layers and only last layer of the model parameters
                pd.DataFrame(reward_all_layer).to_csv(('all_layer.csv'), index=False)

                pd.DataFrame(reward_last_layer).to_csv(('last_layer.csv'), index=False)

                # distance to server model parameters: all layers and only last layer of the model parameters
                pd.DataFrame(dist_all_layer).to_csv(('dist_all_layer.csv'), index=False)

                pd.DataFrame(dist_last_layer).to_csv(('dist_last_layer.csv'), index=False)
                

                # importance coefficients rs
                rs_dict = torch.stack(rs_dict).detach().cpu().numpy()
                df = pd.DataFrame(rs_dict)
                df.to_csv(('rs.csv'), index=False)

                # q values
                qs_dict = torch.stack(qs_dict).detach().cpu().numpy()
                df = pd.DataFrame(qs_dict)
                df.to_csv(('qs.csv'), index=False)

                # federated performance (local objectives weighted w.r.t the importance coefficient rs)
                df = pd.DataFrame(fed_perfs)
                df.to_csv(('fed.csv'), index=False)

                # validation performance
                df = pd.DataFrame(valid_perfs)
                df.to_csv(('valid.csv'), index=False)

                # local performance (only on local training set)
                df = pd.DataFrame(local_perfs)
                df.to_csv(('local.csv'), index=False)

                # store settings
                with open(('settings_dict.txt'), 'w') as file:
                    [file.write(key + ' : ' + str(value) + '\n') for key, value in args.items()]

                with open(('settings_dict.pickle'), 'wb') as f: 
                    pickle.dump(args, f)