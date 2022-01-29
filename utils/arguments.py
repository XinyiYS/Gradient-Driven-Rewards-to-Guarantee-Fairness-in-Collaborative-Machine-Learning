import torch
from torch import nn, optim

from utils.defined_models import CNN_Net, CNN_Text, CNN_Cifar10


mnist_args = {

	# setting parameters
	'dataset': 'mnist',
	'sample_size_cap': 6000,
	'split': 'powerlaw', #or 'classimbalance'

	'batch_size' : 32, 
	'train_val_split_ratio': 0.9,
	'lambda': 0.5, 
	'alpha': 0.95,
	'Gamma': 0.5,
	'lambda': 0, # coefficient between sign_cossim and modu_cossim

	# model parameters
	'model_fn': CNN_Net,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'lr': 0.15,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

}


sst_args = {

	# setting parameters
	'dataset': 'sst',
	'sample_size_cap': 5000,
	'split': 'powerlaw', #or 'powerlaw' classimbalance
	'batch_size' : 256, 

	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'Gamma': 1,
	'lambda': 1, # coefficient between sign_cossim and modu_cossim


	# model parameters
	'model_fn': CNN_Text,
	'embed_num': 20000,
	'embed_dim': 300,
	'class_num': 5,
	'kernel_num': 128,
	'kernel_sizes': [3,3,3],
	'static':False,

	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss(), 
	'lr': 1e-4,
	# 'grad_clip':1e-3,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1
}


mr_args = {

	# setting parameters
	'dataset': 'mr',

	'batch_size' : 128, 
	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'lambda': 0.5, # coefficient between sign_cossim and modu_cossim
	'Gamma':1,

	# model parameters
	'model_fn': CNN_Text,
	'embed_num': 20000,
	'embed_dim': 300,
	'class_num': 2,
	'kernel_num': 128,
	'kernel_sizes': [3,3,3],
	'static':False,

	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss(), 
	'lr': 5e-5,
	# 'grad_clip':1e-3,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

}


cifar_cnn_args = {

	# setting parameters
	'dataset': 'cifar10',

	'batch_size' : 128, 
	'train_val_split_ratio': 0.8,
	'alpha': 0.95,
	'Gamma': 0.15,
	'lambda': 0.5, # coefficient between sign_cossim and modu_cossim

	# model parameters
	'model_fn': CNN_Cifar10,
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(),
	'lr': 0.015,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1


}

