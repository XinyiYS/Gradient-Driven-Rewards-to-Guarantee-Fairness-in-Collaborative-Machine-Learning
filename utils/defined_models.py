import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# for MNIST 32*32
class CNN_Net(nn.Module):

	def __init__(self, device=None):
		super(CNN_Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, 3, 1)
		self.conv2 = nn.Conv2d(64, 16, 7, 1)
		self.fc1 = nn.Linear(4 * 4 * 16, 200)
		self.fc2 = nn.Linear(200, 10)

	def forward(self, x):
		x = x.view(-1, 1, 32, 32)
		x = torch.tanh(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = torch.tanh(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4 * 4 * 16)
		x = torch.tanh(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)



class CNN_Cifar10(nn.Module):
	def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
		super(CNN_Cifar10, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
		self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, out_dim)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(x.shape[0], -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# return x
		return F.log_softmax(x, dim=1)


class CNN_Text(nn.Module):
	
	def __init__(self, args=None, device=None):
		super(CNN_Text,self).__init__()

		
		self.args = args
		self.device = device
		
		V = args['embed_num']
		D = args['embed_dim']
		C = args['class_num']
		Ci = 1
		Co = args['kernel_num']
		Ks = args['kernel_sizes']

		self.embed = nn.Embedding(V, D)
		self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
		'''
		self.conv13 = nn.Conv2d(Ci, Co, (3, D))
		self.conv14 = nn.Conv2d(Ci, Co, (4, D))
		self.conv15 = nn.Conv2d(Ci, Co, (5, D))
		'''
		self.dropout = nn.Dropout(0.5)
		# self.dropout = nn.Dropout(args.dropout)
		self.fc1 = nn.Linear(len(Ks)*Co, C)

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x


	def forward(self, x):

		x = self.embed(x) # (W,N,D)
		# x = x.permute(1,0,2) # -> (N,W,D)
		# permute during loading the batches instead of in the forward function
		# in order to allow nn.DataParallel

		if not self.args or self.args['static']:
			x = Variable(x).to(self.device)

		x = x.unsqueeze(1) # (W,Ci,N,D)

		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
		x = torch.cat(x, 1)
		'''
		x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
		x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
		x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
		x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
		'''
		x = self.dropout(x) # (N,len(Ks)*Co)
		logit = self.fc1(x) # (N,C)
		return F.log_softmax(logit, dim=1)
		# return logit