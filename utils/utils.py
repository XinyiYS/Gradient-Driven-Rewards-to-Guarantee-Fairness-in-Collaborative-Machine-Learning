import math
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data import Batch

import torch.nn.functional as F







def compute_grad_update(old_model, new_model, device=None):
	# maybe later to implement on selected layers/parameters
	if device:
		old_model, new_model = old_model.to(device), new_model.to(device)
	return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def add_update_to_model(model, update, weight=1.0, device=None):
	if not update: return model
	if device:
		model = model.to(device)
		update = [param.to(device) for param in update]
			
	for param_model, param_update in zip(model.parameters(), update):
		param_model.data += weight * param_update.data
	return model

def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
	assert len(grad_update_1) == len(
		grad_update_2), "Lengths of the two grad_updates not equal"
	
	for param_1, param_2 in zip(grad_update_1, grad_update_2):
		param_1.data += param_2.data * weight


def flatten(grad_update):
	return torch.cat([update.data.view(-1) for update in grad_update])


def unflatten(flattened, normal_shape):
	grad_update = []
	for param in normal_shape:
		n_params = len(param.view(-1))
		grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size())  )
		flattened = flattened[n_params:]

	return grad_update


def compute_distance_percentage(model, ref_model):
	percents, dists  = [], []
	for layer, ref_layer in zip(model.parameters(), ref_model.parameters()):
		dist = torch.norm(layer - ref_layer)
		dists.append(dist.item())
		percents.append( (torch.div(dist, torch.norm(ref_layer))).item() )

	return percents, dists



def cosine_similarity(grad1, grad2, normalized=False):
	"""
	Input: two sets of gradients of the same shape
	Output range: [-1, 1]
	"""

	cos_sim = F.cosine_similarity(flatten(grad1), flatten(grad2), 0, 1e-10) 
	if normalized:
		return (cos_sim + 1) / 2.0
	else:
		return cos_sim

def evaluate(model, eval_loader, device, loss_fn=None, verbose=False):
	model.eval()
	model = model.to(device)
	correct = 0
	total = 0
	loss = 0

	with torch.no_grad():
		for i, batch in enumerate(eval_loader):

			if isinstance(batch, Batch):
				batch_data, batch_target = batch.text, batch.label
				# batch_data.data.t_(), batch_target.data.sub_(1)  # batch first, index align
				batch_data = batch_data.permute(1, 0)
			else:
				batch_data, batch_target = batch[0], batch[1]

			batch_data, batch_target = batch_data.to(device), batch_target.to(device)
			outputs = model(batch_data)

			if loss_fn:
				loss += loss_fn(outputs, batch_target)
			else:
				loss = None
			correct += (torch.max(outputs, 1)[1].view(batch_target.size()).data == batch_target.data).sum()
			total += len(batch_target)

		accuracy =  correct.float() / total
		if loss_fn:
			loss /= total

	if verbose:
		print("Loss: {:.6f}. Accuracy: {:.4%}.".format(loss, accuracy))
	return loss, accuracy

from torchtext.data import Batch
def train_model(model, loader, loss_fn, optimizer, device, E=1, **kwargs):

	model.train()
	for e in range(E):
		# running local epochs
		for _, batch in enumerate(loader):
			if isinstance(batch, Batch):
				data, label = batch.text, batch.label
				data = data.permute(1, 0)
				# data.data.t_(), label.data.sub_(1)  # batch first, index align
			else:
				data, label = batch[0], batch[1]

			data, label = data.to(device), label.to(device)

			optimizer.zero_grad()
			pred = model(data)
			loss_fn(pred, label).backward()

			optimizer.step()

	if 'scheduler' in kwargs: kwargs['scheduler'].step()
	
	return model


def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):

	if mode == 'all':
		# mask all but the largest <mask_order> updates (by magnitude) to zero
		all_update_mod = torch.cat([update.data.view(-1).abs()
									for update in grad_update])
		if not mask_order and mask_percentile is not None:
			mask_order = int(len(all_update_mod) * mask_percentile)
		
		if mask_order == 0:
			return mask_grad_update_by_magnitude(grad_update, float('inf'))
		else:
			topk, indices = torch.topk(all_update_mod, mask_order)
			return mask_grad_update_by_magnitude(grad_update, topk[-1])

	elif mode == 'layer': # layer wise largest-values criterion
		grad_update = copy.deepcopy(grad_update)

		mask_percentile = max(0, mask_percentile)
		for i, layer in enumerate(grad_update):
			layer_mod = layer.data.view(-1).abs()
			if mask_percentile is not None:
				mask_order = math.ceil(len(layer_mod) * mask_percentile)

			if mask_order == 0:
				grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
			else:
				topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))																																												
				grad_update[i].data[layer.data.abs() < topk[-1]] = 0
		return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):

	# mask all but the updates with larger magnitude than <mask_constant> to zero
	# print('Masking all gradient updates with magnitude smaller than ', mask_constant)
	grad_update = copy.deepcopy(grad_update)
	for i, update in enumerate(grad_update):
		grad_update[i].data[update.data.abs() < mask_constant] = 0
	return grad_update


import os
from contextlib import contextmanager

@contextmanager
def cwd(path):
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


'''


def sign(grad):
	return [torch.sign(update) for update in grad]
def l2norm(grad):
	return torch.sqrt(torch.sum(torch.pow(flatten(grad), 2)))


def cosine_similarity_modified(coalition_grad, coalition_grad_majority, grad_all, grad_all_majority, normalized=False, Lambda=0):
	sign_cossim = F.cosine_similarity(coalition_grad_majority, grad_all_majority, 0, 1e-10) 
	modu_cossim = F.cosine_similarity(coalition_grad, grad_all, 0, 1e-10)

	return Lambda * sign_cossim  + (1 - Lambda) * modu_cossim

def mask_grad_update_by_indices(grad_update, indices=None):
	"""
	Mask the grad.data to be 0, if the position is not in the list of indices
	If indicies is empty, mask nothing.
	
	Arguments: 
	grad_update: as in the shape of the model parameters. A list of tensors.
	indices: a tensor of integers, corresponding to the specific individual scalar values in the grad_update, 
	as if the entire grad_update is flattened.

	e.g. 
	grad_update = [[1, 2, 3], [3, 2, 1]]
	indices = [4, 5]
	returning masked grad_update = [[0, 0, 0], [0, 2, 1]]
	"""

	grad_update = copy.deepcopy(grad_update)
	if indices is None or len(indices)==0: return grad_update

	#flatten and unflatten
	flattened = torch.cat([update.data.view(-1) for update in grad_update])	
	masked = torch.zeros_like(torch.arange(len(flattened)), device=flattened.device).float()
	masked.data[indices] = flattened.data[indices]

	pointer = 0
	for m, update in enumerate(grad_update):
		size_of_update = torch.prod(torch.tensor(update.shape)).long()
		grad_update[m].data = masked[pointer: pointer + size_of_update].reshape(update.shape)
		pointer += size_of_update
	return grad_update


from itertools import chain, combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

from math import factorial as f
def choose(n, r):
	return f(n) // f(r) // f(n-r)

def clip_gradient_update(grad_update, grad_clip):
	"""
	Return a copy of clipped grad update 

	"""
	return [torch.clamp(param.data, min=-grad_clip, max=grad_clip) for param in grad_update]

'''
