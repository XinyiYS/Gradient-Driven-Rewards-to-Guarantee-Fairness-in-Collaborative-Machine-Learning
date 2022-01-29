from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

from utils.utils import choose


from collections import defaultdict
import pandas as pd


import numpy as np
import torch
import torch.nn.functional as F
cos = F.cosine_similarity

N = 3
M = 5
a = torch.arange(1, 1+M)
b = 2 * a
c = torch.div(1, torch.arange(1, 1+M))
d = torch.square(a)


from torch.linalg import norm
def v(coalition_v, grand_v, if_norm=False):
	coalition_v_ = torch.div(coalition_v, norm(coalition_v)) if (if_norm and norm(coalition_v) != 0) else coalition_v
	grand_v_ = torch.div(grand_v, norm(grand_v)) if (if_norm and norm(grand_v) != 0) else grand_v
	return cos(coalition_v_, grand_v_, 0)

from math import factorial as fac


def calculate_svs(vectors, N, d):
	grand = torch.stack(vectors).sum(dim=0)
	svs = torch.zeros(N)
	for coalition in powerset(range(N)):
		if not coalition: continue
		coalition_v = torch.zeros(d)
		for i in coalition:
			coalition_v += vectors[i]
		for i in coalition:
			with_i = v(coalition_v, grand)
			without_i = v(coalition_v - vectors[i], grand)

			svs[i] += 1.0 / choose(N-1, len(coalition)-1) * (with_i - without_i)
	return torch.div(svs, sum(svs))

from itertools import permutations
from random import shuffle
def calculate_sv_hats(vectors, N, d, K=30):
	grand = torch.stack(vectors).sum(dim=0)
	svs = torch.zeros(N)
	all_permutations = list(permutations(range(N)))
	shuffle(all_permutations)

	for permutation in all_permutations[:K]:
		permutation_v = torch.zeros(d)
		for i in permutation:
			without_i = v(permutation_v, grand) 
			permutation_v += vectors[i]
			with_i = v(permutation_v, grand) 
			svs[i] += with_i - without_i
	return torch.div(svs, sum(svs))


from scipy.stats import pearsonr

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

trials = 50
N = 10
d = 1000


from time import process_time

def clock(name, start, time_dict):
	now = process_time() 
	time_dict[name] += now - start
	return now

results = defaultdict(list)

def generate_random_vectors(N, d, uniform=True):

	if uniform:
		return  [torch.randn(d) for i in range(N)]
	else:
		vectors = []
		for i in range(N):
			rand_v = torch.zeros(d)
			for j in range(d):
				if j == 0:
					rand_v[j] = j 
				else:
					rand_v[j] = (torch.randn(1) * rand_v[j-1])**2 + j
			rand_v += torch.randn(d)
			vectors.append(torch.div(rand_v, norm(rand_v))  )
		return vectors

def experiment(N, d, trials=50, epsilon=0.1, delta=0.1):
	time_dict = defaultdict(float)
	now = process_time()
	K = int(np.ceil( np.log(2.0/delta) * 1**2 / (2.0 * epsilon **2)  ))

	# print("For epsilon {}, delta {}, sampling method needs {} samples.".format(epsilon, delta, K))
	for i in range(trials):
		vectors = generate_random_vectors(N, d, True)
		grand = torch.stack(vectors).sum(dim=0)
		now = clock('init', now, time_dict)

		svs = calculate_svs(vectors, N, d)
		now = clock('true svs', now, time_dict)

		sv_hats = calculate_sv_hats(vectors, N, d, K)
		now = clock('sv hats', now, time_dict)

		cosines = torch.tensor([cos(v, grand, 0) for v in vectors])
		cosines = torch.div(cosines, sum(cosines))
		now = clock('cosines', now, time_dict)

		results['svs'].append(svs)
		results['sv_hats'].append(sv_hats)
		diff_cos = cosines - svs
		results['diff'].append(diff_cos)
		results['l1diff'].append(sum(np.abs(diff_cos)).item() )
		results['l2diff'].append(norm(diff_cos).item())
		results['cossim'].append( cos(cosines, svs, 0).item())
		r, p = pearsonr(cosines, svs)
		results['pearsonr'].append(r)
		results['pearsonp'].append(p)

		diff_hat = svs - sv_hats
		results['diff_hat'].append(diff_hat)
		results['l1diff_hat'].append(sum(np.abs(diff_hat)).item() )
		results['l2diff_hat'].append(norm(diff_hat).item())
		results['cossim_hat'].append( cos(sv_hats, svs, 0).item())
		r, p = pearsonr(sv_hats, svs)
		results['pearsonr_hat'].append(r)
		results['pearsonp_hat'].append(p)
		now = clock('results', now, time_dict)
	
	return results, time_dict

import matplotlib.pyplot as plt

trials = 10


# Experiment vs N
Nmin, Nmax = 5, 10
d = 1000

stats_dict = defaultdict(list)
for n in range(5, Nmax+1):

	results, time_dict = experiment(n, d, trials=trials)
	df = pd.DataFrame(results, columns=['l1diff', 'l1diff_hat', 'l2diff', 'l2diff_hat', 'pearsonr', 'pearsonr_hat'])

	stats_dict['n'].append(n)
	for column in df.columns:
		stats_dict[column+'_mean'].append(df[column].mean())
		stats_dict[column+'_std'].append(df[column].std())

	stats_dict['cosines'].append(time_dict['cosines'] / trials)
	stats_dict['sv hats'].append(time_dict['sv hats'] / trials)

stats_df = pd.DataFrame(stats_dict)
stats_df.to_csv('error_vs_N={}-{}.csv'.format(Nmin, Nmax), index=False)


# Experiment vs d
dmin, dmax = 10, 15
n = 10

stats_dict = defaultdict(list)
for d in range(10, dmax+1):
	d = 2**d
	results, time_dict = experiment(n, d, trials=trials)
	df = pd.DataFrame(results, columns=['l1diff', 'l1diff_hat', 'l2diff', 'l2diff_hat', 'pearsonr', 'pearsonr_hat'])

	stats_dict['d'].append(d)
	for column in df.columns:
		stats_dict[column+'_mean'].append(df[column].mean())
		stats_dict[column+'_std'].append(df[column].std())

	stats_dict['cosines'].append(time_dict['cosines'] / trials)
	stats_dict['sv hats'].append(time_dict['sv hats'] / trials)

stats_df = pd.DataFrame(stats_dict)
stats_df.to_csv('error_vs_d={}-{}.csv'.format(2**dmin, 2**dmax), index=False)

exit()



data = defaultdict(list)

for coalition in powerset(range(N)):
	if not coalition: continue
	coalition_v = torch.zeros(M)
	for i in coalition:
		coalition_v += vectors[i]
	data['coalition'].append(coalition)
	data['utility'].append(cos(grand, coalition_v, 0).item() )
	data['sum of svs'].append(sum([svs[i] for i in coalition]).item()  )

df = pd.DataFrame(data)
df['utility_left_over'] = df['utility'] - df['sum of svs']
df['efficient'] = df['utility_left_over'] == 0
print(df)
