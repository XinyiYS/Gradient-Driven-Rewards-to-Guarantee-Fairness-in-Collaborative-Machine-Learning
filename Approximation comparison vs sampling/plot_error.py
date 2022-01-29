import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


vs_D_df = pd.read_csv('error_vs_d=1024-32768.csv')
vs_N_df = pd.read_csv('error_vs_N=5-10.csv')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-', '.-']

rename_dict = {'l1diff':'$||\\mathbb{{\\phi}} -\\mathbb{\\psi}||_{1}  $',
			'l1diff_hat':'$||\\mathbb{{\\phi}} -\\mathbb{\\bar{\\phi}}||_{1}$',
			 'l2diff':'$||\\mathbb{{\\phi}} -\\mathbb{\\psi}||_{2}$',
			'l2diff_hat':'$||\\mathbb{{\\phi}} -\\mathbb{\\bar{\\phi}}||_{2}$',}

def plot(df, vs_N=True):
	index_col = 'n' if vs_N else 'd'

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	x = df[index_col]
	for key in df.columns:
		if key == index_col: continue
		value = df[key]
				
		linestyle = '--' if 'hat' in key else '-'

		if 'l1' in key: color = colors[0]
		elif 'l2' in key: color = colors[1]
		else: color = colors[2]

		if key in ['cosines', 'sv hats']:
			key = key.replace('cosines', 'time $\\mathbb{\\psi}$')
			key = key.replace('sv hats', 'time $\\mathbb{\\bar{\\phi}}$')
			ax2.plot(x, value, linestyle=linestyle, color=color, label=key, linewidth=6)
		elif 'diff' in key and '_mean' in key:
			key = key.replace('_mean', '')
			key = rename_dict[key]
			ax1.plot(x, value, linestyle=linestyle, color=color, label=key, linewidth=6.0)

	# ask matplotlib for the plotted objects and their labels
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	if vs_N:
		ax1.legend(lines + lines2, labels + labels2, loc=2, fontsize=20)

		# ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)


	x_label = '$N$ agents' if vs_N else '$D$'
	ax1.set_xlabel(x_label, fontsize=28)
	if vs_N:
		ax1.set_ylabel('Error', fontsize=28)
	if not vs_N:
		ax2.set_ylabel('Time (seconds)', fontsize=28)
	
	# if not vs_N:
		# ax2.set_xscale('log')
		# ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

	ax1.tick_params(axis="x", labelsize=28)
	ax1.tick_params(axis="y", labelsize=28)
	ax2.tick_params(axis="y", labelsize=28)

	plt.tight_layout()

	save = True
	if save:
		figname = 'vs_N' if vs_N else 'vs_D'

		plt.savefig(figname+'.png')
		plt.clf()
	else:
		plt.show()

	
plot(vs_N_df)
plot(vs_D_df, False)
