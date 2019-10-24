import numpy as np
import pickle
import matplotlib.pyplot as plt
from plot_path import path

def plot_fig3(fname):
	fontsize = 25
	figsize = (8.5,6)
	temped_results = pickle.load(open(fname, 'rb'))
	temped_dp_mcmc_params = temped_results[0]
	temped_chain = temped_results[1]
	temped_privacy_params = temped_results[2]

	# Plot clipped proportion vs. proposal variance
	prop_vars = temped_dp_mcmc_params['prop_vars']
	clip_counts = temped_dp_mcmc_params['clip_counts']
	T = temped_dp_mcmc_params['T']
	batch_size = temped_dp_mcmc_params['B']
	plt.figure(figsize=figsize)
	plt.plot(prop_vars, clip_counts.sum(1)/T/batch_size)
	plt.title('Average proportion of \n clipped llr vs. proposal variance', fontsize=fontsize)
	plt.xlabel(r'$\sigma^2$', fontsize=fontsize)
	plt.ylabel(r'$\frac{\#(clipped)}{Tb}$', fontsize=fontsize)
	plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize-1, rotation=45)
	plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize-1)
	plt.tight_layout()
	plt.savefig(path+'prop_vs_clip.pdf',format='pdf', bbox_inches = 'tight')
	plt.close()
