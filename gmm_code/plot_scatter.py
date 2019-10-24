import numpy as np
import matplotlib.pyplot as plt
from plot_path import path

def plot_fig5(nondp_chain, dp_chain, dp_mcmc_params, dpvi_params):

	# Privacy budgets
	from exact_rdp import get_privacy_spent
	dp_eps_delta = get_privacy_spent(dp_mcmc_params['B'], dp_mcmc_params['N'],\
										   dp_mcmc_params['T'], max_alpha = 100)
	dp_eps_delta = np.array(dp_eps_delta) +\
						np.array([dpvi_params['eps'], dpvi_params['delta']])

	burn_in = 1000
	# Plot results
	plt.cla()
	plt.scatter(nondp_chain[burn_in:,0], nondp_chain[burn_in:,1], alpha=0.1)
	plt.xlim(-2.5, 2.5)
	plt.ylim(-2.5, 2.6)
	plt.title('Samples from target distribution, tempered likelihoods\n'+\
			'Non-DP')
	plt.savefig(path+'NONDP-scatter_plot.pdf',format='pdf')
	plt.close()
	
	plt.cla()
	plt.scatter(dp_chain[burn_in:,0], dp_chain[burn_in:,1], alpha=0.1)
	plt.xlim(-2.5, 2.5)
	plt.ylim(-2.5, 2.6)
	plt.title('Samples from target distribution, tempered likelihoods\n'+\
			'({0},{1})-DP'.format(np.round(dp_eps_delta[0],2), dp_eps_delta[1]))
	plt.savefig(path+'DP-scatter_plot.pdf',format='pdf')
	plt.close()

