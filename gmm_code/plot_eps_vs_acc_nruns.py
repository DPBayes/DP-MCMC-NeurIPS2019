import numpy as np
import matplotlib.pyplot as plt

from exact_rdp import *
from plot_path import path

def plot_fig6(nondp_chain, dp_chains, dp_mcmc_params, dpvi_params):
	
	#####################################################
	burn_in = 1000

	## Compute original average and variance, after burn in
	true_mean = nondp_chain[burn_in:].mean(0)
	true_var= nondp_chain[burn_in:].var(0)

	## burn in DP-chains
	dp_chains_burned_in = dp_chains[:, burn_in:]


	#####################################################
	## Compute mean and variance errors
	T_after_burn = dp_chains_burned_in.shape[1]
	"""
	We have m repeats of the inference and each repeat produces a DP chain of lenght T with 2 params.
	Thus the dp_chains is a list of m Tx2 arrays. We want to compute the absolute error between running mean 
	and true posterior mean. We can write the running mean at iteration t e.g. for parameter theta_0 as 
	$\frac{1}{t}\sum_{i\leq t} theta_0_i$. Therefore we can efficiently compute the running mean for every t
	as running_mean_t = cumsum(theta_0)_t / t, where cumsum(theta_0)_t denotes the cumulative sum of theta_0
	chain until t.
	"""
	running_means = np.cumsum(dp_chains_burned_in, axis=1)/np.arange(1, T_after_burn+1)[np.newaxis,:, np.newaxis]
	mean_errors = np.abs(running_means-true_mean) ## This will be mxTx2
	mean_error_average = mean_errors.mean(0) ## average over m repeats, this will be Tx2
	mean_error_sem = mean_errors.std(0)/np.sqrt(len(dp_chains)) ## std. error of mean over runs

	"""
	We can use similar tricks as above to compute the running variance. We apply Var(X) = E[X^2]-E[X]^2
	to compute the variance.
	"""
	running_means_squared = running_means**2 ## E[X]^2
	running_squared_means = np.cumsum(np.power(dp_chains_burned_in, 2), axis=1)\
								/np.arange(1, T_after_burn+1)[np.newaxis,:, np.newaxis] ## E[X^2]
	var_errors = np.abs(running_squared_means-running_means_squared-true_var)
	var_error_average = var_errors.mean(0) ## average over m repeats
	var_error_sem = var_errors.std(0)/np.sqrt(len(dp_chains)) ## std. error of mean

	#####################################################
	## Pick epsilons to plot
	T = dp_mcmc_params['T']
	steps = np.arange(burn_in, dp_chains.shape[1], 50)+1

	mean_error_average = mean_error_average[steps-burn_in-1]
	mean_error_sem = mean_error_sem[steps-burn_in-1]

	var_error_average = var_error_average[steps-burn_in-1]
	var_error_sem = var_error_sem[steps-burn_in-1]

	## Compute privacy cost
	b = dp_mcmc_params['B']
	N = dp_mcmc_params['N']
	dpvi_eps = dpvi_params['eps']
	Ts = steps
	max_alpha = 100
	max_alpha = min(max_alpha,b//5)
	delta = 1/N
	min_eps_T = np.inf*np.ones(len(Ts))
	q = b/N
	for max_alpha_ in range(3, max_alpha):
		eps_alpha_list = [rd_approx(alpha, b) for alpha in range(2, max_alpha_+1)]
		amplified_eps = amplified_RDP(eps_alpha_list, max_alpha_, q)
		total_eps = [from_RDP_to_DP(T*amplified_eps, max_alpha_, delta) for T in Ts]
		min_eps_T = np.minimum(min_eps_T, total_eps)
	min_eps_T = min_eps_T + dpvi_eps

	#####################################################

	## Plot mean error
	import matplotlib.pyplot as plt
	plt.cla()
	theta_0_mean_error_average =  mean_error_average[:,0]
	theta_1_mean_error_average =  mean_error_average[:,1]
	## clip errorbars to avoid negatives
	theta_0_mean_error_sem_low = np.minimum(theta_0_mean_error_average, mean_error_sem[:, 0])
	theta_0_mean_error_sem = np.array([theta_0_mean_error_sem_low, mean_error_sem[:, 0]])
	theta_1_mean_error_sem_low = np.minimum(theta_1_mean_error_average, mean_error_sem[:, 1])
	theta_1_mean_error_sem = np.array([theta_1_mean_error_sem_low, mean_error_sem[:, 1]])
	plt.errorbar(min_eps_T, theta_0_mean_error_average, yerr=theta_0_mean_error_sem, alpha = 0.1,\
			color='red')
	plt.errorbar(min_eps_T, theta_1_mean_error_average, yerr=theta_1_mean_error_sem, alpha = 0.1,\
			color='blue')
	plt.plot(min_eps_T, theta_0_mean_error_average, color='red', label=r'$\theta_1$')
	plt.plot(min_eps_T, theta_1_mean_error_average, color='blue', label=r'$\theta_2$')
	plt.legend(loc='best')
	plt.xlabel(r'$\epsilon$', fontsize=16)
	plt.ylabel(r'$|\mu_{true}-\mu_{DP}|$', fontsize=16)
	plt.savefig(path+'eps_vs_mean_acc.pdf',format='pdf')
	plt.close()

	## Plot variance error
	theta_0_var_error_average =  var_error_average[:,0]
	theta_1_var_error_average =  var_error_average[:,1]
	## clip errorbars to avoid negatives
	theta_0_var_error_sem_low = np.minimum(theta_0_var_error_average, var_error_sem[:, 0])
	theta_0_var_error_sem = np.array([theta_0_var_error_sem_low, var_error_sem[:, 0]])
	theta_1_var_error_sem_low = np.minimum(theta_1_var_error_average, var_error_sem[:, 1])
	theta_1_var_error_sem = np.array([theta_1_var_error_sem_low, var_error_sem[:, 1]])

	plt.cla()
	plt.errorbar(min_eps_T, theta_0_var_error_average, yerr=theta_0_var_error_sem, alpha = 0.1,\
			color='red')
	plt.errorbar(min_eps_T, theta_1_var_error_average, yerr=theta_1_var_error_sem, alpha = 0.1,\
			color='blue')
	plt.plot(min_eps_T, theta_0_var_error_average, color='red', label=r'$\theta_1$')
	plt.plot(min_eps_T, theta_1_var_error_average, color='blue', label=r'$\theta_2$')
	plt.legend(loc='best')
	plt.xlabel(r'$\epsilon$', fontsize=16)
	plt.ylabel(r'$|\sigma^2_{true}-\sigma^2_{DP}|$', fontsize=16)
	plt.savefig(path+'eps_vs_var_acc.pdf',format='pdf')
	plt.close()
