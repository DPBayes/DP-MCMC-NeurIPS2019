import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from exact_rdp import *
from plot_path import path

def compute_running_means(chains):
	"""
	We have m repeats of the inference and each repeat produces a DP chain of lenght T with 2 params.
	Thus the dp_chains is a list of m Tx2 arrays. We want to compute the absolute error between running mean 
	and true posterior mean. We can write the running mean at iteration t e.g. for parameter theta_0 as 
	$\frac{1}{t}\sum_{i\leq t} theta_0_i$. Therefore we can efficiently compute the running mean for every t
	as running_mean_t = cumsum(theta_0)_t / t, where cumsum(theta_0)_t denotes the cumulative sum of theta_0
	chain until t.
	"""
	T = chains.shape[1]
	return np.cumsum(chains, axis=1)/np.arange(1, T+1)[np.newaxis,:, np.newaxis]

def compute_running_variances(chains, running_means):
	"""
	We can use similar tricks as above to compute the running variance. We apply Var(X) = E[X^2]-E[X]^2
	to compute the variance.
	"""
	T = chains.shape[1]
	running_means_squared = running_means**2 ## E[X]^2
	running_squared_means = np.cumsum(np.power(chains, 2), axis=1)/\
						np.arange(1, T+1)[np.newaxis,:, np.newaxis] ## E[X^2]
	return running_squared_means-running_means_squared

def compute_running_stats(dp_chains, burn_in):
	## burn in DP-chains
	dp_chains_burned_in = dp_chains[:, burn_in:]


	#####################################################
	## Compute means and variances
	running_means = compute_running_means(dp_chains_burned_in)
	running_variances = compute_running_variances(dp_chains_burned_in, running_means)
	return running_means, running_variances

def compute_privacy_cost_mcmc(dp_mcmc_params, dpvi_params, Ts):
	"""
	dp_mcmc_params : a dictionary containing parameters of DP-MCMC
	Ts : time steps on which to evaluate privacy cost
	"""
	b = dp_mcmc_params['B']
	N = dp_mcmc_params['N']
	dpvi_eps = dpvi_params['eps']
	dpvi_delta = dpvi_params['delta']
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
	return min_eps_T, delta+dpvi_delta

def compute_privacy_cost_sgld(dp_sgld_params, Ts):
	from privacy.analysis.compute_dp_sgd_privacy import compute_rdp, get_privacy_spent
	## Compute privacy budget
	N = 1e6
	batch_size = dp_sgld_params['tau']
	noise_sigma = dp_sgld_params['noise_sigma']
	q = batch_size/N
	delta = 2/N
	rdp_orders = range(2, 500)
	rdp_eps0 = compute_rdp(q, noise_sigma, 1, rdp_orders)
	epsilons = np.zeros(len(Ts))
	for i_T, T in enumerate(Ts):
		epsilons[i_T] = get_privacy_spent(rdp_orders, rdp_eps0*T, target_delta=delta)[0]
	return epsilons, delta

######################################################################################################

## DP MCMC results
dp_mcmc_ext = '15_10'
dp_mcmc_res = pd.read_pickle('./results/dp_mcmc_n_runs_results_temped_{}.p'.format(dp_mcmc_ext))
dp_mcmc_chains = np.array([res[1] for res in dp_mcmc_res])
dp_mcmc_params = dp_mcmc_res[0][0]
dpvi_params = pd.read_pickle('./results/dpvi_params.p')

burn_in_mcmc = 1000
running_means_mcmc, running_variances_mcmc = compute_running_stats(dp_mcmc_chains, burn_in_mcmc)

## DP-SGLD results
dp_sgld_chains = np.array([pd.read_pickle('./dp_sgld/results/dp_sgld_res_seed_{}_10_10.p'.format(seed))[0]\
		for seed in range(123,143)])
dp_sgld_params = pd.read_pickle('./dp_sgld/results/dp_sgld_res_seed_{}_10_10.p'.format(123))[1]
burn_in_sgld = int(1e5)
running_means_sgld, running_variances_sgld = compute_running_stats(dp_sgld_chains, burn_in_sgld)

######################################################################################################

## Compute errors
## Compute original average and variance, after burn in
nondp_ext = '16_10'
nondp_chain = pd.read_pickle('./results/non_dp_mcmc_results_temped_{}.p'.format(nondp_ext))[1]

true_mean = nondp_chain[burn_in_mcmc:].mean(0)
true_var= nondp_chain[burn_in_mcmc:].var(0)
## DP-MCMC
Ts_mcmc = np.arange(0, dp_mcmc_params['T']-burn_in_mcmc, 30)

mean_errors_mcmc = np.abs(running_means_mcmc[:,Ts_mcmc]-true_mean) ## This will be mxTx2
mean_error_average_mcmc = mean_errors_mcmc.mean(0) ## average over m repeats, this will be Tx2
mean_error_sem_mcmc = mean_errors_mcmc.std(0)/np.sqrt(len(dp_mcmc_chains)) ## std. error of mean over runs

var_errors_mcmc = np.abs(running_variances_mcmc[:,Ts_mcmc]-true_var) ## This will be mxTx2
var_error_average_mcmc = var_errors_mcmc.mean(0) ## average over m repeats, this will be Tx2
var_error_sem_mcmc = var_errors_mcmc.std(0)/np.sqrt(len(dp_mcmc_chains)) ## std. error of var over runs

## DP-SGLD
Ts_sgld = np.arange(0, dp_sgld_params['T']-burn_in_sgld, 10000)

mean_errors_sgld = np.abs(running_means_sgld[:,Ts_sgld]-true_mean) ## This will be mxTx2
mean_error_average_sgld = mean_errors_sgld.mean(0) ## average over m repeats, this will be Tx2
mean_error_sem_sgld = mean_errors_sgld.std(0)/np.sqrt(len(dp_sgld_chains)) ## std. error of mean over runs

var_errors_sgld = np.abs(running_variances_sgld[:,Ts_sgld]-true_var) ## This will be mxTx2
var_error_average_sgld = var_errors_sgld.mean(0) ## average over m repeats, this will be Tx2
var_error_sem_sgld = var_errors_sgld.std(0)/np.sqrt(len(dp_sgld_chains)) ## std. error of var over runs

######################################################################################################

## Compute privacy costs 
# DP-MCMC
epsilons_mcmc, delta_mcmc = compute_privacy_cost_mcmc(dp_mcmc_params, dpvi_params, Ts_mcmc+burn_in_mcmc)
# DP-SGLD
epsilons_sgld, delta_sgld = compute_privacy_cost_sgld(dp_sgld_params, Ts_sgld+burn_in_sgld)

#####################################################

## Plot mean error
plt.cla()
## DP-MCMC
plt.errorbar(epsilons_mcmc, mean_error_average_mcmc[:,0], yerr=mean_error_sem_mcmc[:,0], alpha = 0.1,\
		color='red')
plt.errorbar(epsilons_mcmc, mean_error_average_mcmc[:,1], yerr=mean_error_sem_mcmc[:,1], alpha = 0.1,\
		color='blue')
plt.plot(epsilons_mcmc, mean_error_average_mcmc[:,0], color='red', label=r'$\theta_1$, DP-MCMC')
plt.plot(epsilons_mcmc, mean_error_average_mcmc[:,1], color='blue', label=r'$\theta_2$, DP-MCMC')
## DP-SGLD
plt.errorbar(epsilons_sgld, mean_error_average_sgld[:,0], yerr=mean_error_sem_sgld[:,0], alpha = 0.1,\
		color='red')
plt.errorbar(epsilons_sgld, mean_error_average_sgld[:,1], yerr=mean_error_sem_sgld[:,1], alpha = 0.1,\
		color='blue')
plt.plot(epsilons_sgld, mean_error_average_sgld[:,0], color='red', linestyle='--', label=r'$\theta_1$, DP-SGLD')
plt.plot(epsilons_sgld, mean_error_average_sgld[:,1], color='blue', linestyle='--', label=r'$\theta_2$, DP-SGLD')

plt.legend(loc='best')
plt.xlim(0.56, 1.22)
plt.xlabel(r'$\epsilon$', fontsize=16)
plt.ylabel(r'$|\mu_{true}-\mu_{DP}|$', fontsize=16)
#plt.show()
plt.savefig(path+'mcmc_vs_sgld_mean_acc.pdf',format='pdf')
plt.close()


## Plot var error
plt.cla()
## DP-MCMC
plt.errorbar(epsilons_mcmc, var_error_average_mcmc[:,0], yerr=var_error_sem_mcmc[:,0], alpha = 0.1,\
		color='red')
plt.errorbar(epsilons_mcmc, var_error_average_mcmc[:,1], yerr=var_error_sem_mcmc[:,1], alpha = 0.1,\
		color='blue')
plt.plot(epsilons_mcmc, var_error_average_mcmc[:,0], color='red', label=r'$\theta_1$, DP-MCMC')
plt.plot(epsilons_mcmc, var_error_average_mcmc[:,1], color='blue', label=r'$\theta_2$, DP-MCMC')
## DP-SGLD
plt.errorbar(epsilons_sgld, var_error_average_sgld[:,0], yerr=var_error_sem_sgld[:,0], alpha = 0.1,\
		color='red')
plt.errorbar(epsilons_sgld, var_error_average_sgld[:,1], yerr=var_error_sem_sgld[:,1], alpha = 0.1,\
		color='blue')
plt.plot(epsilons_sgld, var_error_average_sgld[:,0], color='red', linestyle='--', label=r'$\theta_1$, DP-SGLD')
plt.plot(epsilons_sgld, var_error_average_sgld[:,1], color='blue', linestyle='--', label=r'$\theta_2$, DP-SGLD')

plt.legend(loc='best')
plt.xlim(0.56, 1.22)
plt.xlabel(r'$\epsilon$', fontsize=16)
plt.ylabel(r'$|\sigma^2_{true}-\sigma^2_{DP}|$', fontsize=16)
#plt.show()
plt.savefig(path+'mcmc_vs_sgld_var_acc.pdf',format='pdf')
plt.close()
