"""
This file contains all experiments of the main paper, except DP-SGLD comparison which can be found
in separate folder labeled dp_sgld
"""
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import pickle, datetime, random

import X_corr

## Following booleans set whether to run experiments from scratch or read parameters from files
generate_data = True
learn_start_with_dpvi = True
learn_X_corr = True
draw_non_dp_chain = True
draw_dp_chain = True
learn_multi_var = True

# filename extensions if not from scratch
nondp_ext = '15_10'
dp_ext = '15_10'
prop_var_ext = '15_10'

## Generate toy data from Gaussian mixture
if generate_data :
	## Seed numpy for generating data
	npr.seed(123)
	## Draw samples
	N = 1000000
	data = np.zeros((N))
	thetas = [0,1] # theta_1, theta_2
	for i in range(N):
		if npr.rand()>0.5:
			data[i] = npr.normal(thetas[0], np.sqrt(2))
		else:
			data[i] = npr.normal(np.sum(thetas), np.sqrt(2))

	# Save data 
	data = pd.DataFrame(data)
	data.to_csv('./data/MoG_toy_data.csv', index=False)

else : 
	# Read data
	data = pd.read_csv('./data/MoG_toy_data.csv').values.squeeze()
	N = len(data)

## Init starting point using DPVI
# Seed DPVI
import torch
from moments_accountant import ma
from dpvi import dpvi_mix_gaus
if learn_start_with_dpvi :
	torch.manual_seed(123) # DPVI uses pytorch randomness
	npr.seed(123)
	k_dpvi = 2
	batch_size_dpvi = 100
	noise_sigma_dpvi = 10
	T_dpvi = 1000
	clip_threshold = 1.0
	learning_rate = 0.001
	params_0_dpvi = 2*npr.randn(4)
	dpvi_delta = 1e-6
	dpvi_eps_cost = ma(noise_sigma_dpvi, batch_size_dpvi/N, T_dpvi, dpvi_delta)
	print('Initializing parameters with DPVI, privacy_cost : {}'.format(dpvi_eps_cost))
	params, params_0 =  dpvi_mix_gaus(data, k_dpvi, params_0_dpvi, T_dpvi, batch_size_dpvi,\
									clip_threshold, noise_sigma_dpvi, learning_rate)
	print('Initialization done!')
	theta_from_dpvi = params[:k_dpvi]
	pickle.dump(theta_from_dpvi, open('./results/theta_from_dpvi.p', 'wb'))
	dpvi_params = {'eps': dpvi_eps_cost, 'delta': dpvi_delta}
	pickle.dump(dpvi_params, open('./results/dpvi_params.p', 'wb'))

else :
	theta_from_dpvi = pickle.load(open('./results/theta_from_dpvi.p', 'rb'))
	dpvi_params = pickle.load(open('./results/dpvi_params.p', 'rb'))
	dpvi_eps_cost = dpvi_params['eps']
	dpvi_delta = dpvi_params['delta']

## Learn X_corr distributions
if learn_X_corr:
	from X_corr import get_x_corr_params
	import torch
	torch.manual_seed(123) ## we use pytorch to learn GMM and initialization is done randomly
	# For non-dp C=1.0
	non_dp_X_corr_params = get_x_corr_params(10, 1000, 1.0, K=50, lr=1e-2, T=10000,\
									path_to_file=None, symmetric=True, early_stop=-1)
	# For dp C=2.0
	non_dp_X_corr_params = get_x_corr_params(10, 1000, 2.0, K=50, lr=1e-2, T=10000,\
									path_to_file=None, symmetric=True, early_stop=-1)



## Non-DP chain
if draw_non_dp_chain : 
	npr.seed(123) # seed for numpy random, correction noise and such
	random.seed(123) # seed for subsampling
	nondp_T = 40000 # Number of non-DP draws

	from mog_main import run_chain
	print("Running NON-DP chain")
	(nondp_chain, nondp_n_accepted, nondp_sample_vars), [nondp_mcmc_params, nondp_privacy_pars], nondp_fname = \
		run_chain(data, nondp_T, theta_from_dpvi, privacy=False)
	nondp_to_pickle = [nondp_mcmc_params, nondp_chain, nondp_privacy_pars]
	## Save results
	pickle.dump(nondp_to_pickle, open(nondp_fname, 'wb'))


else :
	nondp_to_pickle = pickle.load(open('./results/non_dp_mcmc_results_temped_{}.p'.format(nondp_ext), 'rb'))
	[nondp_mcmc_params, nondp_chain, nondp_privacy_pars] = nondp_to_pickle

## DP chains
if draw_dp_chain : 
	npr.seed(123) # seed for numpy random, correction noise and such
	random.seed(123) # seed for subsampling
	dp_T = 20000 # Number of DP draws
	n_runs = 20 # number of repeats

	from mog_main import run_chain
	n_runs_res = []
	for n_run in range(n_runs):
		print("Running DP chain, repeat {}".format(n_run))
		(dp_chain, dp_n_accepted, dp_sample_vars), [dp_mcmc_params, dp_privacy_pars], dp_fname = \
			run_chain(data, dp_T, theta_from_dpvi, privacy=True)
		dp_to_pickle = [dp_mcmc_params, dp_chain, dp_privacy_pars]
		n_runs_res.append(dp_to_pickle)
		if n_run == 0:
			n_runs_name = dp_fname
	## Save results
	dp_fname = n_runs_name
	dp_fname_ext_indx = dp_fname.index('_results_')
	dp_fname = dp_fname[:dp_fname_ext_indx] + '_n_runs' + dp_fname[dp_fname_ext_indx:]
	pickle.dump(n_runs_res, open(dp_fname, 'wb'))


else :
	n_runs_res = pickle.load(open('./results/dp_mcmc_n_runs_results_temped_{}.p'.format(dp_ext), 'rb'))
	dp_mcmc_params = n_runs_res[0][0]


## Chains with multiple proposal variances
if learn_multi_var:
	from mog_main_prop_var import run_chain_multiple_prop_var
	print("Running DP chain for multiple proposal variances")
	npr.seed(123) # seed for numpy random, correction noise and such
	random.seed(123) # seed for subsampling
	dp_T = 20000
	fname_multi_vars = run_chain_multiple_prop_var(data, dp_T, theta_from_dpvi)
	[dp_mcmc_params_multi_var, theta_chain_multi_var, privacy_pars_multi_var] = \
			pickle.load(open(fname_multi_vars, 'rb'))
else : 
	fname_multi_vars = './results/dp_mcmc_results_temped_multiple_prop_vars_{}.p'.format(prop_var_ext)

####################################################################
###########  PLOT RESULTS #############

dp_chains = np.array([dp_res[1] for dp_res in n_runs_res])

# Scatter plot
from plot_scatter import plot_fig5
plot_fig5(nondp_chain, dp_chains[0], dp_mcmc_params, dpvi_params)

# Plot epsilon vs accuracy
from plot_eps_vs_acc_nruns import plot_fig6
plot_fig6(nondp_chain, dp_chains, dp_mcmc_params, dpvi_params)

# Plot proposal variance vs ratio of clipped
from plot_var_vs_clip import plot_fig3
plot_fig3(fname_multi_vars)

# Plot RDP bound
from plot_rdp_bounds import plot_fig1, plot_fig2
# w.r.t qs
qs = np.linspace(0.001, 0.01, 100)
Ns = [100000, 1000000, 10000000]
plot_fig1(qs, Ns, T=5000)
# w.r.t Ts
T_plot = 100000
Ts = np.arange(10, T_plot, 10)
bs = [100, 1000, 10000]
plot_fig2(Ts, bs, q=0.001)	
