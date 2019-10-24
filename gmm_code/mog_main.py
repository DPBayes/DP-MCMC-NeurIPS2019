'''
Main function to run the MCMC
'''

import numpy as np
import pickle, datetime

import X_corr


def run_chain(data, T, theta_from_dpvi, privacy=True):
	'''
	Model:
		theta ~ N(0,diag(sigma_1^2, sigma_2^2))
		x_i ~ .5*N(theta_1, sigma_x^2) + .5*N(theta_1+theta_2, sigma_x^2)
		use fixed values
		sigma_1^2 = 10, sigma_2^2 = 1, sigma_x^2 = 2
		theta_1 = 0, theta_2 = 1
	'''
	
	if privacy : 
		from barker_mog import run_dp_Barker
	else:
		from barker_mog_nondp import run_dp_Barker

	####################################
	N = len(data)
	# MCMC for posterior estimation
	## Set path to save results
	fname = 'dp_mcmc_results_temped_'
	if privacy==False:
		fname = 'non_'+fname
	fname = './results/'+fname
	from utils import fnamer
	fname = fnamer(fname)

	batch_size = 1000
	burn_in = 0
	prop_var = .01 # Gaussian proposal variance 
	temp_scale = 100/N

	# exact DP Barker: exact if correction to logistic is exact, and no clipping
	privacy_pars = {}
	if privacy : privacy_pars['noise_scale'] = np.sqrt(2.0)
	else : privacy_pars['noise_scale'] = 0
	privacy_pars['clip'] = [0, 0.99*np.sqrt(batch_size)/temp_scale/N]
	# Parameters for X_corr
	x_max = 10 # Sets the bound to grid [-x_max, x_max]
	n_points = 1000 # Number of grid points used
	normal_variance = np.round(privacy_pars['noise_scale']**2) # C in the paper
	if not privacy:
		normal_variance = 1.0
	# Set X_corr filename and try to read parameters from file
	if privacy : x_corr_filename =  './X_corr/X_corr_{}_{}_{}_torch.pickle'\
					.format(n_points,x_max,normal_variance)
	else : x_corr_filename =  './X_corr/X_corr_{}_{}_{}_torch.pickle'\
					.format(n_points, x_max, normal_variance)

	try:
		# Try to read X_corr-MoG parameters from file
		print('reading X_corr params from file')
		xcorr_params = pickle.load(open(x_corr_filename, 'rb'))
	except:
		# Learn X_corr-MoG parameters for given normal variance 
		print('no existing file found, creating new x_corr parameter & saving to file')
		xcorr_params = X_corr.get_x_corr_params(x_max=x_max, n_points=n_points,\
				C=normal_variance, path_to_file=x_corr_filename)
	# Run the DP-MCMC
	theta_chain, n_accepted, sample_vars = run_dp_Barker(T, prop_var, theta_from_dpvi, data, privacy_pars,\
					xcorr_params, n_points, batch_size=batch_size, temp_scale=temp_scale)
	

	dp_mcmc_params = {'N':N, 'B': batch_size, 'T': T, 'temp': temp_scale,\
					  'prop_var':prop_var}
	to_pickle = [dp_mcmc_params, privacy_pars]
	return (theta_chain, n_accepted, sample_vars), to_pickle, fname
