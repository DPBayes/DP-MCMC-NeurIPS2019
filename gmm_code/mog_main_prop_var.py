'''
Script for creating data for Figure 3.
Runs the DP subsampled MCMC method for multiple proposal variances
'''

from collections import OrderedDict as od
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
from scipy import stats
import sys, pickle, datetime
from barker_mog import run_dp_Barker

import X_corr


def run_chain_multiple_prop_var(data, T, theta_from_dpvi):
	'''
	Model:
		theta ~ N(0,diag(sigma_1^2, sigma_2^2))
		x_i ~ .5*N(theta_1, sigma_x^2) + .5*N(theta_1+theta_2, sigma_x^2)
		use fixed values
		sigma_1^2 = 10, sigma_2^2 = 1, sigma_x^2 = 2
		theta_1 = 0, theta_2 = 1
	'''
	
	####################################
	N = len(data)
	# MCMC for posterior estimation
	## Set path to save results
	fname = './results/dp_mcmc_results_temped_multiple_prop_vars_'

	batch_size = 1000
	burn_in = 0
	prop_vars = np.linspace(0.005, 0.02, 10) # Gaussian proposal variances 
	temp_scale = 100/N

	# exact DP Barker: exact if correction to logistic is exact, and no clipping
	privacy_pars = {}
	privacy_pars['noise_scale'] = np.sqrt(2.0)
	privacy_pars['clip'] = [0, 0.99*np.sqrt(batch_size)/temp_scale/N]
	# Parameters for X_corr
	x_max = 10 # Sets the bound to grid [-x_max, x_max]
	n_points = 1000 # Number of grid points used
	normal_variance = np.round(privacy_pars['noise_scale']**2) # C in the paper
	# Set X_corr filename and try to read parameters from file
	x_corr_filename =  './X_corr/X_corr_{}_{}_{}_torch.pickle'.format(n_points,x_max,normal_variance)
	try:
		# Try to read X_corr-MoG parameters from file
		print('reading X_corr params from file')
		xcorr_params = pickle.load(open(x_corr_filename, 'rb'))
	except:
		# Learn X_corr-MoG parameters for given normal variance 
		print('no existing file found, creating new x_corr parameter & saving to file')
		xcorr_params = X_corr.get_x_corr_params(x_max=x_max, n_points=n_points,\
				C=normal_variance, path_to_file=x_corr_filename)
	# Run the DP-MCMC for multiple proposal variances
	theta_chains = []
	n_accepteds = []
	sample_varss = []
	clip_counts = []
	for i, prop_var in enumerate(prop_vars):
		theta_chain, n_accepted, sample_vars, clip_count = run_dp_Barker(T, prop_var,\
						theta_from_dpvi, data, privacy_pars, xcorr_params, n_points,\
						batch_size=batch_size, temp_scale=temp_scale, count_clipped=True)
		theta_chains.append(theta_chain)
		n_accepteds.append(n_accepted)
		sample_varss.append(sample_vars)
		clip_counts.append(clip_count)
	
	theta_chain = np.array(theta_chains)
	n_accepted = np.array(n_accepteds)
	sample_vars = np.array(sample_varss)
	clip_counts = np.array(clip_counts)
	# Save results to a pickle file
	date = datetime.date.today()
	fname += str(date.day)+'_'+str(date.month)+'.p'
	fname_extension = 0
	while True:
		try: 
			f = open(fname, 'rb')
			f.close()
			if fname_extension==0:
				fname = fname[:-2]+'({}).p'.format(fname_extension)
			else:
				fname = fname[:-len('({}).p'.format(fname_extension))]+'({}).p'.format(fname_extension)
			fname_extension += 1
		except:
			f = open(fname, 'wb')
			print('Wrote results to {}'.format(fname))
			break

	dp_mcmc_params = {'N':N, 'B': batch_size, 'T': T, 'temp': temp_scale, \
						'prop_vars':prop_vars, 'clip_counts':clip_counts}
	to_pickle = [dp_mcmc_params, theta_chain, privacy_pars]
	pickle.dump(to_pickle, f)
	f.close()
	return fname
