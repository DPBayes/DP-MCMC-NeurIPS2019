'''
Script for plotting approximation errors due to X_cor
'''

from collections import OrderedDict as OD
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import stats

from X_corr import *
import X_corr_seita
from plot_path import path

def plot_fig4(filename_res=None):
	"""
	filename_res : filename to precomputed results
	if filename_res == None, calculate approximate X_corr distributions
	using both Seita et al. method and GMM based method proposed in our paper.
	"""
	save_plot_to_disk = True
	use_log = False
	if filename_res!=None:
		plot_only = True # load pre-calculated results, otherwise calculate and save res
	else:
		filename_res = './results/approx_error_res.pickle'
		plot_only = False

	N = int(1e5) # number of samples for estimating ecdfs
	# note: this doesn't update to figure title at the moment
	n_repeats = 20

	x_max = 10 # end points for sampling
	# original Seita et al paper uses n=4000, lam=10.0
	n = 4000 # (actual number of grid points used is 2n+1)
	lam = 10.0

	# should have same number of sigma and C values, so sigma=sqrt(C)
	filename_sigmas = [1.0, 1.225, 1.323, 1.414]
	all_C = [1.0, 1.5, 1.75, 2.0]

	x = np.linspace(-x_max, x_max, 2*n+1)
	if not plot_only:
		# load pre-calculated results
		x_corr_seita_all = OD()
		for sigma in filename_sigmas:
			x_corr_seita_all[sigma] = np.zeros(2*n+1)
			filename = './X_corr/norm2log{}_{}_{}_{}.txt'.format(n,x_max,lam,np.round(sigma,1))
			with open(filename, 'r') as f:
				lines = f.readlines()
			for i,l in enumerate(lines):
				x_corr_seita_all[sigma][i] = float(l.split('\t')[1])
		
		all_params = OD()
		for i,C in enumerate(all_C):
			filename = './X_corr/supplement_test_x_corr_params_C{}.pickle'.format(C)
			with open(filename, 'rb') as f:
				all_params[str(C)] = pickle.load(f)
		
		# calculate max_y |S'(y)-S(y)| for different C values
		sups = np.zeros((2,len(all_C),n_repeats))
		for i, (C,sigma) in enumerate(zip(all_C, filename_sigmas)):
			print('Calculating errors with C={}'.format(C))
			x_corr = x_corr_seita_all[sigma]
			df =  pd.DataFrame({'x':x, 'pdf':x_corr})
			mus, sigmas, pis = all_params[str(C)]
			normal_sigma = np.sqrt(C)
			for ii in range(n_repeats):
				samples = sample_from_mix([mus, sigmas, pis], N)
				samples_seita = X_corr_seita.sample_X_corr(df,N,n)
				noise = np.random.normal(0,normal_sigma,N)
				# calculate ecdfs
				t = np.sort(samples_seita+noise).copy()
				tt = np.sort(samples+noise).copy()
				s = np.arange(1, len(t)+1)/float(len(t))
				ss = np.arange(1, len(tt)+1)/float(len(tt))
				# get max errors
				sups[0,i,ii] = np.amax(np.abs(s-stats.logistic.cdf(t)))
				sups[1,i,ii] = np.amax(np.abs(ss-stats.logistic.cdf(tt)))

		# save results
		with open(filename_res, 'wb') as f:
			pickle.dump(sups,f)
		print('Saved results to {}'.format(filename_res))
	else:
		with open(filename_res, 'rb') as f:
			sups = pickle.load(f)
		print('Loaded results from {}'.format(filename_res))

	# plot results
	x_points = np.linspace(1,len(all_C),len(all_C))
	if use_log:
		plt.errorbar(x_points, np.mean(np.log(sups[0,:,:]),1),\
				yerr=np.std(np.log(sups[0,:,:]),1)/np.sqrt(n_repeats),label='Ridge regression', alpha=.9)
		plt.errorbar(x_points+.03, np.mean(np.log(sups[1,:,:]),1),\
				yerr=np.std(np.log(sups[1,:,:]),1)/np.sqrt(n_repeats),label='GMM', alpha=.9)
		plt.ylabel('log max|S\'(y)-S(y)|')
	else: 
		plt.ylim((1e-3,1e-1))
		ax = plt.gca()
		ax.set_yscale("log", nonposy='clip')#, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
		plt.errorbar(x_points, np.mean(sups[0,:,:],1),\
				yerr=np.std(sups[0,:,:],1)/np.sqrt(n_repeats),label='Ridge regression', alpha=.9)
		plt.errorbar(x_points+.03, np.mean(sups[1,:,:],1),\
				yerr=np.std(sups[1,:,:],1)/np.sqrt(n_repeats),label='GMM', alpha=.9)
		plt.ylabel('max|S\'(y)-S(y)|')
		
	plt.xticks(x_points, all_C)
	plt.xlabel('C')
	plt.suptitle(r'Approximation error due to $\tilde V_{cor}$')
	plt.title(r'$\tilde V_{cor}+N(0,C)$ estimated with $10^5$ samples')
	plt.legend(loc='upper left')

	if save_plot_to_disk:
		plt.savefig(path+'v_cor_approx_errors.pdf',format='pdf')
		plt.close()
	else:
		plt.show()

