'''
Mixture of Gaussians model for exact MCMC
'''

import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp


def log_prior(theta):
	return norm.logpdf(theta[0], 0, np.sqrt(10))+norm.logpdf(theta[1], 0, np.sqrt(1))


def log_likelihood(X, theta):
	## Log-likelihood for GMM with 0.5 weight for each comp and variances 1
	logp_1 = np.log(0.5)+norm.logpdf(X, theta[0], np.sqrt(2))
	logp_2 = np.log(0.5)+norm.logpdf(X, sum(theta), np.sqrt(2))	
	return logsumexp(np.stack([logp_1, logp_2]), axis=0) 

