import numpy as np
import numpy.random as npr
from scipy.special import expit as logistic
import X_corr
from mog_model import log_likelihood, log_prior
import random

def run_dp_Barker(T, prop_var, theta_0, X, privacy_pars, x_corr_df, n_points, batch_size=100,\
		temp_scale=1, count_clipped=False):
	"""
	T : number of iterations
	theta_0 : the starting value for chain
	"""
	N = len(X)
	d = 2 # Dimensionality of MoG example
	clip_bounds = [0, np.inf]
	theta_chain = np.zeros((T+1,d)) # (alpha, beta)
	theta_chain[0,:] = theta_0 # Initialize chain to given point
	n_accepted = [0,0] # track dp appr. and exact logistic separately
	
	print('Running approximate DP Barker')
	
	sample_vars = []
	if count_clipped: 
		clip_count = [] # We use this list for figure 3
	
	# Run the chain
	for i in range(1,T+1):
		if i % 1000 == 0:
			print('Step {}'.format(i))
		
		#X_batch = np.random.choice(X,batch_size,replace=False)
		#indices = np.random.choice(len(X), batch_size, replace=False)
		indices = random.sample(range(len(X)), batch_size)
		while True: ## If batch_var > 1, draw more samples, else break and continue
			# draw minibatch, use fixed batch size without regard to batch var
			X_batch = X[indices]
			# proposals from standard Gaussians
			proposal = theta_chain[i-1,:] + np.random.normal(0, np.sqrt(prop_var), d)
			# Compute log likelihoods for current and proposed value of the chain
			log_likelihood_theta_prime = log_likelihood(X_batch, proposal)
			log_likelihood_theta = log_likelihood(X_batch, theta_chain[i-1])
			# Compute log prior probabilities for current and proposed value of the chain
			log_prior_theta_prime = log_prior(proposal)
			log_prior_theta = log_prior(theta_chain[i-1])
			# Compute the log likelihood ratios
			ratio = log_likelihood_theta_prime-log_likelihood_theta
			if count_clipped:
				clip_count.append(sum(np.abs(ratio)>clip_bounds[1]))
			sign = np.sign(ratio)
			# Clip the ratios
			R = ratio
			# Compute mean and sample variance of log-likelihoods
			r = R.mean()
			s2 = R.var()
			# Compute \Theta^*(\theta', \theta)
			logp_ratio = N*temp_scale*r+(log_prior_theta_prime-log_prior_theta)
			# Compute scaled batch var
			#batch_var = s2*((N*temp_scale)**2/batch_size)
			batch_var = s2*((N*temp_scale)**2/len(indices))
			sample_vars.append(batch_var)

			## Add normal and correction variables
			if batch_var <= 1:
				normal_noise = npr.randn(1)*np.sqrt(1.-batch_var)
				x_corr = X_corr.sample_from_mix(x_corr_df, 1)[0]
				break
			else:
				print('Batch var > 1, drawing more samples')
				#new_indices = np.random.choice(N, 100, replace=False) ## Draw 100 more samples to lower batch var
				new_indices = random.sample(range(N), 100) ## Draw 100 more samples to lower batch var
				indices = np.unique(np.concatenate([indices, new_indices]))

		acc =  logp_ratio + normal_noise + x_corr

		if acc > 0:
			# accept
			theta_chain[i,:] = proposal
			n_accepted[0] += 1
		else:
			# reject
			theta_chain[i,:] = theta_chain[i-1,:]
	if count_clipped: return theta_chain, n_accepted, sample_vars, clip_count
	else : return theta_chain, n_accepted, sample_vars

