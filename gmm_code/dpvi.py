
"""
This file contains functions for private initialization of the chain using DPVI
"""

import torch, math
from torch.distributions import Normal as norm
torch.set_default_tensor_type('torch.DoubleTensor')


def log_prior(params, draw):
	theta = reparametrize(params, draw)
	return norm(0, math.sqrt(10)).log_prob(theta[0])+norm(0, math.sqrt(1)).log_prob(theta[1]) 

def log_likelihood(params, X, draw):
	theta = reparametrize(params, draw)
	## Log-likelihood for GMM with 0.5 weight for each comp and variances 1
	logp_1 = math.log(0.5)+norm(theta[0], math.sqrt(2)).log_prob(X)
	logp_2 = math.log(0.5)+norm(torch.sum(theta), math.sqrt(2)).log_prob(X)
	return torch.logsumexp(torch.stack([logp_1, logp_2]), dim=0) 

def reparametrize(params, z):
	mu, log_sigma = torch.split(params, 2)
	return mu+torch.exp(log_sigma)*z

def mvn_entropy(params):
	mu, log_sigma = torch.split(params, 2)
	return 0.5*torch.sum(2*log_sigma+math.log(2*math.pi*math.e))

def clip(x, C):
	x_norm = x.norm()
	return x*min(1, C/x_norm) ## FIXED CLIPPING

import numpy.random as npr
def dpvi_mix_gaus(data, k, params_0, T, batch_size, C, noise_sigma, learning_rate):
	"""
	data : np.array
	k : number of mixture components
	params_0 : initial parameters (contains both mus and unconstrained sigmas)
	T : number of iterations
	batch_size : size of the minibatch
	C : clipping threshold
	noise_sigma : Std. for DP noise
	learning_rate : step size for optimizer
	"""
	N = len(data)
	data = torch.tensor(data)
	params = torch.tensor(params_0, requires_grad=True)
	optimizer = torch.optim.Adam([params], lr=learning_rate)
	for t in range(T):
		optimizer.zero_grad()
		draw = torch.randn(k) # Only one mc integration
		indices = npr.choice(len(data), batch_size, replace=False)
		minibatch = data[indices]
		# Likelihoods
		ll_loss = 0
		for i, sample in enumerate(minibatch):
			loss_i = -1.*log_likelihood(params, sample, draw)
			loss_i.backward(retain_graph=True)
			ll_loss += loss_i.item()
			if i == 0:
				g = clip(params.grad.data, C).clone() ## FIXED CLIPPING
				optimizer.zero_grad()
			else:
				g += clip(params.grad.data, C).clone()
				optimizer.zero_grad()
		params.grad.data += (g+C*noise_sigma*torch.randn(len(params)))*(N/batch_size) ## Added scaling

		# Prior
		prior_loss = -1*log_prior(params, draw)
		prior_loss.backward(retain_graph=True)
		# Entropy
		entropy_loss = -1*mvn_entropy(params)
		entropy_loss.backward(retain_graph=True)
		# Take step
		optimizer.step()
		if t % 100 == 0 :
			loss = ll_loss+prior_loss.item()+entropy_loss.item()
			print(loss)
	params.detach_()
	return params.data.numpy(), params_0
