import numpy as np
import numpy.random as npr
import torch, math, pickle
from torch.distributions import Normal, Uniform
from torch.distributions import SigmoidTransform, TransformedDistribution, AffineTransform
import sys

#def mix_logp(params, x, normal_sigma, y_log):
def loss_func(params, x, normal_sigma, y_log, symmetric=True):
	mus, sigmas_, pi_ = params
	K = len(mus)
	sigmas = torch.sqrt(torch.exp(2*sigmas_)+normal_sigma**2)
	log_pi = torch.nn.LogSoftmax(dim=-1).forward(pi_)
	if not symmetric:
		logps = torch.stack([Normal(mus[k], sigmas[k]).log_prob(x)+log_pi[k] for k in range(K)])
	else: # make pdf symmetric by taking mean from model and mirrored model
	# note: this is not exactly the best way to code this, fix if too slow/symmetric enforced in other way
		logps = torch.stack([torch.log(.5*torch.exp(Normal(mus[k], sigmas[k]).log_prob(x))*torch.exp(log_pi[k]) + .5*torch.exp(Normal(-mus[k], sigmas[k]).log_prob(x))*torch.exp(log_pi[k])) for k in range(K)] )
	
	return torch.norm(torch.logsumexp(logps, dim=0)-y_log)

# assume params are numpy arrays as returned by get_x_corr_params
def mix_logpdf(x, mus, sigmas, pi):
	x = torch.Tensor(x)
	mus, sigmas, pi = torch.Tensor(mus), torch.Tensor(sigmas), torch.Tensor(pi)
	K = len(mus)
	# model as is
	log_pi = torch.log(pi)
	logps = torch.logsumexp(torch.stack([Normal(mus[k], sigmas[k]).log_prob(x)+log_pi[k] for k in range(K)]), dim=0)
	return logps.detach().numpy()


## Define and train
def get_x_corr_params(x_max, n_points, C, K=50, lr=1e-2, T=10000, path_to_file=None, symmetric=True, early_stop=-1):
	"""
	C : the variance on X_normal
	symmetric: enforce symmetric model; in practice use mean of model and mirrored model; returned parameters then include the mirrored copies (so have 2K components)
	"""
	torch.set_default_tensor_type('torch.DoubleTensor')
	base_distribution = Uniform(0, 1)
	transforms = [SigmoidTransform().inv, AffineTransform(loc=0, scale=1)]
	logistic = TransformedDistribution(base_distribution, transforms)
	mus0 = 0.1*torch.randn(K)
	#mus0[K//2:] = -mus0[:K//2]
	mus = mus0.detach().requires_grad_(True)
	sigmas0 = 0.1*torch.randn(K)
	#sigmas0[K//2:] = sigmas0[:K//2]
	sigmas = sigmas0.detach().requires_grad_(True)
	pis0 = torch.rand(K) #0.2*
	pis = pis0.detach().requires_grad_(True)
	normal_sigma = torch.sqrt(torch.ones(1)*C)

	x_log = torch.linspace(-x_max, x_max, n_points)
	y_log = logistic.log_prob(x_log)
	params = [mus, sigmas, pis]
	optimizer = torch.optim.Adam(params, lr=lr)
	
	min_loss = 10**5
	counter = 0
	for i in range(T):
		optimizer.zero_grad()
		loss = loss_func(params, x_log, normal_sigma, y_log)
		if loss < min_loss:
		    min_loss = loss
		    counter = 0
		else:
		    counter += 1
		    if early_stop == counter:
		        print('Stopping early..')
		        break
		if i % 1000 == 0: print('loss: {}, iter: {}/{}'.format(loss.detach().numpy(),i,T))
		loss.backward(retain_graph=True)
		optimizer.step()
	
	mus, sigmas, pis = params
	mus = mus.data.numpy()
	sigmas = np.exp(sigmas.data.numpy())
	pis = torch.softmax(pis, dim=-1).data.numpy()
	if symmetric:
		mus = np.concatenate((mus, -mus))
		sigmas = np.concatenate((sigmas, sigmas))
		pis = np.concatenate((.5*pis, .5*pis))
		
	if path_to_file==None:
		#fname = '../Corr_MoG/X_corr_{}_{}_{}_torch.pickle'.format(n_points,x_max,C)
		fname = './X_corr/X_corr_{}_{}_{}_torch.pickle'.format(n_points,x_max,C)
	else:
		fname = path_to_file
	if path_to_file != 'no_write':
		pickle.dump([mus, sigmas, pis], open(fname, 'wb'))	
		print('Wrote params to {}'.format(fname))
	return [mus, sigmas, pis]


def sample_from_mix(params, n_samples):
	mus, sigmas, pis = params
	K = len(mus)
	samples = np.zeros(n_samples)
	for i in range(n_samples):
		k = npr.choice(K, p=pis)
		samples[i] = mus[k]+npr.randn(1)*sigmas[k]
	return samples

