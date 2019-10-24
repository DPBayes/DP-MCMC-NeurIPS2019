import pandas as pd
import numpy as np
import numpy.random as npr
from scipy.special import logsumexp
from scipy.stats import norm as scipy_norm
import sys, pickle

seed = int(sys.argv[1])
npr.seed(seed)

def main():
	print('Running DP-SGLD for seed {}'.format(seed))
	# Read data
	data = pd.read_csv('../data/MoG_toy_data.csv').values.squeeze()
	N = len(data)
	# Init params
	params0 = npr.randn(2)
	params = params0.copy()
	batch_size = 100
	L = .01
	#T = 1000000
	T = 6000000
	q = batch_size/N
	thetas = np.zeros([T,2])
	## set learning rate
	## noise sigma = 2.0 and T=2e5 would produce eps=1.25
	#eta = (batch_size/(np.sqrt(N)*1.0*L))**2 # This is same as having sigma = 4.0
	eta = 70.0
	delta = 2e-6
	noise_sigma = batch_size/np.sqrt(eta*N)/L
	## Compute privacy budget
	from privacy.analysis.compute_dp_sgd_privacy import compute_rdp, get_privacy_spent
	rdp_orders = range(2, 500)
	rdp_eps = compute_rdp(q, noise_sigma, T, rdp_orders)
	epsilon = get_privacy_spent(rdp_orders, rdp_eps, target_delta=delta)[0]
	print("Epsilon : {}".format(epsilon))

	temp_scale=(100./N)
	# run dp-sgld
	for t in range(T):
		theta = params
		thetas[t] = theta
		X = npr.choice(data, batch_size)
		# likelihood grads
		logp_1 = scipy_norm.logpdf(X, loc=theta[0], scale=np.sqrt(2))
		logp_2 = scipy_norm.logpdf(X, loc=np.sum(theta), scale=np.sqrt(2))

		loss = logsumexp(np.stack([logp_1, logp_2]), axis=0) 

		T1 = (X-theta[0])/2
		T2 = (X-theta.sum())/2

		g1 = (T1*np.exp(logp_1)+T2*np.exp(logp_2))/np.exp(loss)
		g2 = (T2*np.exp(logp_2))/np.exp(loss)

		grads = np.vstack([g1, g2]).T*temp_scale
		# clip
		grads_norm = np.linalg.norm(grads, axis=1)
		grads = (grads.T*(np.minimum(1, L/grads_norm))).T
		assert np.linalg.norm(grads, axis=1).max()<(L+1e-12)
		ll_grad = grads.sum(0)
		## prior grad
		prior_g1 = -(theta[0])/10.
		prior_g2 = -(theta[1])/1.

		logprior_grad = np.array([prior_g1, prior_g2])

		z = npr.randn(2)*np.sqrt(eta/N)
		grad = eta*(ll_grad*(1./batch_size)+logprior_grad/N)+z
		params += grad
		
	## save results
	from utils import fnamer
	fname = fnamer('./results/dp_sgld_res_seed_{}_'.format(seed))
	dp_sgld_params = {'L':L, 'eta':eta, 'T':T, 'tau':batch_size, 'noise_sigma':noise_sigma,\
			'temp':temp_scale, 'seed':seed}
	to_pickle = [thetas, dp_sgld_params]
	pickle.dump(to_pickle, open(fname, 'wb'))

if __name__=='__main__':
	main()
