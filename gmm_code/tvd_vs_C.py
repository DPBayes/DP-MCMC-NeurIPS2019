import numpy as np
import pickle

from X_corr import get_x_corr_params, mix_logpdf
from scipy.special import logit
from scipy.stats import logistic
from plot_path import path

def TVD(q):
	"""
	Computes Total Variation Distance between exact logistic and approximate logistic distributions
	q : pdf of the approximate distribution
	Omega : interval on which to evaluate TVD (defaults to interval in which the P(Omega)>1-machine_eps)
	"""
	mach_eps = np.finfo(float).eps
	lower = logit(mach_eps/2)
	Omega, delta = np.linspace(lower, -lower, 10000, retstep=1)
	## Approximate integral in Omega
	p = logistic.pdf(Omega)
	q = q(Omega)
	tvd = 0.5*np.linalg.norm(p-q, ord=1)*delta
	return tvd 

from scipy.stats import norm
def mix_cdf(x, mus, sigmas, pis):
	return np.sum([pi*norm.cdf(x, mu, sigma) for mu,sigma,pi in zip(mus, sigmas, pis)], axis=0)

def main():
	Cs = [0.0, 0.1, 0.5, 1.0, 2.0]
	n_points = 1000
	x_max = 15
	early_stop = 200 # use -1 for no early stop
	
	TVDs = {}
	## Load / Train GMMs
	for C in Cs:
		fname = './X_corr/X_corr_{}_{}_{}_torch.pickle'.format(n_points,x_max,C)
		try : 
			handle = open(fname, 'r')
			handle.close()
			[mus, sigmas, pis] = pickle.load( open(fname, 'rb'))	
		except:
			mus, sigmas, pis = get_x_corr_params(x_max, n_points, C, K=50, lr=1e-2, T=20000, early_stop=early_stop, path_to_file='no_write')#10000) 
		# should add early stop? C:s seem to require quite different iterations before starting to overfit
		
		q = lambda x : np.exp(mix_logpdf(x, mus, np.sqrt(C+sigmas**2), pis))

		TVDs[C] = TVD(q)
	pickle.dump(TVDs, open('tvd_{}.p'.format(Cs), 'wb')) # Save TVDs

	import matplotlib.pyplot as plt
	plt.cla()
	plt.plot(TVDs.keys(), TVDs.values())
	plt.ylabel(r'TVD$(f_{log} || \tilde{f}_{log})$', fontsize=16)
	plt.xlabel('C', fontsize=16)
	plt.savefig(path+'tvd_vs_C.pdf', format='pdf', bbox_inches='tight')
	plt.close()

if __name__=="__main__":
	main()
