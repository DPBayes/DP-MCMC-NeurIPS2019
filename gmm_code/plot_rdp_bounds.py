import numpy as np
import matplotlib.pyplot as plt

from exact_rdp import *
from plot_path import path

def plot_fig1(qs, Ns, T):
	# Plot epsilon vs. q
	fontsize = 25
	figsize = (8.5,6)
	res = []
	for N in Ns:
		max_alpha = 200
		min_eps_q = np.array([get_privacy_spent(int(q*N), N, T, max_alpha=200) for q in qs])[:,0]
		res.append(min_eps_q)
	plt.figure(figsize=figsize)
	for N ,min_eps_q in zip(Ns, res):
		plt.plot(qs, min_eps_q, label='N=1e{}'.format(int(np.log10(N))))
	plt.yscale('log')
	plt.yticks([0.5, 1.0, 6.0], labels=[0.5,1.0,6.0])
	plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize-1)
	plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize-1)
	plt.xlabel('q', fontsize=fontsize)
	plt.ylabel(r'$\epsilon$', fontsize=fontsize)
	plt.title('Privacy budget of the subsampled DP MCMC \n'+r'$T={0}, \delta=1/N$'.format(T), fontsize=fontsize)
	plt.legend(loc=4, fontsize=fontsize-1)
	plt.tight_layout()
	plt.savefig(path+'eps_vs_q.pdf',format='pdf')
	plt.close()

def plot_fig2(Ts, bs, q):
	# Plot epsilon vs. T
	fontsize = 25
	figsize = (8.5,6)
	res = []	
	for b in bs:
		max_alpha = 200
		max_alpha = min(max_alpha,b//5-1)
		N = int(b/q)
		delta = 1/N
		min_eps_T = np.inf*np.ones(len(Ts))
		for max_alpha_ in range(3, max_alpha):
			eps_alpha_list = [rd_approx(alpha, b) for alpha in range(2, max_alpha_+1)]
			amplified_eps = amplified_RDP(eps_alpha_list, max_alpha_, q)
			if np.isfinite(amplified_eps):
				total_eps = [from_RDP_to_DP(T*amplified_eps, max_alpha_, delta) for T in Ts]
				min_eps_T = np.minimum(min_eps_T, total_eps)
		res.append(min_eps_T)
	plt.figure(figsize=figsize)
	for b, min_eps_T in zip(bs, res):
		N = int(b/q)
		plt.plot(Ts, min_eps_T, label='N=1e{}'.format(int(np.log10(N))))
	plt.yscale('log')
	plt.yticks([0.06, 1.0, 3.0], labels=[0.06,1.0,3.0])
	plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize-1)
	plt.gca().set_xticklabels(['', '0','2e4','4e4','6e4','8e4', '1e5'])
	plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize-1)
	plt.xlabel('Number of iterations', fontsize=fontsize)
	plt.ylabel(r'$\epsilon$', fontsize=fontsize)
	plt.title('Privacy budget of the subsampled DP MCMC \n'+r'$q={0}, \delta=1/N$'.format(q), fontsize=fontsize)
	plt.legend(loc=4, fontsize=fontsize-1)
	plt.tight_layout()
	plt.savefig(path+'eps_vs_T.pdf',format='pdf')
	plt.close()

