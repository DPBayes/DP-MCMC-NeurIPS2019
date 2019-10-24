'''
Script for plotting supplement pdf figures for GMM
'''

from collections import OrderedDict as OD
import matplotlib.pyplot as plt
import pickle

from X_corr import *
from plot_path import path

save_to_disk = True
use_log = True

all_C = [1.0, 1.5, 1.75, 2.0] # C values, assume pre-calculated models
n=1000 # number of grid points for plotting


all_params = OD()
x = np.linspace(-10, 10, n)
fig, axes = plt.subplots(len(all_C),1, sharex=True)
if use_log: fig.suptitle('GMM approximate correction distribution log-pdfs with varying C\nn={}'.format(n))
else: fig.suptitle('GMM approximate correction distribution pdfs with varying C\nn={}'.format(n))
for i,C in enumerate(all_C):
	filename = './X_corr/supplement_test_x_corr_params_C{}.pickle'.format(C)
	with open(filename, 'rb') as f:
		all_params[str(C)] = pickle.load(f)
	normal_sigma = np.sqrt(C)
	mus, sigmas, pis = all_params[str(C)]
	if use_log: axes[i].plot(x, mix_logpdf(x, mus, sigmas, pis),label='C={}'.format(C))
	else: axes[i].plot(x, np.exp(mix_logpdf(x, mus, sigmas, pis)),label='C={}'.format(C))
	axes[i].legend()

fig.subplots_adjust(hspace=0.1)
ticks = axes[0].get_yticks()
for i in range(len(all_C)):
	axes[i].set_yticks(([]))
#plt.tight_layout()

if save_to_disk:
	if use_log: plt.savefig(path+'GMM_log_x_corrs.pdf',format='pdf')
	else: plt.savefig(path+'GMM_x_corrs.pdf',format='pdf')
else:
	plt.show()

plt.close()
done = 1
