'''
Script for plotting Seita et al. pdf plots for Supplement
'''

from collections import OrderedDict as OD
from matplotlib import pyplot as plt
import numpy as np
from plot_path import path

save_to_disk = True
use_log = True
seita_folder = './X_corr/' # the folder should contain pre-calculated norm2log-files 
x_max = 10 # end points for sampling
# original Seita et al paper uses n=4000, lam=10.0
n = 4000 # (actual number of grid points used is 2n+1)
lam = 10.0
sigmas = [1.0, 1.225, 1.323, 1.414]


x_corr_all = OD()
for sigma in sigmas:
    x_corr_all[sigma] = np.zeros(2*n+1)
    filename = 'norm2log{}_{}_{}_{}.txt'.format(n,x_max,lam,np.round(sigma,1))
    with open(seita_folder + filename, 'r') as f:
        lines = f.readlines()
    for i,l in enumerate(lines):
        x_corr_all[sigma][i] = float(l.split('\t')[1])

fig, axes = plt.subplots(4,1, sharex=True)
fig.subplots_adjust(hspace=0.1)
if use_log: fig.suptitle('Seita et al. approximate correction distribution log-pdfs with varying C\nn={}, $\lambda$={}'.format(n, lam))
else: fig.suptitle('Seita et al. approximate correction distribution pdfs with varying C\nn={}, $\lambda$={}'.format(n, lam))
for i,sigma in enumerate(sigmas):
    if use_log: axes[i].plot(np.linspace(-x_max,x_max,2*n+1),np.log(x_corr_all[sigma]), label='C={}'.format(np.round(sigma**2, 3)))
    else: axes[i].plot(np.linspace(-x_max,x_max,2*n+1),x_corr_all[sigma], label='C={}'.format(np.round(sigma**2, 3)))
    axes[i].legend(loc='upper right')
ticks = axes[0].get_yticks()
for i in range(len(sigmas)):
    axes[i].set_yticks(([]))
#plt.tight_layout()

if save_to_disk:
	if use_log: plt.savefig(path+'seita_log_x_corrs.pdf',format='pdf')
	else: plt.savefig(path+'seita_x_corrs.pdf',format='pdf')
else:
	plt.show()

plt.close()
done = 1
