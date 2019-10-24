'''
Scipt for plotting approximate V_log cdfs for supplement 
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.stats import logistic
from scipy import stats

import X_corr_seita
from X_corr import *
from plot_path import path

np.random.seed(1606)

save_to_file = True

n_bins = 1000 # number of points used for plotting cdfs
x_max = 10 # end points for sampling
C = 2.0
normal_sigma = np.sqrt(C)
N = 50000 # number of samples to draw

filename = './X_corr/supplement_test_x_corr_params_C{}.pickle'.format(C) # optimised model params for GMM
filename = '../Corr_MoG/X_corr_{}_{}_{}_torch.pickle'.format(n_bins,x_max,C)

# seita params: only used for reading the files
n = 4000 # grid point for seita approximation (actual number is then 2n+1)
lam = 10.0
seita_folder = './X_corr/' # should contain pre-calculated norm2log-file
filename_seita = 'norm2log{}_{}_{}_{}.txt'.format(n,x_max,lam,np.round(normal_sigma,1))

x_corr = np.zeros(2*n+1)
with open(seita_folder + filename_seita, 'r') as f:
    lines = f.readlines()
for i,l in enumerate(lines):
    x_corr[i] = float(l.split('\t')[1])
y = np.linspace(-x_max, x_max, 2*n+1, endpoint=True)
df =  pd.DataFrame({'x':y, 'pdf':x_corr})

with open(filename, 'rb') as f:
	params = pickle.load(f)
mus, sigmas, pis = params

samples = sample_from_mix([mus, sigmas, pis], N)
samples_seita = X_corr_seita.sample_X_corr(df,N,n)
noise = np.random.normal(0,normal_sigma,N)

# plot X_norm+X_corr ecdf and standard logistic cdf:
plt.hist(samples_seita+noise, bins=n_bins ,density=True, cumulative=True, color='b', histtype='step', label='$V_{cor}$ from ridge regression', alpha=.8)
plt.hist(samples+noise, bins=n_bins ,density=True, cumulative=True, color='r', histtype='step', label='$V_{cor}$ from GMM', alpha=.8, linewidth=1.0)
plt.plot(np.linspace(-x_max,x_max,n_bins),stats.logistic.cdf(np.linspace(-x_max,x_max,n_bins),0,1),'--', color='black', alpha=.5 ,label='True logistic cdf')
plt.suptitle('Sample ecdf for $C={}$ and true logistic cdf'.format(C))
plt.title(r'$V_{cor}+\mathcal{N} (0,C)$ estimated with 50000 samples')
plt.xlim(-7.5,7.5)
plt.legend()

if save_to_file:
		plt.savefig(path+'supplement_v_corr_cdfs.pdf', format='pdf')
		plt.close()
else: plt.show()

# plot the differences between V_norm+V_cor ecdfs and standard logistic cdf:
xs1 = np.sort(samples_seita+noise)
ys1 = np.abs(np.arange(1, len(xs1)+1)/float(len(xs1)) - stats.logistic.cdf(xs1))
xs2 = np.sort(samples+noise)
ys2 = np.abs(np.arange(1, len(xs2)+1)/float(len(xs2)) - stats.logistic.cdf(xs2))

plt.plot(xs1,ys1, label='$V_{cor}$ from ridge regression', alpha=.8,color='b')
plt.plot(xs2,ys2, label='$V_{cor}$ from GMM', alpha=.8,color='r')
plt.xlim(-7.5,7.5)
plt.ylabel('|S\'(y)-S(y)|')
plt.suptitle('Abs differences between approximations and true logistic cdf for $C={}$'.format(C))
plt.title(r'$V_{cor}+\mathcal{N} (0,C)$ estimated with 50000 samples')
plt.legend()

if save_to_file:
		plt.savefig(path+'supplement_v_corr_cdfs2.pdf', format='pdf')
		plt.close()
else: plt.show()

done = 1

