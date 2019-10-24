import pandas as pd
import numpy as np
from scipy.stats import norm, logistic

def sample_X_corr(table, n, n_points=2000):
	if type(table)==str:
		table = pd.read_csv(table, sep=';', header=None, names=['x', 'pdf'])
	t = table['x']
	table['cdf'] = table['pdf'].cumsum()/table.pdf.sum()
	samples = np.zeros(n)
	for i in range(n):
		u = np.random.rand()
		#x = np.argmin(np.abs(table['cdf']-u))
		x = (np.abs(table['cdf']-u)).idxmin()
		if u>table['cdf'][x]:
			x0 = x
			#x1 = min(x+1, 4000)
			x1 = min(x+1, 2*n_points+1)
		else:
			x0 = max(x-1,0)
			x1 = x
		
		delta = table['cdf'][x1]-table['cdf'][x0]
		assert not delta < 0
		if delta == 0:
		    print('delta = 0')
		    delta = 1
		alpha = (u-table['cdf'][x0])/delta
		samples[i] = alpha*table['x'][x1]+(1-alpha)*table['x'][x0]
	return  samples

#def create_table(x_max, n_points, sigma = 1.0, print_to_file=False, lam=10):
#	# x_max = V in the paper
#	# n_points = N in the paper
#	# print_to_file = filename of false
#	x = np.linspace(-2*x_max, 2*x_max, 4*n_points+1, endpoint=True)
#	y = np.linspace(-x_max, x_max, 2*n_points+1, endpoint=True)
#	M = norm.cdf(x.reshape([4*n_points+1,1])-y, scale=sigma)
#	v = logistic.cdf(x)
#	# u* = (M^TM+\lambda*I)^-1M^Tv
#	A = M.T.dot(M)+lam*np.eye(2*n_points+1)
#	u = np.maximum(np.linalg.solve(A, M.T.dot(v)), 0.0)
#	df =  pd.DataFrame({'x':y, 'pdf':u})
#	if print_to_file!=False:
#		df.to_pickle(print_to_file)
#		return df
#	return df
