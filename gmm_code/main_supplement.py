"""
This file contains all experiments of the supplement.
"""
import numpy as np
from X_corr import get_x_corr_params
from plot_approx_errors import plot_fig4
import torch

learn_X_corr = True

if learn_X_corr:
	np.random.seed(123)
	torch.manual_seed(123)
	T = 20000
	x_max = 10
	n_points = 1000
	all_C = [1.0, 1.5, 1.75, 2.0]
	lr = 1e-2
	for C in all_C:
		path_to_file = './X_corr/supplement_test_x_corr_params_C{}.pickle'.format(np.round(C,2))
		get_x_corr_params(x_max=x_max, n_points=n_points,C=C, lr=lr, T=T, path_to_file=path_to_file)
	# Note, it takes a while to run above
else:
	try : plot_fig4('./results/approx_error_res.pickle')
	except : plot_fig4()


from plot_supplement_GMM_appr import done
from plot_supplement_cdfs import done
from plot_supplement_seita_pdf import done
