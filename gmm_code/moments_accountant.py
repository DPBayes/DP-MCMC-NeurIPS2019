import numpy as np
from gaussian_moments import compute_log_moment, get_privacy_spent

def ma(sigma, q, num_steps, delta, max_lamb=64):
	max_moment_order = np.floor(-sigma**2*np.log(q*sigma)).astype('int')
	max_moment_order = min(max_moment_order, max_lamb)
	assert q < 1/(16*sigma)
	epsilon = np.inf
	for lam in range(1, max_moment_order+1):
		try : 
			log_moment = compute_log_moment(q, sigma, num_steps, lam, verify=False)
			eps = get_privacy_spent([(lam, log_moment)], target_delta=delta)[0]
			epsilon = min(epsilon, eps)
		except : break
		if eps == np.inf: break
	return epsilon
