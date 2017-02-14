## 2016/07/19 Lingxue Zhu
## A class to perform M-step in LogitNormalGEM
import logging
import numpy as np
import scipy as scp
from scipy.special import psi, gammaln
from utils import * ## helper functions


class LogitNormalMLE(object):

	def __init__(self, 
				## starting values for gradient descent
				start_A, start_alpha=None, 
				BKexpr=None, ## M-by-N, bulk expression
				SCexpr=None, ## L-by-N, single cell expression
				G=None, ## L-by-1, single cell types
				K=3, ## number of cell types
				hasBK=True, hasSC=True, ## model specification
				init_A = None, ## N-by-K, initial value for A
				init_alpha = None, ## K-by-1, initial value for alpha
				init_pkappa = None, init_ptau = None, ## 2-by-1, mean and 1/var
				min_A=1e-6, ## minimal value of A; must be positive
				min_alpha=1, ## minimum value of alpha
				MLE_CONV=1e-4, MLE_maxiter=100
				):
		(self.K, self.min_A, self.min_alpha) = (K, min_A, min_alpha)
		(self.MLE_CONV, self.MLE_maxiter) = (MLE_CONV, MLE_maxiter)
		(self.SCexpr, self.G, self.BKexpr) = (SCexpr, G, BKexpr)
		(self.hasBK, self.hasSC) = (hasBK, hasSC)
		## starting values
		self.A = np.copy(start_A)
		self.alpha = np.copy(start_alpha)
		## dimensions and other parameters that are iteratively used in MLE
		if self.hasSC: 
			## SC parameters
			(self.L, self.N) = self.SCexpr.shape
			## read depth: (L, )
			self.SCrd = self.SCexpr.sum(axis=1) 
			## cell ids in each type
			self.itype = [] 
			for k in xrange(self.K):
				self.itype += [np.where(self.G == k)[0]]
		if self.hasBK: ## bulk parameters
			(self.M, self.N) = self.BKexpr.shape
		## initialize parameters
		self.A=init_A
		self.alpha=init_alpha
		self.pkappa=init_pkappa
		self.ptau=init_ptau


	def update_suff_stats(self, suff_stats): 
		""" 
		Update the new sufficient statistics obtained from E-step
		these suff.stats are used to compute MLEs.
		"""
		self.suff_stats = suff_stats

		## compute the coefficient for 1/A, A, constant in gradient
		self.coeffAinv = np.zeros(self.A.shape)
		self.coeffA = np.zeros(self.A.shape)

		if self.hasBK:
			## E[mean_j Z_ijk], N x K
			self.coeffAinv += suff_stats["exp_Zik"]
			self.suff_stats["exp_mean_logW"] = suff_stats["exp_logW"].mean(axis=1)

		if self.hasSC:
			## Pre-calculate: unchanged across iterations
			## E[- tau_l^2 * w_il]
			self.coeffA = suff_stats["coeffAsq"] * self.L
			## sum_l S_il * R_l / u_l, where sum is within cell type
			self.type_expr = np.zeros(self.A.shape, dtype=float)
			for k in xrange(self.K):
				itype = self.itype[k]
				self.type_expr[:, k] = (self.SCexpr[itype, :] * \
				 			suff_stats["exp_S"][itype, :]).sum(axis=0)
			## SCexpr .* E(S)
			self.coeffAinv += self.type_expr
			## auxilliary
			self.u = (np.transpose(self.A[:, self.G]) *\
							self.suff_stats["exp_S"]).sum(axis=1)
				
		# if self.hasSC: 
			## auxiliary u: (L, )
			## t(A)*S: L x N
			# self.u = (np.transpose(self.A[:, self.G]) *\
			# 						 suff_stats["exp_S"]).sum(axis=1)
			# ## Pre-calculate: unchanged across MLE iterations
			# ## (sum_l Y_li S_li) / L, sum within cell type
			# self.type_expr = np.zeros(self.A.shape, dtype=float)
			# for k in xrange(self.K):
			# 	itype = self.itype[k]
			# 	self.type_expr[:, k] = (self.SCexpr[itype, :] * \
			# 	 			suff_stats["exp_S"][itype, :]).sum(axis=0)
			# 	self.type_expr[:, k] /= self.L	


	def opt_kappa_tau(self):
		"""Optimize mean and 1/variance of (kappa, tau).
		 This has closed form solution."""
		kappa_mean = np.mean(self.suff_stats["exp_kappa"])
		tau_mean = np.mean(self.suff_stats["exp_tau"])
		kappa_var = np.mean(self.suff_stats["exp_kappasq"]) - kappa_mean**2
		tau_var = np.mean(self.suff_stats["exp_tausq"]) - tau_mean**2

		self.pkappa = np.array([kappa_mean, 1.0/kappa_var])
		self.ptau = np.array([tau_mean, 1.0/tau_var])


	def opt_A_u(self):
		"""
		Optimize the profile matrix A and auxiliary u
		"""

		(niter, converged) = (0, self.MLE_CONV+1)
		old_elbo = -10**6
		old_A = np.copy(self.A)

		while (converged > self.MLE_CONV and niter < self.MLE_maxiter):
			## update auxilliary u
			self.u = (np.transpose(self.A[:, self.G]) *\
							self.suff_stats["exp_S"]).sum(axis=1)

			## update the constant coefficient in the gradient of A
			## E[tau_l*(S_il-0.5) - kappa_l*tau_l*w_il]
			self.coeffConst = self.suff_stats["coeffA"] * self.L
			## sum_l S_il * R_l / u_l, where sum is within cell type
			for k in xrange(self.K):
				itype = self.itype[k]
				self.coeffConst[:, k] -= ((self.suff_stats["exp_S"][itype,:] * \
						(self.SCrd/self.u)[itype, np.newaxis]).sum(axis=0))
			self.avg_coeffConst = np.mean(self.coeffConst, axis=0)

			## optimize A
			self.opt_A()
			# elbo = self.compute_elbo_A()
			# logging.debug("A: %.6f, ", 100*elbo)

			## convergence
			# elbo = self.compute_elbo_A()
			# converged = abs(elbo - old_elbo)
			converged = np.linalg.norm(self.A - old_A, 1)
			niter += 1
			old_A = np.copy(self.A)
			# logging.debug("u: %.6f, " , 100*elbo)

		logging.debug("\t\tOptimized A and u after %d iterations", niter)

		return niter

	# def opt_A_fp(self):
	# 	"""
	# 	Optimize the k-th column of A using fixed-point method
	# 	"""
	# 	old_A = np.copy(self.A)
	# 	(converged, niter) = (self.MLE_CONV+1, 0)
		
	# 	## pre-calculate to avoid redundant computation
	# 	term_ratio = self.coeffConst / self.coeffA
	# 	avg_ratio = np.mean(term_ratio, axis=0)
	# 	avg_inv_coeffA = np.mean(1.0 / self.coeffA, axis=0)

	# 	while (converged > self.MLE_CONV) and (niter < self.MLE_maxiter):
			
	# 		## first solve for the lagrangian parameter lamk
	# 		# term_Ainv = self.coeffAinv / (old_A * self.coeffA)
	# 		# lagrange = 1.0/self.N - np.mean(term_Ainv, axis=0) - avg_ratio
	# 		# lagrange /= avg_inv_coeffA 
	# 		# new_A = term_Ainv + term_ratio + lagrange[np.newaxis, :] / self.coeffA

	# 		new_A = self.coeffAinv/(old_A * self.coeffA) + term_ratio
	# 		new_A = self.get_proj_A(new_A)

	# 		## update
	# 		niter += 1
	# 		converged = np.linalg.norm(new_A - old_A, 1)
	# 		old_A = np.copy(new_A)

	# 	self.A = np.copy(new_A)
	# 	return niter

	def opt_A(self):
		"""
		Optimize profile matrix A.
		"""
		niters = np.zeros([self.K])
		for k in xrange(self.K):
			niters[k] = self.opt_Ak(k)
		return niters


	def opt_Ak(self, k):
		"""
		Optimize the k-th column of A using projected gradient descent
		with backtracking
		"""
		old_Ak = np.copy(self.A[:, k])
		(converged, niter) = (self.MLE_CONV+1, 0)

		while (converged > self.MLE_CONV) and (niter < self.MLE_maxiter):	
			## 1 step of projected gradient descent	
			(new_Ak, obj_new, stepsize) = self.backtracking(
						old_val=old_Ak, 
						grad_func=lambda Ak: -self.get_grad_A(Ak, k), 
						obj_func=lambda Ak: -self.get_obj_A(Ak, k), 
						proj_func=lambda Ak: simplex_proj(Ak, self.min_A))		
			self.A[:, k] = np.copy(new_Ak)
			## update
			niter += 1
			converged = np.linalg.norm(new_Ak - old_Ak, 1)
			old_Ak = np.copy(new_Ak)

		return niter


	def opt_alpha(self):
		"""Optimize alpha using gradient descent"""
		(converged, niter) = (self.MLE_CONV+1, 0)
		while (converged > self.MLE_CONV) and (niter < self.MLE_maxiter):
			old_alpha = np.copy(self.alpha)
			## pgradient descent
			self.alpha = self.backtracking(old_val=old_alpha, 
					grad_func=lambda alpha: (-self.get_grad_alpha(alpha)), 
					obj_func=lambda alpha: (-self.get_obj_alpha(alpha)))[0]
			## constraint: alpha>=1
			self.alpha = np.maximum(self.min_alpha, self.alpha)

			## update
			niter += 1
			converged = np.linalg.norm(self.alpha - old_alpha)

		logging.debug("\t\tOptimized alpha in %s iterations: ", niter)

		alpha_info = "\t\t\talpha = "
		for k in xrange(self.K):
			alpha_info += ("%.2f, " % self.alpha[k])
		logging.debug(alpha_info)

		return niter


	def backtracking(self, old_val, grad_func, obj_func, proj_func=None):
		"""Backtracking line search for (projected) gradient descent."""
		grad_old = grad_func(old_val)
		obj_old = obj_func(old_val)

		## x_new = x_old - t * G_t(x_old)
		stepsize = 0.1
		if proj_func is not None:
			new_val = proj_func(old_val - stepsize * grad_old)
		else:
			new_val = old_val - stepsize * grad_old

		Gt_old = (old_val - new_val) / stepsize
		obj_new = obj_func(new_val)
		while obj_new > (obj_old + (stepsize*0.5) * (Gt_old**2).sum() \
							- stepsize * np.dot(grad_old, Gt_old)):
			stepsize = stepsize * 0.5
			if proj_func is not None:
				new_val = proj_func(old_val - stepsize * grad_old)
			else:
				new_val = old_val - stepsize * grad_old

			Gt_old = (old_val - new_val) / stepsize
			obj_new = obj_func(new_val)

		return (new_val, obj_new, stepsize)


	def get_grad_alpha(self, alpha_val):
		"""Calculate the gradient of alpha: K x 1."""
		## exp_logW: K x 1, E[mean_j log W_kj]
		grad_alpha = self.suff_stats["exp_mean_logW"] + \
									psi(sum(alpha_val)) - psi(alpha_val)
		return grad_alpha


	def get_obj_alpha(self, alpha_val):
		"""Get the objective function value at given alpha_val"""
		obj_alpha = np.dot(self.suff_stats["exp_mean_logW"], alpha_val)
		obj_alpha += gammaln(sum(alpha_val)) - sum(gammaln(alpha_val))
		return obj_alpha


	def get_proj_A(self, A_val):
		"""Given A_new (N x K), project onto feasible sets"""
		proj_A_val = np.zeros(A_val.shape)
		for k in xrange(self.K):
			proj_A_val[:, k] = simplex_proj(A_val[:, k], self.min_A)
		return proj_A_val


	def get_grad_A(self, Ak, k):
		"""
		Calculate the gradient of k-th column of A: K x 1.
		To avoid overflow, consider f/L for single cell, f/M for bulk,
		and f/(M*L) for complete model 
		"""
		grad_A = np.zeros([self.N], dtype=float)

		if self.hasBK:
			## exp_Zik: E[mean_j Z_ijk], N x K
			grad_BK = self.suff_stats["exp_Zik"][:, k] / Ak
			if self.hasSC:
				grad_BK /= self.L
			grad_A += grad_BK

		if self.hasSC:
			## type_expr, coeffA, coeffAsq: N x K
			grad_SC = self.type_expr[:, k] / Ak
			grad_SC += self.suff_stats["coeffAsq"][:, k] * Ak
			grad_SC += self.suff_stats["coeffA"][:, k]
			## sum_l S_il * R_l / u_l, where sum is within cell type
			itype = self.itype[k]
			grad_SC -= ((self.suff_stats["exp_S"][itype,:] * \
					(self.SCrd/self.u)[itype, np.newaxis]).sum(axis=0)) / self.L
			if self.hasBK:
				grad_SC /= self.M
			grad_A += grad_SC

		grad_A /= float(self.N)

		return grad_A


	def get_obj_A(self, Ak, k):
		"""
		Get the objective function value at given A_val for k-th column of A.
		To avoid overflow, consider f/L for single cell, f/M for bulk,
		and f/(M*L) for complete model 
		"""
		obj_A = 0.0
		if self.hasBK:
			obj_A_BK = (self.suff_stats["exp_Zik"][:, k] * np.log(Ak)).sum()
			if self.hasSC:
				obj_A_BK /= self.L
			obj_A += obj_A_BK

		if self.hasSC:
			obj_A_SC = (self.type_expr[:, k] * np.log(Ak)).sum()
			obj_A_SC += (self.suff_stats["coeffAsq"][:, k] * np.square(Ak)).sum()/2.0
			obj_A_SC += (self.suff_stats["coeffA"][:, k] * Ak).sum()
			itype = self.itype[k]
			obj_A_SC -= (Ak * 
					(self.suff_stats["exp_S"][itype,:] * \
						(self.SCrd/self.u)[itype, np.newaxis] ).sum(axis=0)
							).sum()/self.L
			if self.hasBK:
				obj_A_SC /= self.M
			obj_A += obj_A_SC

		obj_A /= float(self.N)

		return obj_A

	def compute_elbo(self):
		"""Compute Evidence Lower Bound."""
		## the part envolving A and u
		elbo = self.compute_elbo_A()

		if self.hasBK:
			elbo_BK = self.get_obj_alpha(self.alpha)
			## remaining part:
			## sum_ijk Z_jik * log W_kj / M = sum_jk exp_Zjk * exp_logW
 			elbo_BK += (np.transpose(self.suff_stats["exp_Zjk"]) * \
 							self.suff_stats["exp_logW"]).sum() / self.M
 			if self.hasSC:
 				elbo_BK /= self.L
 			elbo += elbo_BK/float(self.N)

		if self.hasSC:
			elbo_SC = self.suff_stats["exp_elbo_const"]
			elbo_SC += np.log(self.pkappa[1]*self.ptau[1]) / 2.0
			if self.hasBK:
				elbo_SC /= self.M
			elbo += elbo_SC/float(self.N)

		return elbo	

	# def compute_logll(self):
	# 	"""Compute the log likelihood"""
	# 	logll = 0.0
	# 	for k in xrange(self.K):
	# 		logll += self.get_obj_A(self.A[:, k], k)	

	# 	return logll

	def compute_elbo_A(self):
		"""Only compute the part involving A and u,
			for optimizing A"""
		elbo = 0.0

		for k in xrange(self.K):
			elbo += self.get_obj_A(self.A[:, k], k)	
		if self.hasSC:
			elbo_SC = - np.mean(self.SCrd * np.log(self.u))
			if self.hasBK:
				elbo_SC /= self.M
			elbo += elbo_SC/float(self.N)

		return elbo	



