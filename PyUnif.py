## 2016/07/21 Lingxue Zhu
## Gibbs EM for modeling bulk and single cell RNA seq data
##
## #################
##  -- parameters:
##     A: N x K, gene expression profiles; colSums(A) = 1
##     G: {1, ..., K}^L, cell type
##     mu_kappa, mu_tau, sigma_kappa^-2, sigma_tau^-2
##     alpha: K x 1
##
## Signel cell model:
##  -- latent variables (same as a Bayesian logistic regression model):
##     (kappa_l, tau_l) ~ N( (mu_kappa, mu_tau), diag(sigma_kappa^2, sigma_tau^2) )
##     S_li ~ Bernoulli( logistic(Psi_li) )
##      where Psi_li = kappa_l + tau_l * A[i, G[l]]
##
##  -- observed data:
##     Y_l ~ Multinomial(R_l, probs_l): N x 1
##      where R_l = sum(Y_l) read depth; probs_l = normalize(A[, G[l]] * S[, l])
## 
## Bulk model:
##  -- latent variable:
##    W_j ~ Dirichlet(alpha): K x 1
##    
##  -- observed data:
##    X_j ~ Multinomial(R_j, A W_j): N x 1
##    
## #################
## Gibbs sampling:
## 
## Single cell: use data augmentation (Polson and Scott (2013))
##  -- w_li ~ PG(1, 0), Polya-Gamma latent variables
##
##  -- Key: the likelihood can be written as:
##   p(kappa, tau | mu, sigma) * p(S | kappa, tau, A) * p(Y | S, A)
##   \propto p(kappa, tau | mu, sigma) * ( E_w{ f(w, kappa, tau, S, A)} ) * p(Y | S, A)
##       (where E_w is the expectation taken over w ~ PG(1, 0))
##   \propto integral_w{  p(kappa, tau | mu, sigma) * f(w, kappa, tau, S, A) * p(w) * p(Y | S, A)}
##
##   hence we get a "complete" likelihood for p(kappa, tau, w, S, Y | mu, sigma, A)
##   and we get the target posterior after marginalize out w
##
## Bulk: use alternative parametrization:
##  -- W_j ~ Dirichlet(alpha): K x 1
##     Z'_rj ~ Multinomial(1, W_j): K x 1, for r=1, ..., R_j
##     d_rj ~ Multinomial(1, A Z'_rj): N x 1
##     X_j = sum_r d_rj
##
##  -- Key: note that we don't need to get all samples for d and Z' 
##         Especially, for all feasible d, let
##               Z_ij = sum_{r: d_rj=i} Z'_rj
##         then
##            Z_ij | d, X, W ~ Multinomial(X_ij, normalized(W_j * A[i,:]))
##            W_j | d, X, Z ~ Dirichlet( alpha + sum_i Z_ij )
##         
## ##################

from utils import * ## helper functions
from e_step_gibbs import * ## Gibbs samplers
from m_step import * ## M-step
import logging
import numpy as np
from numpy import linalg 
import scipy as scp
from scipy.special import psi, gammaln
import json
import sys, os
import datetime

class LogitNormalGEM(object):

	def __init__(self, 
				BKexpr=None, ## M-by-N, bulk expression
				SCexpr=None, ## L-by-N, single cell expression
				G=None, ## L-by-1, single cell types
				K=3, ## number of cell types
				hasBK=True, hasSC=True, ## model specification
				init_A = None, ## N-by-K, initial value for A
				min_A=1e-6, ## minimal value of A; must be positive
				init_alpha = None, ## K-by-1, initial value for alpha
				est_alpha=True, ## whether to estimate alpha or use initial value
				init_pkappa = None, init_ptau = None, ## 2-by-1, mean and 1/var	
				## for Gibbs sampling
				burnin=200, sample=200, thin=1, Gibbs_verbose=True, 
				## for M-step
				MLE_CONV=1e-6, MLE_maxiter=100, MLE_verbose=True,
				## for EM procedure
				EM_CONV=1e-3, EM_maxiter=100, EM_verbose=True
				):

		(self.K, self.min_A, self.est_alpha) = (K, min_A, est_alpha)
		(self.EM_CONV, self.EM_maxiter) = (EM_CONV, EM_maxiter)
		(self.SCexpr, self.G, self.BKexpr) = (SCexpr, G, BKexpr)
		(self.hasBK, self.hasSC, self.EM_verbose) = (hasBK, hasSC, EM_verbose)

		self.init_para(init_A, init_pkappa, init_ptau, init_alpha)
		self.init_gibbs(burnin, sample, thin, Gibbs_verbose)
		self.init_mle(MLE_CONV, MLE_maxiter, MLE_verbose)


	def estep_gibbs(self):
		## E-step: gibbs sampling and record suff stats
		self.suff_stats = {} ## sufficient statistics

		if self.hasSC:
			logging.info("E-step for single cells started.")

			self.Gibbs_SC.update_parameters(self.A, self.pkappa, self.ptau)
			self.Gibbs_SC.gibbs(burnin=self.burnin, sample=self.sample, 
				thin=self.thin)

			self.suff_stats["exp_S"] = self.Gibbs_SC.exp_S
			self.suff_stats["exp_kappa"] = self.Gibbs_SC.exp_kappa
			self.suff_stats["exp_tau"] = self.Gibbs_SC.exp_tau
			self.suff_stats["exp_kappasq"] = self.Gibbs_SC.exp_kappasq
			self.suff_stats["exp_tausq"] = self.Gibbs_SC.exp_tausq
			self.suff_stats["coeffA"] = self.Gibbs_SC.coeffA
			self.suff_stats["coeffAsq"] = self.Gibbs_SC.coeffAsq
			self.suff_stats["exp_elbo_const"] = self.Gibbs_SC.exp_elbo_const

		if self.hasBK:
			logging.info("E-step for bulk samples started.")

			self.Gibbs_BK.update_parameters(self.A, self.alpha)
			self.Gibbs_BK.gibbs(burnin=self.burnin, sample=self.sample, 
				thin=self.thin)

			self.suff_stats["exp_Zik"] = self.Gibbs_BK.exp_Zik
			self.suff_stats["exp_Zjk"] = self.Gibbs_BK.exp_Zjk
			self.suff_stats["exp_W"] = self.Gibbs_BK.exp_W
			self.suff_stats["exp_logW"] = self.Gibbs_BK.exp_logW


	def gem(self):
		(converged, niter) = (self.EM_CONV+1, 0)
		old_elbo = -10**6
		path_elbo = np.array([])

		while (abs(converged) > self.EM_CONV) and (niter < self.EM_maxiter):
			## E-step: gibbs sampling and record suff stats
			self.estep_gibbs()
			## update suff stats and calculate elbo
			self.mle.update_suff_stats(self.suff_stats) ## update suff stats
			elbo = self.mle.compute_elbo()
			logging.info("E-step finished: elbo=%.6f", elbo)

			## M-step
			logging.info("M-step started.")	
			if self.hasSC:
				self.mle.opt_kappa_tau()
				self.pkappa = self.mle.pkappa
				self.ptau = self.mle.ptau
			if self.hasBK and self.est_alpha:
				niter_alpha = self.mle.opt_alpha()
				self.alpha = self.mle.alpha

			niter_A = self.mle.opt_A_u()
			self.A = self.mle.A

			## converged?
			elbo = self.mle.compute_elbo()
			logging.info("M-step finished: elbo=%.6f", elbo)

			converged = abs(elbo - old_elbo)
			old_elbo = elbo
			niter += 1
			path_elbo = np.append(path_elbo, [elbo])

			if self.EM_verbose:
				logging.info("%d-th EM iteration finished, ELBO=%.6f", niter, elbo)
		
		self.path_elbo = path_elbo
		return (niter, elbo, converged, path_elbo)


	def init_mle(self, MLE_CONV, MLE_maxiter, MLE_verbose):
		"""Initialize the class for M-step"""
		self.mle = LogitNormalMLE(start_A=self.init_A, 
				start_alpha=self.init_alpha, 
				BKexpr=self.BKexpr, SCexpr=self.SCexpr,
				G=self.G, K=self.K, hasBK=self.hasBK, hasSC=self.hasSC,
				init_A = self.init_A, init_alpha = self.init_alpha,
				init_pkappa = self.init_pkappa, init_ptau=self.init_ptau,
				min_A=self.min_A, MLE_CONV=MLE_CONV, MLE_maxiter=MLE_maxiter, 
				MLE_verbose=MLE_verbose)


	def init_para(self, init_A, init_pkappa, init_ptau, init_alpha):
		"""Initialize parameters for model"""
		if self.hasSC: ## SC parameters
			(self.L, self.N) = self.SCexpr.shape	
			self.init_para_SC(init_pkappa, init_ptau)
			self.init_alpha = None ## need this to initialize self.mle

		if self.hasBK: ## bulk parameters
			(self.M, self.N) = self.BKexpr.shape
			self.init_para_BK(init_alpha)

		self.init_para_A(init_A) ## profile matrix


	def init_para_A(self, init_A):
		"""Initialize profile matrix A"""
		if init_A is not None:
			self.init_A = init_A
		elif self.hasSC:	
			self.init_A = np.zeros([self.N, self.K])
			## standardize to get proportions of reads and take means
			stdSCexpr = std_row(self.SCexpr)
			for k in xrange(self.K):
				self.init_A[:, k] = (stdSCexpr[np.where(self.G==k)]).mean(axis=0)
		else:
			self.init_A = np.zeros([self.N, self.K])
			## standardize to get proportions of reads and take means
			BKmean = std_row(self.BKexpr).mean(axis=0)
			for k in xrange(self.K):
				self.init_A[:, k] = BKmean

		## project A to simplex, with constraint A >= min_A
		for k in xrange(self.K):
			self.init_A[:, k] = simplex_proj(self.init_A[:, k], self.min_A)

		self.A = np.copy(self.init_A)


	def init_para_SC(self, init_pkappa, init_ptau):
		"""Initialize the parameters for single cell model"""
		## parameters for dropouts
		if (init_pkappa is not None) and (len(init_pkappa)==2):
			self.init_pkappa = init_pkappa
		else:
			self.init_pkappa = np.array([-1., 10.], dtype=float)

		if (init_ptau is not None) and (len(init_ptau)==2):
			self.init_ptau = init_ptau
		else:
			self.init_ptau = np.array([300., 0.1], dtype=float)
		self.pkappa = np.copy(self.init_pkappa)
		self.ptau = np.copy(self.init_ptau)


	def init_para_BK(self, init_alpha):
		"""Initialize the parameters for bulk model"""
		if (init_alpha is not None) and (len(init_alpha)==self.K):
			self.init_alpha = init_alpha
		else:
			self.init_alpha = np.ones([self.K])
		self.alpha = np.copy(self.init_alpha)


	def init_gibbs(self, burnin, sample, thin, Gibbs_verbose):
		"""Initialize Gibbs sampler"""
		(self.burnin, self.sample, self.thin) = (burnin, sample, thin)
		self.Gibbs_verbose = Gibbs_verbose

		if self.hasSC:
			self.Gibbs_SC = LogitNormalGibbs_SC(A=self.init_A, 
					pkappa=self.init_pkappa, ptau=self.init_ptau, 
					SCexpr=self.SCexpr, G=self.G)
			self.Gibbs_SC.init_gibbs()

		if self.hasBK:
			self.Gibbs_BK =  LogitNormalGibbs_BK(A=self.init_A, 
					alpha=self.init_alpha, BKexpr=self.BKexpr)
			self.Gibbs_BK.init_gibbs()


###################################
## save results to file
###################################
def gem2csv(dirname, filename, gem):
	prefix = dirname + "/" + filename	
	mtx2csv(prefix + 'est_A.csv', gem.A)
	mtx2csv(prefix + 'path_elbo.csv', gem.path_elbo)

	if gem.hasSC:
		mtx2csv(prefix + 'exp_S.csv', gem.suff_stats['exp_S'])
		mtx2csv(prefix + 'est_pkappa.csv', gem.pkappa)
		mtx2csv(prefix + 'est_ptau.csv', gem.ptau)
	if gem.hasBK:
		mtx2csv(prefix + 'est_alpha.csv', gem.alpha)
		mtx2csv(prefix + 'exp_W.csv', gem.suff_stats['exp_W'])


def mtx2csv(filename, nparray):
	with open(filename, 'w') as handle:
		np.savetxt(handle, nparray, delimiter=',')


###############
## read data from files
###############

if __name__ == "__main__":
	if len(sys.argv) != 3:
		logging.warning("usage: python LogitNormalGEM.py setting_file log_directory")
		sys.exit(1)

	## command line input: tmp directory and file signature
	setting_file = str(sys.argv[1])
	setting = json.load(open(setting_file, mode="r"))	

	## set up logging to a file
	logdir = str(sys.argv[2])
	if not os.path.exists(logdir):
		os.makedirs(logdir)

	i = datetime.datetime.now()
	logging.basicConfig(filename= "%s/LogitNormalGEM_%s.log" % 
					(logdir, setting["filesig"]), 
					level=logging.DEBUG,
					## record message time
					#format = "%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
					)

	## read in parameters
	if "BKexpr_file" in setting:
		BKexpr = np.loadtxt(setting["BKexpr_file"], dtype=float, delimiter=",")
	else:
		BKexpr = None

	if "SCexpr_file" in setting:
		SCexpr = np.loadtxt(setting["SCexpr_file"], dtype=float, delimiter=",")
	else:
		SCexpr = None

	if "G_file" in setting:
		G = np.loadtxt(setting["G_file"], dtype=int, delimiter=",")
	else:
		G = None

	if ("init_A_file") in setting:
		init_A = np.loadtxt(setting["init_A_file"], dtype=float, delimiter=",")
	else:
		init_A = None
	if ("init_alpha_file") in setting:
		init_alpha = np.loadtxt(setting["init_alpha_file"], dtype=float, delimiter=",")
	else:
		init_alpha = None
	if ("init_pkappa_file") in setting:
		init_pkappa = np.loadtxt(setting["init_pkappa_file"], dtype=float, delimiter=",")
	else:
		init_pkappa = None
	if ("init_ptau_file") in setting:
		init_ptau = np.loadtxt(setting["init_ptau_file"], dtype=float, delimiter=",")
	else:
		init_ptau = None

	## perform Gibbs-EM
	myGEM = LogitNormalGEM(
					BKexpr=BKexpr, SCexpr=SCexpr, G=G, K=setting["K"], 
					hasBK=setting["hasBK"], hasSC=setting["hasSC"],
					init_A=init_A, min_A=setting["min_A"],
					init_alpha=init_alpha, 
					est_alpha=setting["est_alpha"],
					init_pkappa=init_pkappa, init_ptau=init_ptau,
					burnin=setting["burnin"], sample=setting["sample"], 
					thin=setting["thin"], 
					MLE_CONV=setting["MLE_CONV"], MLE_maxiter=setting["MLE_maxiter"], 
					EM_CONV=setting["EM_CONV"], EM_maxiter=setting["EM_maxiter"])
	(niter, elbo, converged, path_elbo) = myGEM.gem()

	## save results
	gem2csv(setting["tmpdir"], setting["filesig"], myGEM)



