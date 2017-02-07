## 2016/07/19 Lingxue Zhu
## A class to perform Gibbs sampling

import logging
import numpy as np
from numpy import linalg 
import scipy as scp
from scipy import special
import pypolyagamma as ppg
import os

#######################
## A base class for Gibbs sampling
########################
class LogitNormalGibbs_base(object):

	def __init__(self, parameters):
		pass

	def init_gibbs(self, keepChain=False):
		"""initialize values of random variables"""
		pass

	def init_suffStats(self):
		"""initialize sufficient statistics"""
		pass

	def update_chain(self, isample):
		"""add current values to the i-th recorded sample"""
		pass

	def update_suffStats(self, sample):
		"""update sufficient stats using current values of latent variables"""
		pass

	def update_parameters(self, parameters):
		"""update parameters"""
		pass

	def gibbs_cycle(self):
		"""perform one cycle of Gibbs sampling"""
		pass

	## Main sampling iteration
	def gibbs(self, burnin=100, sample=100, thin=1, 
				keepChain=False):
		"""Gibbs sampling"""
		## initialize
		self.init_gibbs(keepChain)
		self.init_suffStats()
		isample = 0

		## burnin period
		logging.debug("\t\tBurnin period: %s samples..." % burnin)

		for giter in xrange(burnin):
			self.gibbs_cycle()
		## sampling
		logging.debug("\t\tBurnin finished. Gibbs sampling started...")
		for giter in xrange(sample*thin):
			self.gibbs_cycle()
			if giter % thin == 0:
				## update sufficient statistics
				self.update_suffStats(sample)
				## record samples
				if keepChain:
					self.update_chain(isample)
				## print progress for every 10 collected samples
				isample += 1
				if (isample % 50 == 0):
					logging.debug("\t\t\t%s/%s finished.", isample, sample)
		
		## finished
		logging.debug("\t\tGibbs samples finished: %s samples.", sample)


#######################
## Gibbs sampler for bulk data
#######################
class LogitNormalGibbs_BK(LogitNormalGibbs_base):

	## constructor: initialize with parameters
	def __init__(self, 
				A=np.empty([0,0]), ## profile matrix
				alpha=np.empty([0,0]), ## mixture propotion prior 
				BKexpr=np.empty([0,0]) ## M-by-N, bulk expression
				):
		## data: unchanged throughout sampling
		self.BKexpr = BKexpr
		(self.M, self.N, self.K) = (BKexpr.shape[0], A.shape[0], A.shape[1])
		## read depths
		self.BKrd = BKexpr.sum(axis=1)
		## parameters: can only be changed by self.update_parameters()
		self.A = np.array(A, dtype=float, copy=True)
		self.alpha = np.array(alpha, dtype=float, copy=True)
	

	def init_gibbs(self, keepChain=False):
		"""initialize latent variable values"""
		self.W = np.full([self.K, self.M], 1.0/self.K, dtype=float)
		self.Z = np.zeros([self.M, self.N, self.K])
		self.AW = np.dot(self.A, self.W)

		## store sampling results
		if keepChain:
			self.W_gibbs = np.zeros([sample, self.K, self.M], dtype=float)
			self.Z_gibbs = np.zeros([sample, self.M, self.N, self.K], dtype=float)


	def init_suffStats(self):
		"""initialize sufficient statistics"""
		## E[mean_j Z_jik] and E[mean_j log W_kj]
		self.exp_Zik = np.zeros([self.N, self.K], dtype=float)
		self.exp_Zjk = np.zeros([self.M, self.K], dtype=float)
		self.exp_logW = np.zeros([self.K, self.M], dtype=float)
		self.exp_W = np.zeros([self.K, self.M], dtype=float)


	def update_chain(self, isample):
		"""let current values to be the i-th recorded sample"""
		self.W_gibbs[isample, :, :] = self.W
		self.Z_gibbs[isample, :, :, :] = self.Z


	def update_suffStats(self, sample):
		"""update sufficient stats using current values of latent variables"""
		## E[mean_j Z_jik]
		self.exp_Zik += (self.Z).mean(axis=0) / float(sample)
		## E[sum_i Z_jik]
		self.exp_Zjk += (self.Z).sum(axis=1) / float(sample)
		## E[mean_j log W_kj]
		self.exp_logW += np.log(self.W) / float(sample)
		## E[W]
		self.exp_W += self.W / float(sample)


	def update_parameters(self, A, alpha):
		"""update parameters"""
		self.A = np.array(A, dtype=float, copy=True)
		self.alpha = np.array(alpha, dtype=float, copy=True)


	#########################
	## draw Gibbs samples
	#########################
	def gibbs_cycle(self):
		"""perform one cycle of Gibbs sampling"""
		## note: Z is not carefully initialized
		## so have to draw Z first!
		self.draw_Z()
		self.draw_W()


	def draw_Z(self):
		"""Z: M x N x K, counts"""
		for j in xrange(self.M):
			for i in range(self.N):
				pval = self.W[:, j]*self.A[i, :]
				self.Z[j, i, :] = np.random.multinomial(n=self.BKexpr[j, i],
									pvals = pval/self.AW[i, j])


	def draw_W(self):
		"""W: K x M, proportions"""
		post_alpha = self.Z.sum(axis=1)
		for j in xrange(self.M):
			self.W[:, j] = np.random.dirichlet(self.alpha + post_alpha[j, :])
		## update AW: N x M
		self.AW = np.dot(self.A, self.W)



#######################
## Gibbs sampler for single cell
#######################
class LogitNormalGibbs_SC(LogitNormalGibbs_base):

	## constructor: initialize with parameters
	def __init__(self, 
				A=np.empty([0,0]), ## profile matrix
				pkappa=np.empty([2]), ## [mean, var] for kappa
				ptau=np.empty([2]), ## [mean, var] for tau
				SCexpr=np.empty([0,0]), ## L-by-N, single cell expression
				G=np.empty([0]) ## L-by-1, single cell types
				):
		## data: never changed
		(self.SCexpr, self.G, self.L) = (SCexpr, G, SCexpr.shape[0]) 
		(self.N, self.K) = A.shape
		self.SCrd = SCexpr.sum(axis=1) ## read depths
		self.itype = [] ## cell ids in each type
		for k in xrange(self.K):
			self.itype += [np.where(self.G == k)[0]]
		## parameters: can only be changed by self.update_parameters()
		self.A = np.array(A, dtype=float, copy=True)
		self.pkappa = np.array(pkappa, dtype=float, copy=True)
		self.ptau = np.array(ptau, dtype=float, copy=True)
		## zero-expressed entries
		self.izero = np.where(self.SCexpr==0)
		## for sampling from Polya-Gamma
		# self.ppgs = ppg.PyPolyaGamma(seed=0)
		num_threads = ppg.get_omp_num_threads()
		seeds = np.random.randint(2**16, size=num_threads)
		self.ppgs = self.initialize_polya_gamma_samplers()


	def initialize_polya_gamma_samplers(self):
		if "OMP_NUM_THREADS" in os.environ:
			self.num_threads = int(os.environ["OMP_NUM_THREADS"])
		else:
			self.num_threads = ppg.get_omp_num_threads()
		assert self.num_threads > 0

		# Choose random seeds
		seeds = np.random.randint(2**16, size=self.num_threads)
		return [ppg.PyPolyaGamma(seed) for seed in seeds]


	def init_gibbs(self, keepChain=False):
		"""initialize latent variable values"""
		self.kappa = np.full([1,self.L], self.pkappa[0], dtype=float)
		self.tau = np.full([1,self.L], self.ptau[0], dtype=float)
		self.S = np.reshape(np.random.binomial(1, 0.5, size=self.L*self.N), 
							[self.L, self.N])
		## note: use broadcasting
		self.psi = np.transpose(self.kappa + self.tau * self.A[:, self.G])
		self.w = np.ones([self.L, self.N], dtype=float)
		## when expression > 0, it's known for sure that S=1
		ipos = np.where(self.SCexpr>0)
		self.S[ipos] = 1
		## keep track of A[:, G]*S to reduce computation time
		self.sum_AS = (np.transpose(self.A[:, self.G]) * self.S).sum(axis=1)
		## store sampling results
		if keepChain:
			self.kappa_gibbs = np.zeros([sample, self.L], dtype=float)
			self.tau_gibbs = np.zeros([sample, self.L], dtype=float)
			self.w_gibbs = np.zeros([sample, self.L, self.N], dtype=float)
			self.S_gibbs = np.zeros([sample, self.L, self.N], dtype=float)

	def init_suffStats(self):
		"""initialize sufficient statistics to be zeros"""
		## posterior expectations
		self.exp_S = np.zeros([self.L, self.N], dtype=float) ## E[S]
		self.exp_kappa = np.zeros([1, self.L], dtype=float) ## E[kappa]
		self.exp_tau = np.zeros([1, self.L], dtype=float) ## E[tau]
		self.exp_kappasq = np.zeros([1, self.L], dtype=float) ## E[kappa^2]
		self.exp_tausq = np.zeros([1, self.L], dtype=float) ## E[tau^2]
		## part of coefficient for A: E[tau_l*(S_il-0.5) - kappa_l*tau_l*w_il]
		self.coeffA = np.zeros([self.N, self.K], dtype=float)
		## coefficient for A^2: E[- tau_l^2 * w_il]
		self.coeffAsq = np.zeros([self.N, self.K], dtype=float)
		## elbo that doesn't involve A
		self.exp_elbo_const = 0


	def update_chain(self, isample):
		"""make current values to be the i-th recorded sample"""
		self.kappa_gibbs[isample, :] = self.kappa
		self.tau_gibbs[isample, :] = self.tau
		self.w_gibbs[isample, :, :] = self.w
		self.S_gibbs[isample, :, :] = self.S


	def update_suffStats(self, sample):
		"""Update sufficient stats using current values of latent variables"""
		self.exp_S += self.S / float(sample)
		self.exp_kappa += self.kappa / sample
		self.exp_tau += self.tau / sample
		self.exp_kappasq += np.square(self.kappa) / sample
		self.exp_tausq += np.square(self.tau) / sample

		## sum_il E[- kappa_l^2 * w_il/2 + (S_il-0.5)*kappa_l ] / L 
		self.exp_elbo_const += (-self.w * \
				np.transpose(np.square(self.kappa))).sum() / (2.0*self.L*sample)
		self.exp_elbo_const += ((self.S - 0.5) * np.transpose(self.kappa)).sum()/ \
									(self.L*sample)

		## E[tau_l*(S_il-0.5) - kappa_l*tau_l*w_il]
		coeffA = (self.S - 0.5) * np.transpose(self.tau) - \
					self.w * np.transpose(self.tau * self.kappa)
		## E[- tau_l^2 * w_il]
		coeffAsq = (-self.w * np.transpose(np.square(self.tau)))
		## sum over l and divide by L
		for k in xrange(self.K):
			self.coeffA[:, k] += coeffA[self.itype[k],:].sum(axis=0) / \
													 float(sample*self.L)
			self.coeffAsq[:, k] += coeffAsq[self.itype[k],:].sum(axis=0) / \
													 float(sample*self.L)


	def update_parameters(self, A, pkappa, ptau):
		"""update parameters"""
		self.A = np.array(A, dtype=float, copy=True)
		self.pkappa = np.array(pkappa, dtype=float, copy=True)
		self.ptau = np.array(ptau, dtype=float, copy=True)

		
	#########################
	## draw Gibbs samples
	#########################
	def gibbs_cycle(self):
		"""One cycle through latent variables in Gibbs sampling"""
		self.draw_w()
		self.draw_S()
		self.draw_kappa_tau()
		self.update_psi()

	def update_psi(self):
		"""psi: L-by-N; logistic(psi) is the dropout probability"""
		self.psi = np.transpose(self.kappa + self.tau * self.A[:, self.G])

	def draw_w(self):
		"""w: L-by-N; augmented latent variable"""
		ns = np.ones(self.N, dtype=np.float)
		## draw polya gamma parallelly
		for l in xrange(self.L):
			ppg.pgdrawvpar(self.ppgs, ns, self.psi[l, :], self.w[l, :])


	def draw_S(self):
		"""S: L-by-N; binary variables"""
		## only update the entries where self.SCexpr==0
		for index in xrange(len(self.izero[0])):
			(l, i) = (self.izero[0][index], self.izero[1][index])
			A_curr = self.A[i, self.G[l]]

			sum_other = self.sum_AS[l] - A_curr * self.S[l, i]
			if sum_other == 0:
				b = scp.special.expit(self.psi[l][i])
			else:
				b = scp.special.expit(self.psi[l][i] - 
						self.SCrd[l] * np.log(1 + A_curr/sum_other))
			## note: scp.stats.bernoulli.rv is slow!!!
			self.S[l][i] = np.random.binomial(1, b)
			## update sum_AS
			self.sum_AS[l] = sum_other + A_curr * self.S[l, i]


	def draw_kappa_tau(self):
		"""kappa, tau: scalars that defines psi"""
		for l in xrange(self.L):
			A_curr = self.A[:, self.G[l]]
			## precision matrix
			offdiag = sum(self.w[l, :] * A_curr)
			diag = [sum(self.w[l, :]) + self.pkappa[1],
				sum(self.w[l, :] * np.square(A_curr)) + self.ptau[1]]
			PP = np.array([[diag[0], offdiag], [offdiag, diag[1]]])

			## PP * mP = bP: solve for the mean vector mP
			bP = np.array([sum(self.S[l,:]) - self.N/2.0 +\
								self.pkappa[0]*self.pkappa[1],
							self.sum_AS[l] - 0.5 + \
								self.ptau[0]*self.ptau[1]])
			mP = np.linalg.solve(PP, bP)

			## draw (kappa, tau) ~ Gaussian(mP, PP^-1)
			newdraw = np.random.multivariate_normal(mP, np.linalg.inv(PP))
			(self.kappa[0, l], self.tau[0, l]) = newdraw


##########################
## test & time
##########################
# import time

# data = np.load("simulated.npz")

# myGibbs = LogitNormalGibbs_SC(A=data["A"], 
# 			pkappa=data["pkappa"], ptau=data["ptau"],
# 			SCexpr=data["SCexpr"], G=data["G"])

# starttime = time.time()
# myGibbs.gibbs(burnin=0, sample=100)
# print time.time()-starttime
## N=300, L=150, 100 Gibbs samples: 29 sec


# myGibbs = LogitNormalGibbs_BK(A=data["A"], 
# 			alpha=data["alpha"], BKexpr=data["BKexpr"])

# starttime = time.time()
# myGibbs.gibbs(burnin=0, sample=100)
# print time.time()-starttime

## N=300, M=200, 100 Gibbs samples: 42 sec


# ############# profiling ################
# import cProfile
# cProfile.run('myGibbs.gibbs(burnin=0, sample=10)')

################ check results ################
## BK
# W = data["W"]
# W = W[:, 0, :]
# print W.shape
# print myGibbs.exp_W.shape
# corrcoeff = np.corrcoef(W[:, 2], myGibbs.exp_W[2, :])

## SC
# idrop = np.where(data["SCexpr"] == 0)
# istruct0 = np.where(np.logical_and(data["SCexpr"] == 0, data["S"] == 1))

# np.mean(myGibbs.exp_S[idrop])
# np.mean(myGibbs.exp_S[istruct0])

# import matplotlib as mpl
# import matplotlib.pyplot as plt

# fig = plt.figure(1, figsize=(9, 6))
# ax = fig.add_subplot(111)
# SCbox = ax.boxplot([myGibbs.exp_S[idrop], myGibbs.exp_S[istruct0]])
# fig.savefig("SC_boxplot.png", bbox_inches="tight")
