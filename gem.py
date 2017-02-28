##
## Gibbs EM for modeling bulk and single cell RNA seq data
##
## Copyright Lingxue Zhu (lzhu1@cmu.edu).
## All Rights Reserved.
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

class LogitNormalGEM(object):
    """
    Gibbs-EM algorithm for inference and parameter estimation
    in joint modeling of bulk and single cell RNA seq data.
    """

    def __init__(self, BKexpr=None, SCexpr=None, K=3, G=None, iMarkers=None,
                init_A = None, min_A=1e-6, init_alpha = None, est_alpha=True, 
                init_pkappa = None, init_ptau = None, 
                burnin=200, sample=200, thin=1,
                burnin_bk=50, sample_bk=5,
                MLE_CONV=1e-6, MLE_maxiter=100,
                EM_CONV=1e-3, EM_maxiter=100):
        """
        Args:
            BKexpr: M-by-N np matrix, bulk RNA-seq counts.
                    The default is "None", where only single cell data is modeled.
            SCexpr: L-by-N np matrix, single cell RNA-seq counts.
                    The default is "None", where only bulk data is modeled.
            K: an integer indicating the total number of cell types.
            G: L-by-1 np vector, each element takes values from {0, ..., K-1},
                    indicating the cell type for each single cell. 
                    This value must be provided along with "SCexpr".
            iMarkers: a matrix with 2 columns. First column: indices of marker genes;
                    second column: cell types that the genes mark
            init_A: (optional) the initial value of the profile matrix A.
                    The default is to use sample mean of "SCexpr" in each type,
                    if available; otherwise, the sample mean of "BKexpr" 
                    with perturbation is used.
            min_A: (optional) lower-bound of the entries in the profile matrix A.
                    This must be a small positive number.
            init_alpha: K-by-1 vector, the initial value of alpha,
                    the hyper-parameter for Dirichlet prior.
            est_alpha: boolean, if "True" then the em-algorithm estimates alpha,
                    otherwise the algorithm takes "init_alpha" as a fixed prior.
            init_pkappa: (optional) 2-by-1 vector, the initial value of the 
                    mean and variance in the Normal prior for kappa.
            init_ptau: (optional) 2-by-1 vector, the initial value of the mean and 
                    variance in the Normal prior for tau.
            burin: an integer specifying the burn-in length in Gibbs sampling.
            sample: an integer specifying the number of Gibbs samples to keep.
            thin: an integer specifying the thinning steps in Gibbs sampling.
            MLE_CONV: the convergence criteria in m-step.
            MLE_maxiter: the maximal number of interations in m-step.
            EL_CONV: the convergence criteria for the EM-algorithm.
            EL_maxiter: the maximal number of interations in the EM-algorithm.
        """
        self.hasBK, self.hasSC = BKexpr is not None, SCexpr is not None
        self.K, self.min_A, self.est_alpha = K, min_A, est_alpha
        self.EM_CONV, self.EM_maxiter = EM_CONV, EM_maxiter
        self.burnin_bk, self.sample_bk = burnin_bk, sample_bk
        self.burnin, self.sample, self.thin = burnin, sample, thin
        self.MLE_CONV, self.MLE_maxiter = MLE_CONV, MLE_maxiter
        self.SCexpr, self.G, self.BKexpr = SCexpr, G, BKexpr
        self.iMarkers = iMarkers

        if self.hasSC:
            self.itype = [] ## cell ids in each type
            for k in xrange(self.K):
                self.itype += [np.where(self.G == k)[0]]
        else:
            self.itype = None
 
        self.init_para(init_A, init_pkappa, init_ptau, init_alpha)


    def estep_gibbs(self):
        ## E-step: gibbs sampling and record suff stats
        self.suff_stats = {} ## sufficient statistics

        if self.hasSC:
            logging.debug("\tE-step for single cells started.")

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
            self.suff_stats["sc_exp_elbo_const"] = self.Gibbs_SC.exp_elbo_const

        if self.hasBK:
            logging.debug("\tE-step for bulk samples started.")

            self.Gibbs_BK.update_parameters(self.A, self.alpha)
            self.Gibbs_BK.gibbs(burnin=self.burnin_bk, sample=self.sample_bk,
                thin=self.thin)

            self.suff_stats["exp_Zik"] = self.Gibbs_BK.exp_Zik
            self.suff_stats["exp_Zjk"] = self.Gibbs_BK.exp_Zjk
            self.suff_stats["exp_W"] = self.Gibbs_BK.exp_W
            self.suff_stats["exp_logW"] = self.Gibbs_BK.exp_logW


    def gem(self):
        (converged, niter) = (self.EM_CONV+1, 0)
        old_elbo = -10**6
        path_elbo = np.array([])

        ## initialize
        self.init_gibbs()
        self.init_mle()

        ## gem
        while (abs(converged) > self.EM_CONV) and (niter < self.EM_maxiter):
            logging.info("%d-th EM iteration started...", niter+1)
            ## E-step: gibbs sampling and record suff stats
            self.estep_gibbs()
            ## update suff stats and calculate elbo
            self.mle.update_suff_stats(self.suff_stats) ## update suff stats
            elbo = self.mle.compute_elbo()
            logging.info("\tE-step finished: elbo=%.6f", elbo)

            ## M-step
            # logging.info("\tM-step started...")
            if self.hasSC:
                self.mle.opt_kappa_tau()
                self.pkappa = self.mle.pkappa
                self.ptau = self.mle.ptau
                logging.debug("\t\tptau = (%.f, %.f)", self.ptau[0], self.ptau[1])

            if self.hasBK and self.est_alpha:
                niter_alpha = self.mle.opt_alpha()
                self.alpha = self.mle.alpha

            niter_A = self.mle.opt_A_u()
            self.A = self.mle.A

            ## converged?
            elbo = self.mle.compute_elbo()
            logging.info("\tM-step finished: elbo=%.6f", elbo)

            converged = abs(elbo - old_elbo)
            old_elbo = elbo
            niter += 1
            path_elbo = np.append(path_elbo, [elbo])

            # logging.info("%d-th EM iteration finished, ELBO=%.6f", niter, elbo)
        
        self.path_elbo = path_elbo
        return (niter, elbo, converged, path_elbo)


    def init_mle(self):
        """Initialize the class for M-step"""
        self.mle = LogitNormalMLE(BKexpr=self.BKexpr, SCexpr=self.SCexpr,
                G=self.G, K=self.K, itype=self.itype,
                hasBK=self.hasBK, hasSC=self.hasSC,
                init_A = self.init_A, init_alpha = self.init_alpha,
                init_pkappa = self.init_pkappa, init_ptau=self.init_ptau,
                min_A=self.min_A, MLE_CONV=self.MLE_CONV, MLE_maxiter=self.MLE_maxiter)


    def init_para(self, init_A, init_pkappa, init_ptau, init_alpha):
        """Initialize parameters for model"""
        if self.hasSC: ## SC parameters
            (self.L, self.N) = self.SCexpr.shape    
            self.init_para_SC(init_pkappa, init_ptau)
            self.init_alpha = None ## need this to initialize self.mle

        if self.hasBK: ## bulk parameters
            (self.M, self.N) = self.BKexpr.shape
            self.init_para_BK(init_alpha)
            ## need this to initialize self.mle
            self.init_pkappa = np.array([-1., 0.01], dtype=float)
            self.init_ptau = np.array([self.N, 0.01*self.N], dtype=float)

        self.init_para_A(init_A) ## profile matrix


    def init_para_A(self, init_A):
        """Initialize profile matrix A"""
        if self.hasSC:
            ## standardize to get proportions of reads and take means
            stdSCexpr = std_row(self.SCexpr)
        if self.hasBK:
            ## standardize to get proportions of reads and take means
            BKmean = std_row(self.BKexpr).mean(axis=0)

        ## initialize A
        if init_A is not None:
            self.init_A = init_A
        elif self.hasSC:    
            self.init_A = np.ones([self.N, self.K], dtype=float)/self.N
            for k in xrange(self.K):
                ## use single cell sample mean if possible
                if len(self.itype[k]) > 0:
                    self.init_A[:, k] = (stdSCexpr[self.itype[k], :]).mean(axis=0)
                ## if not available, then try to use bulk sample means
                elif self.hasBK:
                    self.init_A[:, k] = BKmean
        else:
            self.init_A = np.zeros([self.N, self.K])     
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
            self.init_pkappa = np.array([-1., 0.01], dtype=float)

        if (init_ptau is not None) and (len(init_ptau)==2):
            self.init_ptau = init_ptau
        else:
            self.init_ptau = np.array([self.N, 0.01*self.N], dtype=float)
        self.pkappa = np.copy(self.init_pkappa)
        self.ptau = np.copy(self.init_ptau)


    def init_para_BK(self, init_alpha):
        """Initialize the parameters for bulk model"""
        if (init_alpha is not None) and (len(init_alpha)==self.K):
            self.init_alpha = init_alpha
        else:
            self.init_alpha = np.ones([self.K])
        self.alpha = np.copy(self.init_alpha)


    def init_gibbs(self):
        """Initialize Gibbs sampler"""
        if self.hasSC:
            self.Gibbs_SC = LogitNormalGibbs_SC(A=self.init_A, 
                    pkappa=self.init_pkappa, ptau=self.init_ptau, 
                    SCexpr=self.SCexpr, G=self.G, itype=self.itype)
            self.Gibbs_SC.init_gibbs()

        if self.hasBK:
            self.Gibbs_BK =  LogitNormalGibbs_BK(A=self.init_A, 
                    alpha=self.init_alpha, BKexpr=self.BKexpr,
                    iMarkers=self.iMarkers)
            self.Gibbs_BK.init_gibbs()



