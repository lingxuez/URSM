## 2016/07/19 Lingxue Zhu
## A class to perform M-step in LogitNormalGEM
import logging
import numpy as np
import scipy as scp
from scipy.special import psi, gammaln
from utils import * ## helper functions


class LogitNormalMLE(object):

    def __init__(self, 
                BKexpr=None, ## M-by-N, bulk expression
                SCexpr=None, ## L-by-N, single cell expression
                G=None, ## L-by-1, single cell types
                K=3, ## number of cell types
                itype=None, ## cell ids in each type
                hasBK=True, hasSC=True, ## model specification
                init_A = None, ## N-by-K, initial value for A
                init_alpha = None, ## K-by-1, initial value for alpha
                init_pkappa = None, init_ptau = None, ## 2-by-1, mean and var
                min_A=1e-6, ## minimal value of A; must be positive
                min_alpha=1, ## minimum value of alpha
                MLE_CONV=1e-4, MLE_maxiter=100
                ):
        (self.K, self.min_A, self.min_alpha) = (K, min_A, min_alpha)
        (self.MLE_CONV, self.MLE_maxiter) = (MLE_CONV, MLE_maxiter)
        (self.SCexpr, self.G, self.BKexpr) = (SCexpr, G, BKexpr)
        (self.hasBK, self.hasSC, self.itype) = (hasBK, hasSC, itype)

        ## dimensions and other parameters that are iteratively used in MLE
        if self.hasSC: 
            ## SC parameters
            (self.L, self.N) = self.SCexpr.shape
            ## read depth: (L, )
            self.SCrd = self.SCexpr.sum(axis=1) 
        if self.hasBK: ## bulk parameters
            (self.M, self.N) = self.BKexpr.shape

        ## initialize parameters
        self.A = np.copy(init_A)
        self.alpha = np.copy(init_alpha)
        self.pkappa = np.copy(init_pkappa)
        self.ptau = np.copy(init_ptau)


    def update_suff_stats(self, suff_stats): 
        """ 
        Update the new sufficient statistics obtained from E-step
        these suff.stats are used to compute MLEs.
        """
        self.suff_stats = suff_stats

        ## compute the coefficient for 1/A and A
        ## in the gradient w.r.t. A. 
        ## These terms don't change across iterations.
        self.gd_coeffAinv = np.zeros(self.A.shape)
        self.gd_coeffA = np.zeros(self.A.shape)
        self.gd_coeffConst = np.zeros(self.A.shape)

        ## compute part of elbo that doesn't depend on mle parameters.
        ## this helps avoid non-necessary computations when evaluating elbo.
        self.elbo_const = 0

        if self.hasBK:
            self.elbo_const += (np.transpose(suff_stats["exp_Zjk"]) * \
                                suff_stats["exp_logW"]).sum()
            ## E[sum_j Z_ijk], N x K
            self.gd_coeffAinv += suff_stats["exp_Zik"]
            self.suff_stats["exp_mean_logW"] = suff_stats["exp_logW"].mean(axis=1)
            

        if self.hasSC:
            self.elbo_const += suff_stats["sc_exp_elbo_const"]
            ## the auxilliary parameter u
            self.opt_u()
            ## E[sum_l (- tau_l^2 * w_il)] where sum is within cell type
            self.gd_coeffA = suff_stats["coeffAsq"] * 2.0
            
            ## E[sum_l S_il * Y_il], where sum is within cell type
            for k in xrange(self.K):
                itype = self.itype[k]
                if len(itype) > 0:
                    self.update_gd_coeffConst(k)
                    self.gd_coeffAinv[:, k] += (self.SCexpr[itype, :] * \
                            suff_stats["exp_S"][itype, :]).sum(axis=0)           


    def opt_kappa_tau(self):
        """
        Optimize Gaussian mean and precision (i.e., 1/variance) for kappa and tau.
        This has closed form solution.
         """
        kappa_mean = np.mean(self.suff_stats["exp_kappa"])
        tau_mean = np.mean(self.suff_stats["exp_tau"])
        kappa_var = np.mean(self.suff_stats["exp_kappasq"]) - kappa_mean**2
        tau_var = np.mean(self.suff_stats["exp_tausq"]) - tau_mean**2

        self.pkappa = np.array([kappa_mean, kappa_var])
        self.ptau = np.array([tau_mean, tau_var])

        # logging.debug("\t\toptimized kappa_tau, elbo=%.6f", self.compute_elbo())


    def opt_u(self):
        """
        Update the optimized auxilliary u as well as
        the constant in gradient of A that depends on u
        """
        ## update auxilliary u
        self.u = (np.transpose(self.A[:, self.G]) *\
                        self.suff_stats["exp_S"]).sum(axis=1)


    def update_gd_coeffConst(self, k):
        ## update the constant coefficient in the gradient of A
        ## sum_l E[tau_l*(S_il-0.5) - kappa_l*tau_l*w_il],
        ## where sum is within cell type
        itype = self.itype[k]
        self.gd_coeffConst[:, k] = self.suff_stats["coeffA"][:, k] - \
                    ((self.suff_stats["exp_S"][itype,:] * \
                    (self.SCrd/self.u)[itype, np.newaxis]).sum(axis=0))


    def opt_A_u(self):
        """
        Optimize the profile matrix A and auxiliary u
        """
        ## initial stepsize in gradient descent
        if self.hasBK and self.hasSC:
            init_step = 0.01 / (self.L + self.M)
        elif self.hasBK:
            init_step = 0.01 / self.M
        else:
            init_step = 0.01 / self.L

        ## Optimize column-by-column
        for k in xrange(self.K):
            ## with single cell part, need coordinate descent
            if self.hasSC and len(self.itype[k]) > 0:
                niter = self.opt_Ak(k, init_step)
                logging.debug("\t\tOptimized A%d in %d iterations", k, niter)
                # logging.debug("\t\t\telbo=%f", self.compute_elbo())

            ## with only bulk data, we have closed form for Ak: proportional to Zik
            else:
                self.A[:, k] = self.suff_stats["exp_Zik"][:, k] / \
                                float(np.sum(self.suff_stats["exp_Zik"][:, k]))
                self.A[:, k] = simplex_proj(self.A[:, k], self.min_A)
                logging.debug("\t\tOptimized A%d with closed form solution", k)
                # logging.debug("\t\t\telbo=%f", self.compute_elbo())


    def opt_Ak(self, k, init_step):
        """
        Optimize the k-th column of A using projected gradient descent
        with backtracking
        """
        old_Ak = np.copy(self.A[:, k])
        (converged, niter) = (self.MLE_CONV+1, 0)
        while (converged > self.MLE_CONV) and (niter < self.MLE_maxiter): 
            ## projected gradient descent with backtracking for optimize A[:, k]
            (self.A[:, k], obj_new, stepsize) = self.backtracking(
                        old_val=old_Ak, 
                        grad_func=lambda Ak: -self.get_grad_A(Ak, k), 
                        obj_func=lambda Ak: -self.get_obj_A(Ak, k), 
                        proj_func=lambda Ak: simplex_proj(Ak, self.min_A),
                        init_step=init_step) 
            ## optimize auxiliary u
            self.opt_u()
            ## update coefficients based on new u
            self.update_gd_coeffConst(k) 

            ## convergence
            niter += 1
            converged = np.linalg.norm(self.A[:, k] - old_Ak, 1)
            old_Ak = np.copy(self.A[:, k])

        return niter


    def opt_alpha(self):
        """Optimize alpha using gradient descent"""
        (converged, niter) = (self.MLE_CONV+1, 0)
        while (converged > self.MLE_CONV) and (niter < self.MLE_maxiter):
            old_alpha = np.copy(self.alpha)
            ## pgradient descent
            # print self.get_grad_alpha(self.alpha)
            self.alpha = self.backtracking(old_val=old_alpha, 
                    grad_func=lambda alpha: (-self.get_grad_alpha(alpha)), 
                    obj_func=lambda alpha: (-self.get_obj_alpha(alpha)),
                    init_step=10)[0]
            ## constraint: alpha>=1
            self.alpha = np.maximum(self.min_alpha, self.alpha)

            ## update
            niter += 1
            converged = np.linalg.norm(self.alpha - old_alpha)

        logging.debug("\t\tOptimized alpha in %s iterations", niter)
        logging.debug("\t\t\telbo=%.6f", self.compute_elbo())

        return niter


    def backtracking(self, old_val, grad_func, obj_func, init_step=0.1,
        proj_func=None):
        """Backtracking line search for (projected) gradient descent."""
        grad_old = grad_func(old_val)
        obj_old = obj_func(old_val)

        stepsize = init_step
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
        """
        Get the objective function value at given alpha_val.
        This has been scaled by 1/M
        """
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
        We consider the average log-likelihood as objective,
        i.e., f/L for single cell, f/M for bulk, and f/(M*L) for complete model 
        """
        grad_A = self.gd_coeffAinv[:, k]/Ak + self.gd_coeffA[:, k]*Ak + self.gd_coeffConst[:, k]
        ## scale the gradient s.t. it's roughly same order with A
        grad_A /= self.N

        return grad_A


    def get_obj_A(self, Ak, k):
        """
        Get the objective function value at given A_val for k-th column of A,
        scaled by 1/N.
        """
        ## use the coefficients for gradient to compute objective
        obj_A = (self.gd_coeffAinv[:, k] * np.log(Ak)).sum()
        if self.hasSC:
            obj_A += (self.gd_coeffA[:, k] * np.square(Ak)).sum()/2.0
            obj_A += (self.gd_coeffConst[:, k] * Ak).sum()
        obj_A /= self.N

        return obj_A


    def compute_elbo(self):
        """
        Compute Evidence Lower Bound, scaled by 1/N
        """
        ## constant terms that do not depend on mle parameters
        elbo = self.elbo_const
        ## terms that only involve log(A)
        elbo += (self.gd_coeffAinv * np.log(self.A)).sum()

        # terms involving alpha, W, Z
        if self.hasBK:
            elbo += self.get_obj_alpha(self.alpha) * self.M

        if self.hasSC: 
            ## terms for A and A^2
            elbo += (self.gd_coeffA * np.square(self.A)).sum()/2.0
            elbo += (self.gd_coeffConst * self.A).sum() 
            ## an extra term for u
            elbo -= np.sum(self.SCrd * np.log(self.u))
            ## other terms involving pkappa, ptau
            elbo -= np.log(self.pkappa[1]*self.ptau[1]) * self.L / 2.0
            elbo -= (np.sum(self.suff_stats["exp_kappasq"]) - \
                    2 * self.pkappa[0] * np.sum(self.suff_stats["exp_kappa"]) + \
                    self.L * (self.pkappa[0]**2)) / (2.0 * self.pkappa[1])
            elbo -= (np.sum(self.suff_stats["exp_tausq"]) - \
                    2 * self.ptau[0] * np.sum(self.suff_stats["exp_tau"]) + \
                    self.L * (self.ptau[0]**2))  / (2.0 * self.ptau[1]) 

        elbo /= self.N

        return elbo 



