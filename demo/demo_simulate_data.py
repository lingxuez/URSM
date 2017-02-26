## 
## Simulate data for demo purpose.
##

from __future__ import with_statement
import numpy as np
import logging, sys
import scipy as scp
from scipy import special

def simulate_A(N, K, anchor_size=2, anti_size=2):
    """
    Simulate the profile matrix.

    Args:
        N: integer number of genes.
        K: integer number of cell types.
        anchor_size: integer number of anchor genes per cell type.
        anti_size: integer number of anti-anchor genes per cell type.
            Note that 
            (anchor_size + anti_size) * K <= N.

    Return:
        An N-by-K profile matrix which columns sum to 1.
    """
    if anchor_size*K + anti_size*K > N:
        logging.error("Invalid arguments: (anchor_size + anti_size) * K > N.")
        sys.exit(1)

    ## log-normal entries
    A = np.exp(np.random.normal(size=(N, K)))
    ## set anchor genes
    for k in range(K):
        for l in np.arange(K)[np.arange(K)!=k]:
            A[np.arange(anchor_size*k, anchor_size*(k+1)), l] = 0
    ## set anti-anchor genes
    for k in range(K):
        A[np.arange(anchor_size*K + anti_size*k, anchor_size*K + anchor_size*(k+1)), k] = 0
    ## normalize to sum to 1
    A = np.divide(A, A.sum(axis=0), out=A)

    return A


def simulate_bulk(N, M, K, alpha, A, depths):
    """
    Simulate bulk expression.

    Args:
        N: integer number of genes
        M: integer number of bulk samples
        K: integer number of cell types
        alpha: numpy vector of length K, hyper parameter for mixing proportions
        A: N-by-K profile matrix
        depths: numpy vector of length M, sequencing depths for bulk samples

    Returns:
        simulated bulk expression data (M-by-N) and mixing proportions (M-by-K).
    """

    ## mixing proportion M-by-K
    W = np.random.dirichlet(alpha, M)
    ## expected expression level M-by-N
    mix_profile = np.dot(W, A.transpose())
    ## expression level
    expr = np.empty((M, N))
    for m in range(M):
        expr[m, ] = np.random.multinomial(depths[m], mix_profile[m, ])
    
    return (expr, W)

def simulate_sc(N, L, K, G, A, depths, tau, kappa, tau_sd, kappa_sd):
    """
    Simulate single cell expression.

    Args:
        N: integer number of genes
        L: integer number of cells
        K: integer number of cell types
        G: numpy vector of length L indicating cell types, 
            each value is one of {0, ..., K-1}
        A: N-by-K profile matrix
        depths: numpy vector of length L, sequencing depths for single cells
        tau: mean slope of dropout curve
        kappa: mean intercept of dropout curve
        tau_sd: sd of tau
        kappa_sd: sd of kappa

    Returns:
        simulated single cell expression data (L-by-N).
    """

    ## observation probability L-by-N
    kappas = np.random.normal(loc=kappa, scale=kappa_sd, size=(1,L))
    taus = np.random.normal(loc=tau, scale=tau_sd, size=(1,L))
    obs_prob = (special.expit(A[:, G] * taus + kappas)).transpose()
    
    ## dropout status
    S = np.empty((L, N), dtype=int)
    for l in range(L):
        for n in range(N):
            S[l, n] = np.random.binomial(1, obs_prob[l, n])

    ## expression
    expr = np.empty((L, N))
    for l in range(L):
        ## dropout
        probs = np.multiply(A[:, G[l]], S[l, :])
        ## re-normalize to sum to 1
        probs = np.divide(probs, np.sum(probs), out=probs)
        ## expression
        expr[l, :] = np.random.multinomial(depths[l], probs)

    return (expr, S)


###############
## simulate data
##############
if __name__ == "__main__":
    N = 100
    sc_K = 3
    bk_K = 5
    L = 20
    M = 50
    alpha = np.arange(1, bk_K+1)
    G = np.array([0]*5 + [1]*5 + [2]*(L-10))
    tau = 1.5*N
    tau_sd = tau*0.1
    kappa = -1
    kappa_sd = 0.5
    K = max(sc_K, bk_K)

    anchor_size=3
    anti_size=2
    A = simulate_A(N, K, anchor_size, anti_size)

    bk_depths = [100*N] * M
    (bk_expr, bk_W) = simulate_bulk(N, M, bk_K, alpha, A[:, :bk_K], bk_depths)


    sc_depths = [10*N] * L
    (sc_expr, sc_S) = simulate_sc(N, L, sc_K, G, A[:, :sc_K], 
                                sc_depths, tau, kappa, tau_sd, kappa_sd)

    ## save results files
    data_dir = "demo_data/"
    np.savetxt(data_dir + "demo_bulk_rnaseq_counts.csv", bk_expr, delimiter=",")
    np.savetxt(data_dir + "demo_bulk_mix_proportions.csv", bk_W, delimiter=",")
    np.savetxt(data_dir + "demo_single_cell_rnaseq_counts.csv", sc_expr, delimiter=",")
    np.savetxt(data_dir + "demo_single_cell_dropout_status.csv", sc_S, fmt="%d", delimiter=",")
    np.savetxt(data_dir + "demo_single_cell_types.csv", G, fmt="%d", delimiter=",")
    np.savetxt(data_dir + "demo_profile_matrix.csv", A, delimiter=",")

    ## marker genes indices
    iMarkers = np.zeros((K*anchor_size, 2), dtype=int)
    for k in range(K):
        ianchor = range(k*anchor_size, (k+1)*anchor_size)
        iMarkers[ianchor, 0] = ianchor
        iMarkers[ianchor, 1] = k
    np.savetxt(data_dir + "demo_iMarkers.csv", iMarkers, fmt="%d", delimiter=",")

    ## initialize A with marker information
    ## standardize to get proportions of reads and take means
    std_sc_expr = sc_expr / sc_expr.sum(axis=1)[:, np.newaxis]
    std_bk_expr = bk_expr / bk_expr.sum(axis=1)[:, np.newaxis]

    init_A = np.ones((N, K), dtype=float) / N
    for k in range(K):
        ## use single cell sample mean if possible
        itype = np.where(G == k)[0]
        if len(itype) > 0:
            init_A[:, k] = (std_sc_expr[itype, :]).mean(axis=0)

    ## set marker genes
    for k in range(K):
        ianchor = range(k*anchor_size, (k+1)*anchor_size)
        for l in range(K):
            if l != k:
                init_A[ianchor, l] = 0
    ## re-normalize
    init_A = init_A / init_A.sum(axis=0)[np.newaxis, :]
    # print init_A

    np.savetxt(data_dir + "demo_init_A.csv", init_A, delimiter=",")









