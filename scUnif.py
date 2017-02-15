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

from gem import * 
import numpy as np
import logging, json, sys, os, argparse, datetime, time

###################################
## file I/O
###################################
def gem2csv(dirname, gem, prefix=""):
    prefix = dirname + "/" + prefix 
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


def load_from_file(filename, dtype=float, delimiter=","):
    if filename is None:
        return None
    else:
        return np.loadtxt(filename, dtype=dtype, delimiter=delimiter)


###############
## read data from files
###############

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-sc", "--single_cell_expr_file", type=str, default=None)
    parser.add_argument("-bk", "--bulk_expr_file", type=str, default=None)
    parser.add_argument("-ctype", "--single_cell_type_file", type=str, default=None)
    parser.add_argument("-K", "--number_of_cell_types", type=int, default=3)
    parser.add_argument("-init_A", "--initial_A_file", type=str, default=None)
    parser.add_argument("-min_A", "--mininimal_A", type=float, default=1e-6)
    parser.add_argument("-init_alpha", "--initial_alpha_file", type=str, default=None)
    parser.add_argument("-est_alpha", "--estimate_alpha", type=bool, default=True)
    parser.add_argument("-pkappa", "--initial_kappa_mean_precision", nargs=2, type=float, 
                            action='store', default=None)
    parser.add_argument("-ptau", "--initial_tau_mean_precision", nargs=2, type=float, 
                            action='store', default=None)
    parser.add_argument("-burnin", "--burn_in_length", type=int, default=200)
    parser.add_argument("-sample", "--gibbs_sample_number", type=int, default=200)
    parser.add_argument("-thin", "--gibbs_thinning", type=int, default=1)
    parser.add_argument("-MLE_CONV", "--Mstep_convergence_tol", type=float, default=1e-6)
    parser.add_argument("-EM_CONV", "--EM_convergence_tol", type=float, default=1e-6)
    parser.add_argument("-MLE_maxiter", "--Mstep_maxiter", type=int, default=100)
    parser.add_argument("-EM_maxiter", "--EM_maxiter", type=int, default=100)
    parser.add_argument("-log", "--logging_file", type=str, default="gem_log.log")
    parser.add_argument("-outdir", "--output_directory", type=str, default="out/")
    parser.add_argument("-outname", "--output_prefix", type=str, default="gemout_")
    parser.add_argument("-verbose", "--verbose_level", type=int, default=2)
    args = parser.parse_args()

    ## verbose level
    if args.verbose_level <= 0:
        level = logging.ERROR
    elif args.verbose_level == 1:
        level = logging.INFO
    elif args.verbose_level >= 2:
        level = logging.DEBUG

    ## set up logging
    logdir = os.path.dirname(args.logging_file)
    if len(logdir) > 0 and not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(level=level, filename= "%s" % args.logging_file, 
                format = '%(message)s',
                filemode= 'w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console)

    ## display general information
    header_info = "#" * 80 + "\n"
    header_info += "Gibbs-EM for %d cell types.\n" % args.number_of_cell_types
    header_info += "Date and time: " + str(datetime.datetime.today()) + "\n"
    header_info += "Algorithm arguments:\n"
    for (argname, argvalue) in vars(args).iteritems():
        header_info += "\t--" + argname + ": " + str(argvalue) + "\n"
    header_info += "#" * 80
    logging.info(header_info)

    ## read data from .csv files
    logging.info("Loading data ...")
    SCexpr = load_from_file(args.single_cell_expr_file)
    BKexpr = load_from_file(args.bulk_expr_file)
    G = load_from_file(args.single_cell_type_file, dtype=int)
    init_A = load_from_file(args.initial_A_file)
    init_alpha = load_from_file(args.initial_alpha_file)
    K = args.number_of_cell_types

    ## check input data
    if SCexpr is None and BKexpr is None:
        logging.error("ERROR: Must provide at least one of single cell or bulk data.")
        sys.exit(1)
    elif SCexpr is not None and BKexpr is not None and SCexpr.shape[1] != BKexpr.shape[1]:
        logging.error("ERROR: Single cell and bulk data must have same number of genes.")
        sys.exit(1)

    if SCexpr is not None:
        if G is None:
            logging.error("ERROR: Must provide cell type information for single cells.")
            sys.exit(1)
        elif SCexpr.shape[0] != G.shape[0]:
            logging.error("ERROR: Mismatched cell dimensions in `%s` and `%s`", 
                args.single_cell_expr_file, args.single_cell_type_file)
            sys.exit(1)
        elif len(set(G) - set(range(K))) > 0:
            logging.error("ERROR: Cell types in `%s` can only take values in {0, ..., K-1}.",
                args.single_cell_type_file)
            sys.exit(1)
        elif len(set(G)) != K:
            logging.error("ERROR: Cell types in `%s` must have K distinct values.",
                args.single_cell_type_file)
            sys.exit(1)

    if BKexpr is not None:
        logging.info("%d bulk samples on %d genes are loaded.", 
                        BKexpr.shape[0], BKexpr.shape[1])
    if SCexpr is not None:
        logging.info("%d single cells on %d genes are loaded.\n", 
                        SCexpr.shape[0], SCexpr.shape[1])
    
    ## perform GEM
    logging.info("Gibbs-EM started ...")
    start = time.time()
    myGEM = LogitNormalGEM(
                  BKexpr=BKexpr, SCexpr=SCexpr, G=G, K=args.number_of_cell_types, 
                  init_A=init_A, min_A=args.mininimal_A,
                  init_alpha=init_alpha, est_alpha=args.estimate_alpha,
                  init_pkappa=args.initial_kappa_mean_precision, 
                  init_ptau=args.initial_tau_mean_precision,
                  burnin=args.burn_in_length, sample=args.gibbs_sample_number, 
                  thin=args.gibbs_thinning, 
                  MLE_CONV=args.Mstep_convergence_tol, MLE_maxiter=args.Mstep_maxiter, 
                  EM_CONV=args.EM_convergence_tol, EM_maxiter=args.EM_maxiter)
    (niter, elbo, converged, path_elbo) = myGEM.gem()
    logging.info("Gibbs-EM finished in %.2f seconds.\n", time.time() - start)

    # save results
    if not os.path.exists(args.output_directory):
      os.makedirs(args.output_directory)
    gem2csv(args.output_directory, myGEM, prefix=args.output_prefix)
    logging.info("Results are under directory %s." % args.output_directory)

    logging.info("Logging info is written to %s." , args.logging_file)
    logging.info("#" * 80 + "\n")



