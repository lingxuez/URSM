## test PyGibbs
rm(list=ls())
source("~/Documents/Thesis/SingleCell/EMtools/simulation/simulate_A.R")
source("~/Documents/Thesis/SingleCell/EMtools/simulation/simulate_lognormal.R")

N = 100
K =3
L = 50
M = 50
G = c(rep(1, 10), rep(2, 10), rep(3, 30))
SC_depths = rep(N*5, L)
BK_depths = rep(N*10, M)

alpha = c(1, 2, 4)

sim.A <- simulate_A(N, anchor.size=5, diff.size=10, mean.scale=1, distr="unif")
A <- sim.A$A_std
colSums(A)

sim.SC <- multdrop_log_normal(N, L, K, G, A, 
                              tau=500, kappa=-0.5, tau_sd=5, kappa_sd=0.05,
                              SC_depths)
SCexpr = t(sim.SC$expr)

sim.BK <- multmix_simulate(N, M, K, alpha, A, BK_depths)
BKexpr <- t(sim.BK$expr)

source("/Users/lingxue/Documents/Thesis/SingleCell/EMtools/PyGibbs/PyGibbs.R")
res <- PyGEM(BKexpr=BKexpr, K=K, SCexpr=SCexpr, G=G,
             min_A = 1e-3/N,
             init_pkappa=c(-0.5, 1/25), init_ptau=c(500, 1/0.05^2),
             burnin=1, sample=2, thin=1, 
             MLE_maxiter=1, EM_maxiter=1)

save(res, sim.A, sim.SC, sim.BK, file="test_pyGibbs.RData")

plot(as.matrix(res$est_A), as.matrix(sim.A$A_std))
abline(0, 1)

plot(t(as.matrix(res$exp_W)), as.matrix(sim.BK$W))
abline(0, 1)

idropout <- which(SCexpr == 0 & t(sim.SC$S) == 0)
istruct0 <- which(SCexpr == 0 & t(sim.SC$S) == 1)
boxplot(list(idropout = as.matrix(res$exp_S)[idropout],
             istruct0 = as.matrix(res$exp_S)[istruct0]))

res$est_ptau
res$est_pkappa


