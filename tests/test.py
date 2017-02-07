
from LogitNormalGEM import *

#########################
## construct
#########################
data = np.load("simulated.npz")
myGEM = LogitNormalGEM(
				BKexpr=data["BKexpr"], SCexpr=data["SCexpr"], G=data["G"], 
				K=3,
				burnin=30, sample=70, EM_maxiter=5, EM_CONV=1e-6,
				MLE_maxiter=5)


myGEM.gem()

