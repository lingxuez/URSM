import numpy as np

#########################
## construct
#########################
data = np.load("simulated.npz")
# myGEM = LogitNormalGEM(
# 				BKexpr=data["BKexpr"], SCexpr=data["SCexpr"], G=data["G"], 
# 				K=3,
# 				burnin=30, sample=70, EM_maxiter=5, EM_CONV=1e-6,
# 				MLE_maxiter=5)


# myGEM.gem()

## 200 bulk tissues, 300 genes
with open("../demo_data/demo_bulk_rnaseq_counts.csv", 'w') as handle:
        np.savetxt(handle, data["BKexpr"][:50, :50], delimiter=',')

## 150 sc cells, 300 genes
with open("../demo_data/demo_single_cell_rnaseq_counts.csv", 'w') as handle:
        np.savetxt(handle, data["SCexpr"][:50, :50], delimiter=',')

with open("../demo_data/demo_single_cell_types.csv", 'w') as handle:
        np.savetxt(handle, data["G"][1:150:3], fmt='%d', delimiter=',')
