# URSM
A Unified RNA Sequencing Model (URSM) for joint analysis of single cell and bulk RNA-seq data.
This is a python implementation for
> Zhu, Lei, Devlin and Roeder (2017), "A Unified Statistical Framework for Single Cell and Bulk RNA Sequencing Data", Annals of Applied Statistics (to appear). 
[[pdf](https://arxiv.org/pdf/1609.08028.pdf)]

Pease cite our paper in your publication if it helps your research:
```
@article{zhu2016a,
    title={A Unified Statistical Framework for Single Cell and Bulk RNA Sequencing Data},
    author={Zhu, Lingxue and Lei, Jing and Devlin, Bernie and Roeder, Kathryn},
    journal={Annals of Applied Statistics},
    year={2017}
}
```

The code was developed under `Mac OS X` using `Python 2.7`. We hope this helps your research, but please note that this release has not been fully tested nor optimized at this point.

## Set Up
Dependencies:
* [Python 2.7](https://www.python.org/downloads/release/python-2713/)
* [pypolyagamma (1.1.1)](https://github.com/slinderman/pypolyagamma)

It can be tricky to install `pypolyagamma`. I find it easier if you install [Cython](http://cython.org/) first, and *after then*, try 
```
pip install pypolyagamma
```
If this still does not work for you, you may want to try the following command 
(you will need to change these paths to the location of `gcc` on your machine):
```
export CC=/usr/local/bin/gcc-5 CXX=/usr/local/bin/g++-5
export DYLD_LIBRARY_PATH=/usr/local/Cellar/gcc/5.3.0/lib/gcc/5/
pip install pypolyagamma
```

In any case, please make sure that you can `import pypolyagamma` in Python without error before you proceed.


## Usage
### Demo
We provide a simple demo with 3 cell types, 50 single cells, 50 bulk samples, and 100 genes. Please run
```
make demo
```
The input simulated data is stored under directory `demo/demo_data/`, 
and the output results are stored under directory `demo/demo_out/`. 
This also generates two plots: `demo/estimation_A.png` and `demo/estimation_W.png`, 
containing the estimated versus true profile matrix `A` and mixing proportions in bulk samples `W`, respectively.

### Python script

To run the python script:
```
python scUnif.py \
	-K 3 \
	-sc path/to/single_cell_rnaseq.csv \
	-ctype path/to/single_cell_types.csv \
	-bk path/to/bulk_rnaseq.csv \
	-outdir path/to/output/directory \
	-log path/to/logging_file.log \
	-burnin 50 \
	-sample 50 \
	-EM_maxiter 50
```

**Basic Options**:
* `K`: an integer, number of cell types.
* `-sc`: path to the single cell RNA-seq `.csv` data, where each row is one cell and each column is one gene, without column or row names.
See `demo/demo_data/demo_single_cell_rnaseq_counts.csv` for an example input. 
If no single cell data is available, then leave out this option.
* `-ctype`: path to the single cell type file, where each row contains a number among `{0, 1, ..., K-1}`, indicating the cell type. 
Please note that cell types are 0-indexed. 
No column or row names should be provided; the cell types should be consistent with the rows in the single cell RNA-seq file. 
If no single cell data is available, then leave out this option.
* `-bk`: path to the bulk RNA-seq `.csv` data, where each row is one bulk sample and each column is one gene, without column or row names.
See `demo/demo_data/demo_bulk_rnaseq_counts.csv` for an example input.
If no bulk data is available, then leave out this option.
* `-outdir`: path to the output directory where the output `.csv` files will be saved. 
* `-log`: path to the logging file. See `demo/demo_out/demo_logging.log` for an example of the logging file.
* `-burnin`: an integer, number of burnin period to use. Typically 50-100 will be enough.
* `-sample`: an integer, number of Gibbs samples to use in each EM iteration. Typically 100-200 will be enough.
* `-EM_maxiter`: the number of maximal EM iterations. Typically 50-100 will be enough.


**Advanced Options**:
more to come.

## Output
The output will be saved under the specified directory by option `-outdir`. See `demo/demo_out/` for an example. This directory will always contain:
* `gemout_est_A.csv`: the estimated profile matrix `A`, where each row is one gene and each column is one cell type.
* `gemout_path_elbo.csv`: the ELBO after each EM iteration. Please note that ELBO sometimes can decrease because of the approximate inference.

If single cell data is provided, the directory will include the following files:
* `gemout_exp_S.csv`: the posterior probability of observation in single cell data. Note that dropout porbability is `1-S`. Each row is one cell and each column is one gene, consistent with the input single cell data.
* `gemout_est_pkappa.csv`: the estimated Normal mean and variance for `kappa_l`.
* `gemout_est_kappa.csv`: the posterior mean of `kappa_l` for each cell.
* `gemout_est_ptau.csv`: the estimated Normal mean and variance for `tau_l`.
* `gemout_est_tau.csv`: the posterior mean of `tau_l` for each cell.

If bulk data is provided, the directory will include the following files:
* `gemout_exp_W.csv`: the posterior expectation of the mixing proportions in bulk samples, where each column is one sample and each row is one cell type.
* `gemout_est_alpha.csv`: the estimated hyperparameter for mixing proportions in bulk samples.




## R wrapper
Usually we recommend to use the python script directly. 
But if you really prefer to work in R, we provided a basic R wrapper `scUnif_wrapper.R`. This wrapper simply calls the python script and passes along the parameters from R. The output will be stored in files, under your specified directory. Please see the previous section for details of the output results. 

Specifically, in R, run

```{r}
source("scUnif_wrapper.R") ## change this to the path to this file on your machine

## The function PyGEM() will call the python script from R
arguments <- PyGEM(

  py_script="scUnif.py", ## change this to the path to scUnif.py on your machine 
  data_dir = "tmp/",  ## a temporary directory to hold intermediate data
  out_dir = "out/", ## the output directory where the results will be stored 
  log_dir = "log/", ## the directory where the logging file will be stored 
  output_prefix = "out_", ## the prefix that will be added to the output files

  BKexpr=NULL, ## a matrix of bulk RNA-seq counts; 
               ## rows are samples and columns are genes; 
               ## NULL if no bulk data is available
  SCexpr=NULL, ## a matrix of single cell RNA-seq counts; 
               ## rows are samples and columns are genes; 
               ## NULL if no single cell data is available
  K=2, ## number of cell types 
  G=NULL, ## a vector of cell types, 1-indexed (note: different from the python script). 
          ## e.g., c(1,1,2,2,2) for 5 cells from 2 types.
          ## NULL if no single cell data is available

  burnin=50, ## number of burnin period to use
  sample=50, ## number of Gibbs samples to use in each EM iteration
  EM_maxiter=50 ## number of maximal EM iterations
)

## The returned “arguments” is a list containing all parameters used by the algorithm, 
## including the following two:
print(paste("Logging has been saved to", arguments$log))
print(paste("Results have been saved under directory", arguments$outdir))
```



