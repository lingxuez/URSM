#############################
## Call python code from R
#############################

## Note: may need to set system path by 
# Sys.setenv(PATH = paste("/usr/local/bin", Sys.getenv("PATH"),sep=":"))


# ## current directory
# script.dir <- dirname(parent.frame(2)$ofile)
##"/Users/lingxue/Documents/Thesis/SingleCell/scUnif/"

data_to_csv <- function(data, data_dir, file_prefix, varname) {
  res = list()
  if (!is.null(data)){
    filename = paste0(data_dir, "/", file_prefix, "_", varname, ".csv")
    write.table(data, file=filename, 
                quote=FALSE, sep=",", row.names=FALSE, col.names=FALSE)
    res[varname] = filename
  }
  return(res)
}

PyGEM <- function(py_script="/Users/lingxue/Documents/Thesis/SingleCell/scUnif/scUnif.py",
                  BKexpr=NULL, ## sample-by-gene
                  K=3, 
                  SCexpr=NULL, ## sample-by-gene
                  G=NULL, ## cell-type info for single cell
                  data_dir = "data/", ## directory to hold data files
                  data_prefix = "", ## prefix added to data files
                  
                  ## model parameters
                  init_A=NULL,  min_A=1e-6,
                  init_alpha=NULL, est_alpha=TRUE,
                  init_pkappa=NULL, init_ptau=NULL, ## mean and precision 
                  burnin=20, sample=20, thin=1, ## for Gibbs sampling
                  MLE_CONV=1e-3, EM_CONV=1e-3, 
                  MLE_maxiter=1, EM_maxiter=2,
                  verbose=1,
                  out_dir="out/", ## output directory
                  output_prefix="out_", ## output prefix
                  log_dir="log/" ## logging directory
                  ) {
  
  if (!dir.exists(data_dir)) {
    dir.create(data_dir, recursive=TRUE)
  }
  if (!dir.exists(log_dir)) {
    dir.create(log_dir, recursive=TRUE)
  }
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive=TRUE)
  }
  
  #########################
  ## write expression data to file
  ## and record file names
  #########################
  arguments = list()
  
  arguments = c(arguments,
                data_to_csv(BKexpr, data_dir, data_prefix, "bulk_expr_file"),
                data_to_csv(SCexpr, data_dir, data_prefix, "single_cell_expr_file"),
                ## note that cell type in python is 0-indexed
                data_to_csv(G-1, data_dir, data_prefix, "single_cell_type_file"),
                data_to_csv(init_A, data_dir, data_prefix, "initial_A_file"),
                data_to_csv(init_alpha, data_dir, data_prefix, "initial_alpha_file"))
  
  ################################
  ## other algorithm parameters
  #################################
  arguments = c(arguments,
              list(output_directory=out_dir, 
                   output_prefix=output_prefix, 
                   logging_file=paste0(log_dir, "/", output_prefix, ".log"),
                   EM_maxiter=EM_maxiter, Mstep_maxiter=MLE_maxiter,
                   EM_convergence_tol=EM_CONV, Mstep_convergence_tol=MLE_CONV,
                   gibbs_thinning=thin, gibbs_sample_number=sample, burn_in_length=burnin,
                   number_of_cell_types=K, 
                   mininimal_A=min_A,
                   verbose_level=verbose)
  )
  
  ## several parameters need special handling to have the right format
  arguments$estimate_alpha = c("False", "True")[est_alpha+1]
  if (!is.null(init_pkappa)) {
    arguments$initial_kappa_mean_precision=paste(init_pkappa, collapse = " ")
  }
  if (!is.null(init_ptau)) {
    arguments$initial_tau_mean_precision=paste(init_ptau, collapse = " ")
  }
  
  
  #########################
  ## run python
  #########################
  ## command line args
  cmdargs = ""
  for (i in 1:length(arguments)) {
    cmdargs = paste0(cmdargs, " --", names(arguments)[i], " ", arguments[i])
  }
  
  ## run
  arguments$R_runlog <- system(paste0("python ", py_script, cmdargs), intern=TRUE)
  return (arguments)
}