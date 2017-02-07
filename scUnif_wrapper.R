#############################
## Call python code from R
#############################

## Note: may need to set system path by 
# Sys.setenv(PATH = paste("/usr/local/bin", Sys.getenv("PATH"),sep=":"))


## current directory
script.dir <- dirname(parent.frame(2)$ofile)
  #"/Users/lingxue/Documents/Thesis/SingleCell/EMtools/PyGibbs/"
library("rjson")


PyGEM <- function(BKexpr=NULL, ## sample-by-gene
                  K=3, 
                  SCexpr=NULL, ## sample-by-gene
                  G=NULL, ## cell-type info for dropout model
                  hasBK=TRUE, hasSC=TRUE,
                  ## initialize parameters
                  init_A=NULL,  min_A=1e-6,
                  init_alpha=NULL, est_alpha=TRUE,
                  init_pkappa=NULL, init_ptau=NULL, ## mean and precision 
                  burnin=20, sample=20, thin=1, ## for Gibbs sampling
                  MLE_CONV=1e-3, EM_CONV=1e-3, 
                  MLE_maxiter=1, EM_maxiter=2,
                  logdir=NULL, ## directory for logging file
                  isCleanUp=TRUE ## remove temporary files
                  ) {
  #########################
  ## pass in data
  #########################
  
  ## directory for temporary data
  tmpdir = paste0(script.dir, "/tmp/")
  if (!dir.exists(tmpdir)) {
    dir.create(tmpdir)
  }
  
  ## filename signature
  filesig <- paste0(format(Sys.time(), "%Y-%m-%d_%H-%M-%S"), "_")
  tmpfile_prefix = paste0(tmpdir, filesig)
  
  ## scalar variables
  scalar_vars = list(filesig=filesig, tmpdir=tmpdir,
                 K=K, min_A=min_A, est_alpha=as.numeric(est_alpha),
                 burnin=burnin, sample=sample, thin=thin,
                 MLE_CONV=MLE_CONV, EM_CONV=EM_CONV,
                 MLE_maxiter=MLE_maxiter, EM_maxiter=EM_maxiter,
                 hasBK=as.numeric(hasBK),
                 hasSC=as.numeric(hasSC))
  
  ## write vectors and matrices variables to csv files for python to read
  file_lists <- list() ## file names
  variable_lists <- c("init_A") ## variable names
  if (hasBK) {
    variable_lists <- c(variable_lists, c("BKexpr", "init_alpha"))
  }
  if (hasSC) {
    G = G-1 ## from 1-based to 0-based indexing
    variable_lists <- c(variable_lists, c("SCexpr", "G", "init_pkappa", "init_ptau"))
  }
  
  for (variable in variable_lists) {
    if (!do.call(is.null, list(as.name(variable)))) { ## if not NULL
      variable_file <- paste0(tmpfile_prefix, variable, ".csv") 
      ## write to file
      write.table(eval(as.name(variable)), variable_file, col.names=FALSE, sep=",", row.names=FALSE)
      ## record file name
      file_lists[[paste0(variable, "_file")]] <- variable_file 
    }
  }
  
  ## record scalar_vars and file_lists in JSON file
  json = toJSON(c(scalar_vars, file_lists))
  file_lists$setting_file = paste0(tmpfile_prefix, "setting.txt")
  write.table(json, file_lists$setting_file, 
              sep="\n", row.names=FALSE, col.names=FALSE, 
              quote=FALSE)
  
  #########################
  ## run python
  #########################
  ## directory for logging files
  if (is.null(logdir)) {
    logdir = tmpdir
  }
  
  runlog <- system(paste0("python ", script.dir, "/LogitNormalGEM.py ", file_lists$setting_file, " ", logdir), 
                   intern=TRUE)
  
  #########################
  ## read-in results
  #########################
  ## get results, and also record initial values
  results <- c(list(runlog=runlog, init_pkappa=init_pkappa, init_ptau=init_ptau,
                    init_alpha=init_alpha),
               scalar_vars)
  
  result_variables <- c("est_A", "path_elbo")
  if (hasSC) {
    result_variables <- c(result_variables, "exp_S", "est_pkappa", "est_ptau")
  }
  if (hasBK) {
    result_variables <- c(result_variables, "exp_W", "est_alpha")
  }
  
  for (variable in result_variables) {
    results[[variable]] <- as.matrix(read.csv(paste0(tmpfile_prefix, variable, ".csv"),
                                              header=FALSE, sep=",", stringsAsFactors=FALSE))
  }

  #########################
  ## clean up temporary files
  #########################
  if (isCleanUp) {
    for (file in file_lists) {
      system(paste0("rm ", file))
    }
    for (variable in result_variables) {
      system(paste0("rm ", tmpfile_prefix, variable, ".csv"))
    }
  }

  return(results)
}