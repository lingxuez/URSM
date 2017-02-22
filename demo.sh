
	## single cell RNA-seq file, where rows are cells and columns are genes
	## single cells' types, one line for each cell, an interger between [0, K-1]
	## bulk RNA-seq file, where rows are bulk samples and columns are genes
	## output directory
	## logging file

python -m cProfile -o demo/demo_profile.txt scUnif.py \
	-sc demo/demo_data/demo_single_cell_rnaseq_counts.csv \
	-ctype demo/demo_data/demo_single_cell_types.csv \
	-K 4 \
	-bk demo/demo_data/demo_bulk_rnaseq_counts.csv \
	-outdir demo/demo_out \
	-log demo/demo_logging.log \
	-verbose 2 \
	-burnin 1 \
	-sample 3 \
	-EM_maxiter 2 -MLE_maxiter 100