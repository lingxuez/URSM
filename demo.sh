
	## single cell RNA-seq file, where rows are cells and columns are genes
	## single cells' types, one line for each cell, an interger between [0, K-1]
	## bulk RNA-seq file, where rows are bulk samples and columns are genes
	## output directory
	## logging file

python -m cProfile -o demo_profile.txt scUnif.py \
	-sc demo_data/demo_single_cell_rnaseq_counts.csv \
	-ctype demo_data/demo_single_cell_types.csv \
	-bk demo_data/demo_bulk_rnaseq_counts.csv \
	-outdir demo_out \
	-log demo_out/demo_logging.log \
	-verbose 2 \
	-burnin 10 \
	-sample 20 \
	-EM_maxiter 5 \