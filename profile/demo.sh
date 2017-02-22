
## run GEM
cd ../
python -m cProfile -o demo/demo_profile.txt scUnif.py \
	-sc demo/demo_data/demo_single_cell_rnaseq_counts.csv \
	-ctype demo/demo_data/demo_single_cell_types.csv \
	-K 3 \
	-bk demo/demo_data/demo_bulk_rnaseq_counts.csv \
	-outdir demo/demo_out \
	-log demo/demo_logging.log \
	-verbose 2 \
	-burnin 50 \
	-sample 50 \
	-EM_maxiter 20 -MLE_maxiter 100

## make plots
cd demo
python demo_results.py