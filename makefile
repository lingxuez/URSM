##################
# Author:
# Lingxue Zhu
# lzhu@cmu.edu
# -no_est_alpha \
##################
.PHONY: demo, clean

simulate:
	cd demo; python demo_simulate_data.py

demo-run:
	python scUnif.py \
	-sc demo/demo_data/demo_single_cell_rnaseq_counts.csv \
	-ctype demo/demo_data/demo_single_cell_types.csv \
	-K 5 \
	-bk demo/demo_data/demo_bulk_rnaseq_counts.csv \
	-iMarkers demo/demo_data/demo_iMarkers.csv \
	-init_A demo/demo_data/demo_init_A.csv \
	-no_est_alpha \
	-outdir demo/demo_out \
	-log demo/demo_out/demo_logging.log \
	-verbose 2 \
	-burnin 50 -sample 50 \
	-burnin_bk 50 -sample_bk 10 \
	-EM_maxiter 50 -MLE_maxiter 100

demo-plot:
	cd demo; python demo_plots.py

demo: 
	make demo-run
	make demo-plot

clean:
	rm *.pyc
