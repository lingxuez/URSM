##################
# Author:
# Lingxue Zhu
# lzhu@cmu.edu
##################
.PHONY: demo, clean

demo-simulate:
	cd demo; python demo_simulate_data.py

demo-run:
	python scUnif.py \
	-sc demo/demo_data/demo_single_cell_rnaseq_counts.csv \
	-ctype demo/demo_data/demo_single_cell_types.csv \
	-K 3 \
	-bk demo/demo_data/demo_bulk_rnaseq_counts.csv \
	-outdir demo/demo_out \
	-log demo/demo_out/demo_logging.log \
	-verbose 1 \
	-burnin 100 \
	-sample 200 \
	-EM_maxiter 5

demo-plot:
	cd demo; python demo_plots.py

demo: 
	make demo-run
	make demo-plot

clean:
	rm *.pyc
