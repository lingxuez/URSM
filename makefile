#####################################################
# Author:
# Lingxue Zhu
# lzhu@cmu.edu
#
# To see a demo, run
# $ make demo
#
# The input data are stored under demo/demo_data
# and the output files are stored under demo/demo_out
#
# To re-simulate the data, run
# $ make simulate
#####################################################
.PHONY: run, plot, demo, clean

demo: run plot clean

simulate:
	@cd demo; python demo_simulate_data.py

run:
	python scUnif.py \
	-K 3 \
	-sc demo/demo_data/demo_single_cell_rnaseq_counts.csv \
	-ctype demo/demo_data/demo_single_cell_types.csv \
	-bk demo/demo_data/demo_bulk_rnaseq_counts.csv \
	-outdir demo/demo_out \
	-log demo/demo_out/demo_logging.log \
	-burnin 10 -sample 20 \
	-EM_maxiter 10


plot:
	@cd demo; python demo_plots.py


clean:
	@rm *.pyc
