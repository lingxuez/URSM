##################
# Author:
# Lingxue Zhu
# lzhu@cmu.edu
# -no_est_alpha \
# -sc demo/demo_data/demo_single_cell_rnaseq_counts.csv \
# -ctype demo/demo_data/demo_single_cell_types.csv \
##################
.PHONY: demo, clean

simulate:
	cd demo; python demo_simulate_data.py

demo-run:
	python scUnif.py \
	-K 5 \
	-bk demo/demo_data/demo_bulk_rnaseq_counts.csv \
	-iMarkers demo/demo_data/demo_iMarkers.csv \
	-init_A demo/demo_data/demo_init_A.csv \
	-no_est_alpha \
	-outdir demo/demo_out \
	-log demo/demo_out/demo_logging.log \
	-verbose 1 \
	-burnin 50 -sample 50 \
	-burnin_bk 1 -sample_bk 1 \
	-EM_maxiter 50 -MLE_maxiter 500


test:
	python scUnif.py \
	-K 5 \
	-bk test_Data/out_smalpha_rerun_mk_em10_bkburn100_bk_200genes_100sc_150bk_Ksc3_Kbk5_tau300_kappa-1_bulk_expr_file.csv \
	-iMarkers test_Data/out_smalpha_rerun_mk_em10_bkburn100_bk_200genes_100sc_150bk_Ksc3_Kbk5_tau300_kappa-1_iMarkers_file.csv \
	-init_A test_Data/out_smalpha_rerun_mk_em10_bkburn100_bk_200genes_100sc_150bk_Ksc3_Kbk5_tau300_kappa-1_initial_A_file.csv \
	-no_est_alpha \
	-outdir test_out \
	-log test_out/demo_logging.log \
	-verbose 2 \
	-burnin 50 -sample 50 \
	-burnin_bk 50 -sample_bk 1 \
	-min_A 1e-10 \
	-EM_maxiter 10 -MLE_maxiter 500


demo-plot:
	cd demo; python demo_plots.py

demo: 
	make demo-run
	make demo-plot

clean:
	rm *.pyc
