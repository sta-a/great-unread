
# fix imports in run_embeddings.py
# Create embeddings
conda activate s2v && cd /cluster/scratch/stahla/src_eng/ && sbatch --ntasks=50 --time=23:00:00 --output output-eng-emb.txt --error error-eng-emb.txt --wrap="python /cluster/scratch/stahla/src_eng/run_embeddings.py eng" && cd $SCRATCH

conda activate s2v && cd /cluster/scratch/stahla/src_ger/ && sbatch --ntasks=50 --time=23:00:00 --output output-ger-emb.txt --error error-ger-emb.txt --wrap="python /cluster/scratch/stahla/src_ger/run_embeddings.py ger" && cd $SCRATCH


conda activate s2v && cd /cluster/scratch/stahla/src_eng_params/ && sbatch --ntasks=50 --time=23:00:00 --output output-eng-emb-params.txt --error error-eng-emb-params.txt --wrap="python /cluster/scratch/stahla/src_eng_params/run_embeddings.py eng" && cd $SCRATCH
conda activate s2v && cd /cluster/scratch/stahla/src_ger_params/ && sbatch --ntasks=50 --time=23:00:00 --output output-ger-emb-params.txt --error error-ger-emb-params.txt --wrap="python /cluster/scratch/stahla/src_ger_params/run_embeddings.py ger" && cd $SCRATCH



# Run matrix combinations
conda activate nlpplus && sbatch --mem-per-cpu=100000 --time=23:55:00 --output output_file-eng-eval.txt --error error_file-eng-eval.txt --wrap="python /cluster/scratch/stahla/src_eval/run_embeddings.py eng"
conda activate nlpplus && sbatch --mem-per-cpu=100000 --time=23:55:00 --output output_file-ger-eval.txt --error error_file-ger-eval.txt --wrap="python /cluster/scratch/stahla/src_eval/run_embeddings.py ger"




###############################################################
# run parameter evaluation images

conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:00:00 --output output-eng-pimgs.txt --error error-eng-pimgs.txt --wrap="python /cluster/scratch/stahla/src_paramimgs/run_embeddings.py eng"
conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:00:00 --output output-ger-pimgs.txt --error error-ger-pimgs.txt --wrap="python /cluster/scratch/stahla/src_paramimgs/run_embeddings.py ger"


# create single images for run parameter
conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:55:00 --output output-eng-rimgs.txt --error error-eng-rimgs.txt --wrap="python /cluster/scratch/stahla/src_runimgs/run_embeddings.py eng"
conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:55:00 --output output-ger-rimgs.txt --error error-ger-rimgs.txt --wrap="python /cluster/scratch/stahla/src_runimgs/run_embeddings.py ger"



conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:55:00 --output output-eng-best.txt --error error-eng-best.txt --wrap="python /cluster/scratch/stahla/src_best/run_embeddings.py --language eng"
conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:55:00 --output output-ger-best.txt --error error-ger-best.txt --wrap="python /cluster/scratch/stahla/src_best/run_embeddings.py --language ger"




# create nk_singleimage with updated code
conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-eng-comb-nkexp.txt --error error-eng-comb-nkexp.txt --wrap="python /cluster/scratch/stahla/src_runimgs/run_cluster.py --language eng --mode nkexp"

conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-ger-comb-nkexp.txt --error error-ger-comb-nkexp.txt --wrap="python /cluster/scratch/stahla/src_runimgs/run_cluster.py --language ger --mode nkexp"



# mirror graphs

conda activate nlpplus_updated && cd /cluster/scratch/stahla/src_eng_mirror/ && sbatch --ntasks=50 --time=23:55:00 --output output-eng_mirror-emb.txt --error error-eng_mirror-emb.txt --wrap="python /cluster/scratch/stahla/src_eng_mirror/run_embeddings.py --language eng" && cd $SCRATCH
conda activate nlpplus_updated && cd /cluster/scratch/stahla/src_ger_mirror/ && sbatch --ntasks=50 --time=23:55:00 --output output-ger_mirror-emb.txt --error error-ger_mirror-emb.txt --wrap="python /cluster/scratch/stahla/src_ger_mirror/run_embeddings.py --language ger" && cd $SCRATCH




