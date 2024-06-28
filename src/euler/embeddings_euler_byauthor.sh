
# fix imports in run_embeddings.py
# Create embeddings




conda activate s2v && cd /cluster/scratch/stahla/src_eng_byauthor/ && sbatch --ntasks=50 --time=23:00:00 --output output-eng-emb_byauthor.txt --error error-eng-emb_byauthor.txt --wrap="python /cluster/scratch/stahla/src_eng_byauthor/run_embeddings.py eng"

conda activate s2v && cd /cluster/scratch/stahla/src_ger_byauthor/ && sbatch --ntasks=50 --time=23:00:00 --output output-ger-emb_byauthor.txt --error error-ger-emb_byauthor.txt --wrap="python /cluster/scratch/stahla/src_ger_byauthor/run_embeddings.py ger"


conda activate s2v && cd /cluster/scratch/stahla/src_eng_params_byauthor/ && sbatch --ntasks=50 --time=23:00:00 --output output-eng-emb-params_byauthor.txt --error error-eng-emb-params_byauthor.txt --wrap="python /cluster/scratch/stahla/src_eng_params_byauthor/run_embeddings.py eng"
conda activate s2v && cd /cluster/scratch/stahla/src_ger_params_byauthor/ && sbatch --ntasks=50 --time=23:00:00 --output output-ger-emb-params_byauthor.txt --error error-ger-emb-params_byauthor.txt --wrap="python /cluster/scratch/stahla/src_ger_params_byauthor/run_embeddings.py ger"



# Run matrix combinations
conda activate nlpplus && sbatch --mem-per-cpu=10000 --time=23:00:00 --output output_file-eng-eval_byauthor.txt --error error_file-eng-eval_byauthor.txt --wrap="python /cluster/scratch/stahla/src_eval_byauthor/run_embeddings.py eng"
conda activate nlpplus && sbatch --mem-per-cpu=10000 --time=23:00:00 --output output_file-ger-eval_byauthor.txt --error error_file-ger-eval_byauthor.txt --wrap="python /cluster/scratch/stahla/src_eval_byauthor/run_embeddings.py ger"




###############################################################
# run parameter evaluation images

conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:00:00 --output output-eng-pimgs_byauthor.txt --error error-eng-pimgs_byauthor.txt --wrap="python /cluster/scratch/stahla/src_paramimgs_byauthor/run_embeddings.py eng"
conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:00:00 --output output-eng-pimgs-mds_byauthor.txt --error error-eng-pimgs-mds_byauthor.txt --wrap="python /cluster/scratch/stahla/src_paramimgs_mds_byauthor/run_embeddings.py eng"


conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:00:00 --output output-ger-pimgs_byauthor.txt --error error-ger-pimgs_byauthor.txt --wrap="python /cluster/scratch/stahla/src_paramimgs_byauthor/run_embeddings.py ger"
conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:00:00 --output output-ger-pimgs_mds_byauthor.txt --error error-ger-pimgs_mds_byauthor.txt --wrap="python /cluster/scratch/stahla/src_paramimgs_mds_byauthor/run_embeddings.py ger"


# create single images for run parameter
conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:00:00 --output output-eng-rimgs_byauthor.txt --error error-eng-rimgs_byauthor.txt --wrap="python /cluster/scratch/stahla/src_runimgs_byauthor/run_embeddings.py eng"
conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:00:00 --output output-ger-rimgs_byauthor.txt --error error-ger-rimgs_byauthor.txt --wrap="python /cluster/scratch/stahla/src_runimgs_byauthor/run_embeddings.py ger"


