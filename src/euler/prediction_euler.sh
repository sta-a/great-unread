
#### run senti
# eng
conda activate nlpplus && sbatch --ntasks=100 --time=300:00:00 --mem-per-cpu=8000 --output="output_eng_pred_senti_chunk.txt" --open-mode=truncate --error="errors_eng_pred_senti_chunk.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_senti_chunk/run_prediction.py eng /cluster/scratch/stahla/data regression-senti"

conda activate nlpplus && sbatch --ntasks=100 --time=300:00:00 --mem-per-cpu=8000 --output="output_eng_pred_senti_cacb.txt" --open-mode=truncate --error="errors_eng_pred_senti_cacb.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_senti_cacb/run_prediction.py eng /cluster/scratch/stahla/data regression-senti"


# ger
conda activate nlpplus && sbatch --ntasks=100 --time=300:00:00 --mem-per-cpu=8000 --output="output_ger_pred_senti_chunk.txt" --open-mode=truncate --error="errors_ger_pred_senti_chunk.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_senti_chunk/run_prediction.py ger /cluster/scratch/stahla/data regression-senti"

conda activate nlpplus && sbatch --ntasks=100 --time=300:00:00 --mem-per-cpu=8000 --output="output_ger_pred_senti_cacb.txt" --open-mode=truncate --error="errors_ger_pred_senti_cacb.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_senti_cacb/run_prediction.py ger /cluster/scratch/stahla/data regression-senti"



# run with only chunk/cacb for speed
conda activate nlpplus && sbatch --ntasks=100 --time=300:00:00 --mem-per-cpu=8000 --output="output_eng_pred_new.txt" --open-mode=truncate --error="errors_eng_pred_new.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_new/run_prediction.py eng /cluster/scratch/stahla/data regression-canon"

iconda activate nlpplus && sbatch --ntasks=100 --time=300:00:00 --mem-per-cpu=8000 --output="output_eng_pred_new_chunk.txt" --open-mode=truncate --error="errors_eng_pred_new_chunk.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_new_chunk/run_prediction.py eng /cluster/scratch/stahla/data regression-canon"

iconda activate nlpplus && sbatch --ntasks=100 --time=300:00:00 --mem-per-cpu=8000 --output="output_eng_pred_new_cacb.txt" --open-mode=truncate --error="errors_eng_pred_new_cacb.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_new_cacb/run_prediction.py eng /cluster/scratch/stahla/data regression-canon"





# normal run mode
conda activate nlpplus && sbatch --ntasks=100 --time=72:00:00 --mem-per-cpu=10000 --output="output_eng_pred.txt" --open-mode=truncate --error="errors_eng_pred.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src/run_prediction.py eng /cluster/scratch/stahla/data regression-canon"


conda activate nlpplus && sbatch --ntasks=100 --time=72:00:00 --mem-per-cpu=10000 --output="output_ger_pred.txt" --open-mode=truncate --error="errors_ger_pred.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src/run_prediction.py ger /cluster/scratch/stahla/data regression-canon"





# conda activate nlpplus && sbatch --ntasks=110 --time=200:00:00 --mem-per-cpu=10000 --output="output_eng_pred_parallel.txt" --open-mode=truncate --error="errors_eng_pred_paralle_parallell.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_parallel/run_prediction.py eng /cluster/scratch/stahla/data regression-canon"
# conda activate nlpplus && sbatch --ntasks=110 --time=200:00:00 --mem-per-cpu=10000 --output="output_eng_pred_parallel.txt" --open-mode=truncate --error="errors_eng_pred_paralle_parallell.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_parallel_2/run_prediction.py eng /cluster/scratch/stahla/data regression-canon"
# conda activate nlpplus && sbatch --ntasks=110 --time=200:00:00 --mem-per-cpu=10000 --output="output_eng_pred_parallel.txt" --open-mode=truncate --error="errors_eng_pred_paralle_parallell.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src_parallel_3/run_prediction.py eng /cluster/scratch/stahla/data regression-canon"


