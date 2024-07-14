conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:58:00 --output output-eng-comb-nkc.txt --error error-eng-comb-nkc.txt --wrap="python /cluster/scratch/stahla/src/run_cluster.py --language eng --mode nkc"

conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:58:00 --output output-ger-comb-nkc.txt --error error-ger-comb-nkc.txt --wrap="python /cluster/scratch/stahla/src/run_cluster.py --language ger --mode nkc"




conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:58:00 --output output-eng-comb-mxc.txt --error error-eng-comb-mxc.txt --wrap="python /cluster/scratch/stahla/src/run_cluster.py --language eng --mode mxc"

conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:58:00 --output output-ger-comb-mxc.txt --error error-ger-comb-mxc.txt --wrap="python /cluster/scratch/stahla/src/run_cluster.py --language ger --mode mxc"












################################################### run experiments

conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-eng-comb-nkexp.txt --error error-eng-comb-nkexp.txt --wrap="python /cluster/scratch/stahla/src/run_cluster.py --language eng --mode nkexp"

conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-ger-comb-nkexp.txt --error error-ger-comb-nkexp.txt --wrap="python /cluster/scratch/stahla/src/run_cluster.py --language ger --mode nkexp"


conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-eng-comb-mxexp.txt --error error-eng-comb-mxexp.txt --wrap="python /cluster/scratch/stahla/src/run_cluster.py --language eng --mode mxexp"

conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-ger-comb-mxexp.txt --error error-ger-comb-mxexp.txt --wrap="python /cluster/scratch/stahla/src/run_cluster.py --language ger --mode mxexp"
