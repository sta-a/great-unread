conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:58:00 --output output-eng-comb-nkc_byauthor.txt --error error-eng-comb-nkc_byauthor.txt --wrap="python /cluster/scratch/stahla/src_byauthor/run_cluster.py --language eng --mode nkc --by_author"

conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:58:00 --output output-ger-comb-nkc_byauthor.txt --error error-ger-comb-nkc_byauthor.txt --wrap="python /cluster/scratch/stahla/src_byauthor/run_cluster.py --language ger --mode nkc --by_author"




conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:58:00 --output output-eng-comb-mxc_byauthor.txt --error error-eng-comb-mxc_byauthor.txt --wrap="python /cluster/scratch/stahla/src_byauthor/run_cluster.py --language eng --mode mxc --by_author"

conda activate nlpplus_updated && sbatch --mem-per-cpu=10000 --time=23:58:00 --output output-ger-comb-mxc_byauthor.txt --error error-ger-comb-mxc_byauthor.txt --wrap="python /cluster/scratch/stahla/src_byauthor/run_cluster.py --language ger --mode mxc --by_author"





################################################### run experiments

conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-eng-comb-nkexp_byauthor.txt --error error-eng-comb-nkexp_byauthor.txt --wrap="python /cluster/scratch/stahla/src_byauthor/run_cluster.py --language eng --mode nkexp --by_author"

conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-ger-comb-nkexp_byauthor.txt --error error-ger-comb-nkexp_byauthor.txt --wrap="python /cluster/scratch/stahla/src_byauthor/run_cluster.py --language ger --mode nkexp --by_author"


conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-eng-comb-mxexp_byauthor.txt --error error-eng-comb-mxexp_byauthor.txt --wrap="python /cluster/scratch/stahla/src_byauthor/run_cluster.py --language eng --mode mxexp --by_author"

conda activate nlpplus_updated && sbatch --mem-per-cpu=100000 --time=23:58:00 --output output-ger-comb-mxexp_byauthor.txt --error error-ger-comb-mxexp_byauthor.txt --wrap="python /cluster/scratch/stahla/src_byauthor/run_cluster.py --language ger --mode mxexp --by_author"
