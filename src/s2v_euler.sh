
# ToDo on Cluster: set data dir paths in DH class in utils (2x), activate conda env, nr workers in s2v, copy utils into analysis folder

scp /home/annina/Downloads/node2vec-master.zip stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp /home/annina/scripts/great_unread_nlp/src/environments/environment_s2v.yml stahla@euler.ethz.ch:
scp /home/annina/scripts/great_unread_nlp/src/environments/environment_network.yml stahla@euler.ethz.ch:
scp /home/annina/scripts/great_unread_nlp/src/environments/environment_nlpplus.yml stahla@euler.ethz.ch:
scp stahla@euler.ethz.ch:/cluster/home/stahla/environment-cluster_network.yml /home/annina/scripts/great_unread_nlp/src/environments
scp stahla@euler.ethz.ch:/cluster/home/stahla/environment-cluster_nlpplus_updated.yml /home/annina/scripts/great_unread_nlp/src/environments


# src dir to cluster
scp -r /home/annina/scripts/great_unread_nlp/src stahla@euler.ethz.ch:/cluster/scratch/stahla
scp -r /home/annina/scripts/great_unread_nlp/src_paramimgs stahla@euler.ethz.ch:/cluster/scratch/stahla
scp -r /home/annina/scripts/great_unread_nlp/src_paramimgs_mds stahla@euler.ethz.ch:/cluster/scratch/stahla


scp -r /home/annina/scripts/great_unread_nlp/src_ger stahla@euler.ethz.ch:/cluster/scratch/stahla
scp -r /home/annina/scripts/great_unread_nlp/src_paramimgs stahla@euler.ethz.ch:/cluster/scratch/stahla

scp -r /home/annina/scripts/great_unread_nlp/src_eng stahla@euler.ethz.ch:/cluster/scratch/stahla
scp -r /home/annina/scripts/great_unread_nlp/src_ger stahla@euler.ethz.ch:/cluster/scratch/stahla
scp -r /home/annina/scripts/great_unread_nlp/src_eng_params stahla@euler.ethz.ch:/cluster/scratch/stahla
scp -r /home/annina/scripts/great_unread_nlp/src_ger_params stahla@euler.ethz.ch:/cluster/scratch/stahla

scp -r /home/annina/scripts/great_unread_nlp/src/struc2vec-master stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eng


# Single files to src dir on cluster
scp /home/annina/scripts/great_unread_nlp/src/run_embeddings.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp /home/annina/scripts/great_unread_nlp/src/run_prediction.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_eval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/analysis

scp -r /home/annina/scripts/great_unread_nlp/data/similarity/eng/simmxs stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng
scp -r /home/annina/scripts/great_unread_nlp/data/similarity/ger/simmxs stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger

scp /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eng/analysis
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eng_params



scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/cluster
scp /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/cluster
scp /home/annina/scripts/great_unread_nlp/src/prediction/prediction_functions.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_new_chunk/prediction


scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_params/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_ger_params/analysis





scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/analysis_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/analysis_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/analysis_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/analysis_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor/analysis


scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/nkviz.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/nkviz.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/analysis_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/analysis_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs_byauthor/analysis




scp -r /home/annina/scripts/great_unread_nlp/src/analysis/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs/
scp -r /home/annina/scripts/great_unread_nlp/src/cluster/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src/
scp -r /home/annina/scripts/great_unread_nlp/src/prediction/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src/

scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor
scp -r /home/annina/scripts/great_unread_nlp/src/cluster stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp -r /home/annina/scripts/great_unread_nlp/src/cluster stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor
scp -r /home/annina/scripts/great_unread_nlp/src/cluster stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval
scp -r /home/annina/scripts/great_unread_nlp/src/cluster stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor


scp -r /home/annina/scripts/great_unread_nlp/src/run_cluster.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp -r /home/annina/scripts/great_unread_nlp/src/run_cluster.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor
scp -r /home/annina/scripts/great_unread_nlp/src/run_cluster.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval
scp -r /home/annina/scripts/great_unread_nlp/src/run_cluster.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor




scp /home/annina/scripts/great_unread_nlp/data/analysis/eng/interesting_networks.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis/eng
scp /home/annina/scripts/great_unread_nlp/data/analysis/ger/interesting_networks.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis/ger


scp -r /home/annina/scripts/great_unread_nlp/data/text_raw stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/canonscores stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/corpus_corrections stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/metadata stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/sentiscores stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/ngram_counts/eng/*.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/ngram_counts/eng
scp -r /home/annina/scripts/great_unread_nlp/data/ngram_counts/ger/*.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/ngram_counts/ger
scp -r /home/annina/scripts/great_unread_nlp/data/features stahla@euler.ethz.ch:/cluster/scratch/stahla/data

scp -r /home/annina/scripts/great_unread_nlp/data/similarity/eng/sparsification_edgelists stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/
scp -r /home/annina/scripts/great_unread_nlp/data/similarity/ger/sparsification_edgelists stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger

scp -r /home/annina/scripts/great_unread_nlp/data/similarity/eng/sparsification_edgelists_s2v stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng
scp -r /home/annina/scripts/great_unread_nlp/data/similarity/ger/sparsification_edgelists_s2v stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger



rsync -avzP --ignore-existing /home/annina/scripts/great_unread_nlp/data/features stahla@euler.ethz.ch:/cluster/home/stahla/data
rsync -avzP --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/features stahla@euler.ethz.ch:/cluster/home/stahla/data_author




# Download whole s2v eng dir, excluding subdir simmxs (too big, not needed)
rsync -avzP --ignore-existing  --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng /home/annina/scripts/great_unread_nlp/data/s2v/
rsync -avzP --ignore-existing  --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger /home/annina/scripts/great_unread_nlp/data/s2v/

# Grid images only
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'embeddings' --exclude 'singleimage' --exclude 'mx_singleimage' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng /home/annina/scripts/great_unread_nlp/data/s2v                 
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'embeddings' --exclude 'singleimage' --exclude 'mx_singleimage' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger /home/annina/scripts/great_unread_nlp/data/s2v                 





# Download single files from cluster
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mxeval /home/annina/scripts/great_unread_nlp/data/s2v/eng
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mxeval /home/annina/scripts/great_unread_nlp/data/s2v/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mxcomb /home/annina/scripts/great_unread_nlp/data/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mxcomb /home/annina/scripts/great_unread_nlp/data/s2v/ger


rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/singleimage /home/annina/scripts/great_unread_nlp/data/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/singleimage /home/annina/scripts/great_unread_nlp/data/s2v/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mx_singleimage /home/annina/scripts/great_unread_nlp/data/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mx_singleimage /home/annina/scriptsis/great_unread_nlp/data/s2v/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mx_gridimage /home/annina/scripts/great_unread_nlp/data/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mx_gridimage /home/annina/scripts/great_unread_nlp/data/s2v/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/gridimage /home/annina/scripts/great_unread_nlp/data/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/gridimage /home/annina/scripts/great_unread_nlp/data/s2v/ger

scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data/s2v/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data/s2v/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mx_log_combinations.txt /home/annina/scripts/great_unread_nlp/data/s2v/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mx_log_combinations.txt /home/annina/scripts/great_unread_nlp/data/s2v/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mx_log_smallmx.txt /home/annina/scripts/great_unread_nlp/data/s2v/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mx_log_smallmx.txt /home/annina/scripts/great_unread_nlp/data/s2v/ger


scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/nkcomb /home/annina/scripts/great_unread_nlp/data/similarity/eng
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/nkcomb /home/annina/scripts/great_unread_nlp/data/similarity/ger
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/nkeval /home/annina/scripts/great_unread_nlp/data/similarity/eng
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/nkeval /home/annina/scripts/great_unread_nlp/data/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/nk_log_clst.txt /home/annina/scripts/great_unread_nlp/data/similarity/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/nk_log_clst.txt /home/annina/scripts/great_unread_nlp/data/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/nk_log_combinations.txt /home/annina/scripts/great_unread_nlp/data/similarity/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/nk_log_combinations.txt /home/annina/scripts/great_unread_nlp/data/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/nk_noedges.txt /home/annina/scripts/great_unread_nlp/data/similarity/eng
# scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/nk_noedges.txt /home/annina/scripts/great_unread_nlp/data/similarity/ger # does not exists
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/mxeval /home/annina/scripts/great_unread_nlp/data/similarity/eng
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/mxeval /home/annina/scripts/great_unread_nlp/data/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data/similarity/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/mx_log_combinations.txt /home/annina/scripts/great_unread_nlp/data/similarity/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/mx_log_combinations.txt /home/annina/scripts/great_unread_nlp/data/similarity/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/mxcomb /home/annina/scripts/great_unread_nlp/data/similarity/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/mxcomb /home/annina/scripts/great_unread_nlp/data/similarity/ger


rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'sparsification*' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng /home/annina/scripts/great_unread_nlp/data/similarity
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'sparsification*' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger /home/annina/scripts/great_unread_nlp/data/similarity
rsync -avz --exclude 'simmxs' --exclude 'sparsification*' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng /home/annina/scripts/great_unread_nlp/data/similarity
rsync -avz --exclude 'simmxs' --exclude 'sparsification*' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger /home/annina/scripts/great_unread_nlp/data/similarity


scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/nested_gridsearch /home/annina/scripts/great_unread_nlp/data
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/fold_idxs /home/annina/scripts/great_unread_nlp/data




# Download commands that contain cluster commands
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/embeddings_euler.sh /home/annina/scripts/great_unread_nlp/src/euler
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/prediction_euler.sh /home/annina/scripts/great_unread_nlp/src/euler
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/prediction_euler_combinations.py /home/annina/scripts/great_unread_nlp/src/euler

scp /home/annina/scripts/great_unread_nlp/src/euler/prediction_euler.sh stahla@euler.ethz.ch:/cluster/scratch/stahla
scp /home/annina/scripts/great_unread_nlp/src/euler/prediction_euler_combinations.py stahla@euler.ethz.ch:/cluster/scratch/stahla
scp /home/annina/scripts/great_unread_nlp/src/euler/prediction_euler_combinations_byauthor.py stahla@euler.ethz.ch:/cluster/scratch/stahla



rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis /home/annina/scripts/great_unread_nlp/data
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis/eng /home/annina/scripts/great_unread_nlp/data/analysis
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis_s2v /home/annina/scripts/great_unread_nlp/data
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v /home/annina/scripts/great_unread_nlp/data
rsync -avz --ignore-existing --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity /home/annina/scripts/great_unread_nlp/data
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'embeddings' --exclude 'singleimage' --exclude 'mx_gridimage' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng /home/annina/scripts/great_unread_nlp/data/s2v                  
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'embeddings' --exclude 'singleimage' --exclude 'mx_gridimage' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger /home/annina/scripts/great_unread_nlp/data/s2v


# This command will recursively search for and delete all directories named mydir from the current directory and all its subdirectories.
find . -type d -name "mxcomb" # find all subdirs
find . -type d -name "*singleimage_cluster" -exec rm -rv {} +
find . -maxdepth 1 -type d ! -name "*singleimage*" ! -name "." -exec rm -rf {} + # delete all dirs that dont have singleimage in their name
find . -type d -name "mxcomb" -exec rm -rv {} +
find . -type d -name "mxeval" -exec rm -rv {} +
find . -type d -name "nkcomb" -exec rm -rv {} +
find . -type d -name "nkeval" -exec rm -rv {} +
# find . -type d -name "analysis" -exec rm -rv {} +
# find . -type d -name "analysis_s2v" -exec rm -rv {} +

find . -type f -name "mx_log*.txt" -delete
find . -type f -name "nk_log*.txt" -delete
find . -type f -name "nk_noedges.txt" -delete


# Add canon-ascat 
scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs_byauthor/analysis

scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor/analysis


scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/* stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/* stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimngs_byauthor/analysis


scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor/analysis




scp /home/annina/scripts/great_unread_nlp/src/run_embeddings.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eng_mirror
scp /home/annina/scripts/great_unread_nlp/src/run_embeddings.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_ger_mirror
scp /home/annina/scripts/great_unread_nlp/src/run_embeddings.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eng_mirror_byauthor
scp /home/annina/scripts/great_unread_nlp/src/run_embeddings.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_ger_mirror_byauthor
scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/nkviz.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_eval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eng_mirror/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/nkviz.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_eval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_ger_mirror/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/nkviz.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_eval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eng_mirror_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/nkviz.py /home/annina/scripts/great_unread_nlp/src/analysis/viz_utils.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_eval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_ger_mirror_byauthor/analysis



scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py /home/annina/scripts/great_unread_nlp/src/cluster/cluster_utils.py /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs/cluster                 
scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py /home/annina/scripts/great_unread_nlp/src/cluster/cluster_utils.py /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs_byauthor/cluster                 
scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py /home/annina/scripts/great_unread_nlp/src/cluster/cluster_utils.py /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs/cluster                 
scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py /home/annina/scripts/great_unread_nlp/src/cluster/cluster_utils.py /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs_byauthor/cluster      

scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py /home/annina/scripts/great_unread_nlp/src/cluster/cluster_utils.py /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/cluster            
scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py /home/annina/scripts/great_unread_nlp/src/cluster/cluster_utils.py /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor/cluster    
scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py /home/annina/scripts/great_unread_nlp/src/cluster/cluster_utils.py /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval/cluster            
scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py /home/annina/scripts/great_unread_nlp/src/cluster/cluster_utils.py /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor/cluster    
  

scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs7/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs7_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs7/cluster
scp /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs7/cluster_byauthor


scp  /home/annina/scripts/great_unread_nlp/src/utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla


scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs_byauthor
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs_byauthor

scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs
scp -r /home/annina/scripts/great_unread_nlp/src/analysis stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgs_byauthor


# Download s2v dir
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/embeddings /home/annina/scripts/great_unread_nlp/data/s2v/eng/
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/embeddings /home/annina/scripts/great_unread_nlp/data/s2v/ger/
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/embeddings /home/annina/scripts/great_unread_nlp/data_author/s2v/eng/
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/embeddings /home/annina/scripts/great_unread_nlp/data_author/s2v/ger/


rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/embeddings/*mirror* /home/annina/scripts/great_unread_nlp/data/s2v/eng/embeddings
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/embeddings/*mirror* /home/annina/scripts/great_unread_nlp/data/s2v/ger/embeddings
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/embeddings/*mirror* /home/annina/scripts/great_unread_nlp/data_author/s2v/eng/embeddings
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/embeddings/*mirror* /home/annina/scripts/great_unread_nlp/data_author/s2v/ger/embeddings



## From Cluster to harddrive
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/nested_gridsearch /media/annina/MyBook/back-to-computer-240615/data
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/nested_gridsearch /media/annina/MyBook/back-to-computer-240615/data_author


rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'mds' --exclude 'mxeval_paramcomb' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng /media/annina/MyBook/back-to-computer-240615/data/s2v 
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'mds' --exclude 'mxeval_paramcomb'  stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger /media/annina/MyBook/back-to-computer-240615/data/s2v
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'mds' --exclude 'mxeval_paramcomb'  stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng /media/annina/MyBook/back-to-computer-240615/data_author/s2v 
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'mds' --exclude 'mxeval_paramcomb'  stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger /media/annina/MyBook/back-to-computer-240615/data_author/s2v

rsync -avz --ignore-existing --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng /media/annina/MyBook/back-to-computer-240615/data/similarity
rsync -avz --ignore-existing --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger /media/annina/MyBook/back-to-computer-240615/data/similarity
rsync -avz --ignore-existing --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng /media/annina/MyBook/back-to-computer-240615/data_author/similarity
rsync -avz --ignore-existing --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger /media/annina/MyBook/back-to-computer-240615/data_author/similarity
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'mds' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis/eng /media/annina/MyBook/back-to-computer-240615/data/analysis 
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'mds' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis/ger /media/annina/MyBook/back-to-computer-240615/data/analysis 
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'mds' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis/eng /media/annina/MyBook/back-to-computer-240615/data_author/analysis 
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'mds' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis/ger /media/annina/MyBook/back-to-computer-240615/data_author/analysis 

rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis_s2v/eng /media/annina/MyBook/back-to-computer-240615/data/analysis_s2v 
rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis_s2v/ger /media/annina/MyBook/back-to-computer-240615/data/analysis_s2v 
rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis_s2v/eng /media/annina/MyBook/back-to-computer-240615/data_author/analysis_s2v 
rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis_s2v/ger /media/annina/MyBook/back-to-computer-240615/data_author/analysis_s2v 

rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis_s2v/eng/MxSingleViz2D3DHorizontal /media/annina/MyBook/back-to-computer-240615/data/analysis_s2v/eng
rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis_s2v/ger/MxSingleViz2D3DHorizontal /media/annina/MyBook/back-to-computer-240615/data/analysis_s2v/ger
rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis_s2v/eng/MxSingleViz2D3DHorizontal /media/annina/MyBook/back-to-computer-240615/data_author/analysis_s2v/eng
rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis_s2v/ger/MxSingleViz2D3DHorizontal /media/annina/MyBook/back-to-computer-240615/data_author/analysis_s2v/ger


rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis_s2v/eng/MxSingleViz2D3DHzAnalysis/*canon.png /media/annina/MyBook/back-to-computer-240615/data/analysis_s2v/eng/MxSingleViz2D3DHzAnalysis
rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis_s2v/ger/MxSingleViz2D3DHzAnalysis /media/annina/MyBook/back-to-computer-240615/data/analysis_s2v/ger
rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis_s2v/eng/MxSingleViz2D3DHzAnalysis /media/annina/MyBook/back-to-computer-240615/data_author/analysis_s2v/eng
rsync -avz --ignore-existing --exclude 'mds' --exclude 'MxSingleVizCluster_delete' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis_s2v/ger/MxSingleViz2D3DHzAnalysis /media/annina/MyBook/back-to-computer-240615/data_author/analysis_s2v/ger




rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis_s2v/eng/mx_topcanon_smallest_variance* /home/annina/scripts/great_unread_nlp/data/analysis_s2v/eng


## From Computer to harddrive
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/s2v/eng /media/annina/MyBook/back-to-computer-240615/data/s2v 
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/s2v/ger /media/annina/MyBook/back-to-computer-240615/data/s2v  
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/s2v/eng /media/annina/MyBook/back-to-computer-240615/data_author/s2v 
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/s2v/ger /media/annina/MyBook/back-to-computer-240615/data_author/s2v 
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/analysis/eng /media/annina/MyBook/back-to-computer-240615/data/analysis 
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/analysis/ger /media/annina/MyBook/back-to-computer-240615/data/analysis  
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/analysis/eng /media/annina/MyBook/back-to-computer-240615/data_author/analysis 
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/analysis/ger /media/annina/MyBook/back-to-computer-240615/data_author/analysis 

rsync -avz --ignore-existing --exclude 'simmxs' /home/annina/scripts/great_unread_nlp/data/similarity/eng /media/annina/MyBook/back-to-computer-240615/data/similarity
rsync -avz --ignore-existing --exclude 'simmxs' /home/annina/scripts/great_unread_nlp/data/similarity/ger /media/annina/MyBook/back-to-computer-240615/data/similarity
rsync -avz --ignore-existing --exclude 'simmxs' /home/annina/scripts/great_unread_nlp/data_author/similarity/eng /media/annina/MyBook/back-to-computer-240615/data_author/similarity
rsync -avz --ignore-existing --exclude 'simmxs' /home/annina/scripts/great_unread_nlp/data_author/similarity/ger /media/annina/MyBook/back-to-computer-240615/data_author/similarity

# From harddrive to computer
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/eng/embeddings /home/annina/scripts/great_unread_nlp/data/s2v/eng
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/ger/embeddings /home/annina/scripts/great_unread_nlp/data/s2v/ger
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/eng/embeddings /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/ger/embeddings /home/annina/scripts/great_unread_nlp/data_author/s2v/ger
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/analysis/eng /home/annina/scripts/great_unread_nlp/data/analysis

rsync -avz /media/annina/MyBook/back-to-computer-240615/data/s2v/eng/mirror_singleimage /home/annina/scripts/great_unread_nlp/data/s2v/eng
rsync -avz /media/annina/MyBook/back-to-computer-240615/data/s2v/eng/mx_mirror_singleimage /home/annina/scripts/great_unread_nlp/data/s2v/eng
rsync -avz /media/annina/MyBook/back-to-computer-240615/data/s2v/ger/mirror_singleimage /home/annina/scripts/great_unread_nlp/data/s2v/ger
rsync -avz /media/annina/MyBook/back-to-computer-240615/data/s2v/ger/mx_mirror_singleimage /home/annina/scripts/great_unread_nlp/data/s2v/ger
rsync -avz /media/annina/MyBook/back-to-computer-240615/data_author/s2v/eng/mirror_singleimage /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
rsync -avz /media/annina/MyBook/back-to-computer-240615/data_author/s2v/eng/mx_mirror_singleimage /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
rsync -avz /media/annina/MyBook/back-to-computer-240615/data_author/s2v/ger/mirror_singleimage /home/annina/scripts/great_unread_nlp/data_author/s2v/ger
rsync -avz /media/annina/MyBook/back-to-computer-240615/data_author/s2v/ger/mx_mirror_singleimage /home/annina/scripts/great_unread_nlp/data_author/s2v/ger

# Harddrive to Cluster
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/s2v/eng/embeddings stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/s2v/ger/embeddings stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/s2v/eng/embeddings stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/s2v/ger/embeddings stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger

rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/eng/mxcomb stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/ger/mxcomb stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/eng/mxcomb stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/ger/mxcomb stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger





# Download sh files from cluster 
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/*.sh /home/annina/scripts/great_unread_nlp/src/euler

scp /home/annina/scripts/great_unread_nlp/src/filter_eval_dfs.py stahla@euler.ethz.ch:/cluster/scratch/stahla/



scp /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_eval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py /home/annina/scripts/great_unread_nlp/src/analysis/embedding_eval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval/cluster
scp /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor/cluster

rsync -avz /home/annina/scripts/great_unread_nlp/data/similarity/eng/sparsification/*.pkl stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/sparsification/
rsync -avz /home/annina/scripts/great_unread_nlp/data/similarity/ger/sparsification/*.pkl stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/sparsification/
rsync -avz /home/annina/scripts/great_unread_nlp/data_author/similarity/eng/sparsification/*.pkl stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/sparsification/
rsync -avz /home/annina/scripts/great_unread_nlp/data_author/similarity/ger/sparsification/*.pkl stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/sparsification/


scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgsingle_hz/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgsingle_hz_byauthor/analysis


scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgsingle_al/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_runimgsingle_al_byauthor/analysis

scp /home/annina/scripts/great_unread_nlp/src/analysis/mxviz.py /home/annina/scripts/great_unread_nlp/src/analysis/experiments.py /home/annina/scripts/great_unread_nlp/src/analysis/topeval.py /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/analysis




scp -r /home/annina/scripts/great_unread_nlp/src/cluster stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp -r /home/annina/scripts/great_unread_nlp/src/cluster stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor


rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/similarity/eng/simmxs stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/similarity/eng/simmxs stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/similarity/eng/simmxs stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/similarity/eng/simmxs stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger



# Reupload src files if they have been deleted by cluster
rsync -avz --ignore-existing --exclude 'networks_to_embeddings' --exclude 'struc2vec-master_rwpath' --exclude 'struc2vec-master_improved_paths'  /home/annina/scripts/great_unread_nlp/src/* stahla@euler.ethz.ch:/cluster/scratch/stahla/src


# Synchronize harddrives /media/annina/MyBook/back-to-computer-240615/data_author/analysis_s2v/eng/MxNkAnalysis
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615 /media/annina/elements
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v /media/annina/elements/back-to-computer-240615/data
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v /media/annina/elements/back-to-computer-240615/data_author
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/analysis_s2v/eng/MxNkAnalysis /media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/eng

rsync -avz --ignore-existing /home/annina/Documents/thesis /media/annina/MyBook/back-to-computer-240615



rsync -avz /home/annina/scripts/great_unread_nlp/data/label_predict stahla@euler.ethz.ch:/cluster/scratch/stahla/data
rsync -avz /home/annina/scripts/great_unread_nlp/data_author/label_predict stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author
scp -r /home/annina/scripts/great_unread_nlp/src_label stahla@euler.ethz.ch:/cluster/scratch/stahla
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/src_label  stahla@euler.ethz.ch:/cluster/scratch/stahla

rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data/label_predict /home/annina/scripts/great_unread_nlp/data
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/label_predict /home/annina/scripts/great_unread_nlp/data_author







scp /home/annina/scripts/great_unread_nlp/src/prediction/prediction_functions.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor/prediction
scp /home/annina/scripts/great_unread_nlp/src/run_prediction.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_byauthor





# Reupload embeddings
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/eng/embeddings/*dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True* stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/embeddings
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/ger/embeddings/*dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True* stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/embeddings
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/eng/embeddings/*dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True* stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/embeddings
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/ger/embeddings/*dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True* stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/embeddings
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/eng/mxcomb stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/ger/mxcomb stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/eng/mxcomb stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/ger/mxcomb stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/eng/mxeval stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data/s2v/ger/mxeval stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/eng/mxeval stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/
rsync -avz --ignore-existing /media/annina/MyBook/back-to-computer-240615/data_author/s2v/ger/mxeval stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/similarity/eng/sparsification_edgelists_s2v/index-mapping.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/sparsification_edgelists_s2v/
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/similarity/ger/sparsification_edgelists_s2v/index-mapping.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/sparsification_edgelists_s2v/
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/similarity/eng/sparsification_edgelists_s2v/index-mapping.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/sparsification_edgelists_s2v/
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/similarity/ger/sparsification_edgelists_s2v/index-mapping.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/sparsification_edgelists_s2v/
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/similarity/eng/sparsification_edgelists/index-mapping.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/sparsification_edgelists/
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data/similarity/ger/sparsification_edgelists/index-mapping.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger/sparsification_edgelists/
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/similarity/eng/sparsification_edgelists/index-mapping.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/sparsification_edgelists/
rsync -avz --ignore-existing /home/annina/scripts/great_unread_nlp/data_author/similarity/ger/sparsification_edgelists/index-mapping.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/sparsification_edgelists/







