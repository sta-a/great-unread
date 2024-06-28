
# ToDo on Cluster: set data dir paths in DH class in utils (2x), activate conda env, nr workers in s2v, copy utils into analysis folder

scp /home/annina/Downloads/node2vec-master.zip stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp /home/annina/scripts/great_unread_nlp/src/environments/environment_s2v.yml stahla@euler.ethz.ch:
scp /home/annina/scripts/great_unread_nlp/src/environments/environment_network.yml stahla@euler.ethz.ch:


scp /home/annina/scripts/great_unread_nlp/src/environments/environment_nlpplus.yml stahla@euler.ethz.ch:

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

scp /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eng/analysis



scp /home/annina/scripts/great_unread_nlp/src/cluster/combinations.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/cluster
scp /home/annina/scripts/great_unread_nlp/src/cluster/evaluate.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src/cluster
scp /home/annina/scripts/great_unread_nlp/src/prediction/prediction_functions.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_new_chunk/prediction


scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_params/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_ger_params/analysis  
scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs_byauthor/analysis



scp -r /home/annina/scripts/great_unread_nlp/src/analysis/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src/
scp -r /home/annina/scripts/great_unread_nlp/src/analysis/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs/
scp -r /home/annina/scripts/great_unread_nlp/src/cluster/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src/
scp -r /home/annina/scripts/great_unread_nlp/src/prediction/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src/



scp /home/annina/scripts/great_unread_nlp/data_author/analysis/eng/interesting_networks.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis/eng
scp /home/annina/scripts/great_unread_nlp/data_author/analysis/ger/interesting_networks.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis/ger


scp -r /home/annina/scripts/great_unread_nlp/data_author/title_mapping stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author
scp -r /home/annina/scripts/great_unread_nlp/data_author/text_raw stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author
scp -r /home/annina/scripts/great_unread_nlp/data_author/canonscores stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author
scp -r /home/annina/scripts/great_unread_nlp/data_author/corpus_corrections stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author
scp -r /home/annina/scripts/great_unread_nlp/data_author/features stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author
scp -r /home/annina/scripts/great_unread_nlp/data_author/metadata stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author
scp -r /home/annina/scripts/great_unread_nlp/data_author/sentiscores stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author
scp -r /home/annina/scripts/great_unread_nlp/data_author/ngram_counts/eng/*.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/ngram_counts/eng
scp -r /home/annina/scripts/great_unread_nlp/data_author/ngram_counts/ger/*.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/ngram_counts/ger

scp -r /home/annina/scripts/great_unread_nlp/data_author/similarity/eng/sparsification_edgelists stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/
scp -r /home/annina/scripts/great_unread_nlp/data_author/similarity/ger/sparsification_edgelists stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger

scp -r /home/annina/scripts/great_unread_nlp/data_author/similarity/eng/sparsification_edgelists_s2v stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng
scp -r /home/annina/scripts/great_unread_nlp/data_author/similarity/ger/sparsification_edgelists_s2v stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger




# Download s2v dir
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/embeddings /home/annina/scripts/great_unread_nlp/data_author/s2v/eng/
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/embeddings /home/annina/scripts/great_unread_nlp/data_author/s2v/ger/

# Download whole s2v eng dir, excluding subdir simmxs (too big, not needed)
rsync -avzP --ignore-existing  --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng /home/annina/scripts/great_unread_nlp/data_author/s2v/
rsync -avzP --ignore-existing  --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger /home/annina/scripts/great_unread_nlp/data_author/s2v/




# Download single files from cluster
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/mxeval /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/mxeval /home/annina/scripts/great_unread_nlp/data_author/s2v/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/mxcomb /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/mxcomb /home/annina/scripts/great_unread_nlp/data_author/s2v/ger


rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/singleimages /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/singleimages /home/annina/scripts/great_unread_nlp/data_author/s2v/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/mx_singleimages /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/mx_singleimages /home/annina/scripts/great_unread_nlp/data_author/s2v/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/mx_gridimage /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/mx_gridimage /home/annina/scripts/great_unread_nlp/data_author/s2v/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/gridimage /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/gridimage /home/annina/scripts/great_unread_nlp/data_author/s2v/ger

scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data_author/s2v/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/mx_log_combinations.txt /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/mx_log_combinations.txt /home/annina/scripts/great_unread_nlp/data_author/s2v/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng/mx_log_smallmx.txt /home/annina/scripts/great_unread_nlp/data_author/s2v/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger/mx_log_smallmx.txt /home/annina/scripts/great_unread_nlp/data_author/s2v/ger


scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/nkeval /home/annina/scripts/great_unread_nlp/data_author/similarity/eng
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/nkeval /home/annina/scripts/great_unread_nlp/data_author/similarity/ger
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/nkcomb /home/annina/scripts/great_unread_nlp/data_author/similarity/eng
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/nkcomb /home/annina/scripts/great_unread_nlp/data_author/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/nk_log_clst.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/nk_log_clst.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/nk_log_combinations.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/nk_log_combinations.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/nk_noedges.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/eng
# scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/nk_noedges.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/ger # does not exists
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/mxeval /home/annina/scripts/great_unread_nlp/data_author/similarity/eng
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/mxeval /home/annina/scripts/great_unread_nlp/data_author/similarity/ger
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/mxcomb /home/annina/scripts/great_unread_nlp/data_author/similarity/eng
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/mxcomb /home/annina/scripts/great_unread_nlp/data_author/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng/mx_log_combinations.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger/mx_log_combinations.txt /home/annina/scripts/great_unread_nlp/data_author/similarity/ger


rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'sparsification_*' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/eng /home/annina/scripts/great_unread_nlp/data_author/similarity
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'sparsification_*' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity/ger /home/annina/scripts/great_unread_nlp/data_author/similarity



scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/nested_gridsearch /home/annina/scripts/great_unread_nlp/data_author
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/fold_idxs /home/annina/scripts/great_unread_nlp/data_author


# Download commands that contain cluster commands
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/embeddings_euler_byauthor.sh /home/annina/scripts/great_unread_nlp/src/euler
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/prediction_euler.sh /home/annina/scripts/great_unread_nlp/src/euler
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/prediction_euler_combinations.py /home/annina/scripts/great_unread_nlp/src/euler



rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis /home/annina/scripts/great_unread_nlp/data_author
rsync -avz --ignore-existing stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/analysis_s2v /home/annina/scripts/great_unread_nlp/data_author
rsync -avz --ignore-existing --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/similarity /home/annina/scripts/great_unread_nlp/data_author
rsync -avz --ignore-existing --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v /home/annina/scripts/great_unread_nlp/data_author




scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_eval.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/embedding_utils.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_eval_byauthor/analysis
scp /home/annina/scripts/great_unread_nlp/src/analysis/s2vcreator.py stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs_byauthor/analysis


# Gridimages
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'embeddings' --exclude 'singleimages' --exclude 'mx_singleimages' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng /home/annina/scripts/great_unread_nlp/data_author/s2v                 
rsync -avz --ignore-existing --exclude 'simmxs' --exclude 'embeddings' --exclude 'singleimages' --exclude 'mx_singleimages' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger /home/annina/scripts/great_unread_nlp/data_author/s2v                 



# To harddrive
rsync -avz --ignore-existing --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/eng /media/annina/MyBook/back-to-computer-240615/data_author/s2v 
rsync -avz --ignore-existing --exclude 'simmxs' stahla@euler.ethz.ch:/cluster/scratch/stahla/data_author/s2v/ger /media/annina/MyBook/back-to-computer-240615/data_author/s2v  


