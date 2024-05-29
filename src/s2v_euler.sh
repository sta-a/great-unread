
# ToDo on Cluster: set paths in DH class in utils (2x), activate conda env, nr workers, copy utils into analysis folder

scp /home/annina/Downloads/node2vec-master.zip stahla@euler.ethz.ch:/cluster/scratch/stahla/src
scp /home/annina/scripts/great_unread_nlp/src/environments/environment_s2v.yml stahla@euler.ethz.ch:
scp /home/annina/scripts/great_unread_nlp/src/environments/environment_network.yml stahla@euler.ethz.ch:
scp /cluster/scratch/stahla/data/s2v/ger/mx_singleimages

scp /home/annina/scripts/great_unread_nlp/src/environments/environment_nlpplus.yml stahla@euler.ethz.ch:

# src dir to cluster
scp -r /home/annina/scripts/great_unread_nlp/src stahla@euler.ethz.ch:/cluster/scratch/stahla
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

scp -r /home/annina/scripts/great_unread_nlp/src/analysis/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src/
scp -r /home/annina/scripts/great_unread_nlp/src/analysis/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src_paramimgs/
scp -r /home/annina/scripts/great_unread_nlp/src/cluster/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src/
scp -r /home/annina/scripts/great_unread_nlp/src/prediction/ stahla@euler.ethz.ch:/cluster/scratch/stahla/src/



scp /home/annina/scripts/great_unread_nlp/data/analysis/eng/interesting_networks.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis/eng
scp /home/annina/scripts/great_unread_nlp/data/analysis/ger/interesting_networks.csv stahla@euler.ethz.ch:/cluster/scratch/stahla/data/analysis/ger


scp -r /home/annina/scripts/great_unread_nlp/data/text_raw stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/canonscores stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/corpus_corrections stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/features stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/metadata stahla@euler.ethz.ch:/cluster/scratch/stahla/data
scp -r /home/annina/scripts/great_unread_nlp/data/sentiscores stahla@euler.ethz.ch:/cluster/scratch/stahla/data

scp -r /home/annina/scripts/great_unread_nlp/data/similarity/eng/sparsification_edgelists stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng/
scp -r /home/annina/scripts/great_unread_nlp/data/similarity/ger/sparsification_edgelists stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger

scp -r /home/annina/scripts/great_unread_nlp/data/similarity/eng/sparsification_edgelists_s2v stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/eng
scp -r /home/annina/scripts/great_unread_nlp/data/similarity/ger/sparsification_edgelists_s2v stahla@euler.ethz.ch:/cluster/scratch/stahla/data/similarity/ger


# Download s2v dir
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/embeddings/burrows-500_simmel-5-10* /home/annina/scripts/great_unread_nlp/data/s2v/eng/embeddings/
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/embeddings /home/annina/scripts/great_unread_nlp/data/s2v/eng/
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/embeddings /home/annina/scripts/great_unread_nlp/data/s2v/ger/
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v_params/eng/embeddings /home/annina/scripts/great_unread_nlp/data/s2v/eng/
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v_params/ger/embeddings /home/annina/scripts/great_unread_nlp/data/s2v/ger/


scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/simmxs /home/annina/scripts/great_unread_nlp/data/s2v/eng/
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/simmxs /home/annina/scripts/great_unread_nlp/data/s2v/ger/

# Download single files from cluster
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mxeval/*_results.csv /home/annina/scripts/great_unread_nlp/data/s2v/eng/mxeval
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mxeval/*_results.csv /home/annina/scripts/great_unread_nlp/data/s2v/ger/mxeval
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mxcomb /home/annina/scripts/great_unread_nlp/data/s2v/eng
scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mxcomb /home/annina/scripts/great_unread_nlp/data/s2v/ger
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/eng/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data/s2v/eng
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/data/s2v/ger/mx_log_clst.txt /home/annina/scripts/great_unread_nlp/data/s2v/ger



# Download commands that contain cluster commands
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/embeddings_euler.sh /home/annina/scripts/great_unread_nlp/src/euler
scp stahla@euler.ethz.ch:/cluster/scratch/stahla/prediction_euler.sh /home/annina/scripts/great_unread_nlp/src/euler





