# source activate nlp
# conda env export > ~/environment.yml
# scp ~/environment.yml stahla@euler.ethz.ch:


# VPN
# Copy to cluster
rm -r /home/annina/scripts/great_unread_nlp/src/__pycache__/ && scp -r /home/annina/scripts/great_unread_nlp/src/ stahla@euler.ethz.ch:/cluster/scratch/stahla/
scp -r /home/annina/scripts/great_unread_nlp/data/text_raw/ stahla@euler.ethz.ch:/cluster/scratch/stahla/data/
scp -r /home/annina/scripts/great_unread_nlp/data/sentiscores/ stahla@euler.ethz.ch:/cluster/scratch/stahla/data/
scp -r /home/annina/scripts/great_unread_nlp/data/metadata/ stahla@euler.ethz.ch:/cluster/scratch/stahla/data/ 
scp -r /home/annina/scripts/great_unread_nlp/data/features_None/ stahla@euler.ethz.ch:/cluster/scratch/stahla/data/
scp -r /home/annina/scripts/great_unread_nlp/data/canonscores/ stahla@euler.ethz.ch:/cluster/scratch/stahla/data/
scp -r /home/annina/scripts/great_unread_nlp/data/text_tokenized/ stahla@euler.ethz.ch:/cluster/scratch/stahla/data/


#ssh stahla@euler.ethz.ch
# Copying a file from the cluster to your PC (current directory)
rm -r /home/annina/scripts/great_unread_nlp/data/nested_gridsearch && cd /home/annina/scripts/great_unread_nlp/data/ && scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/nested_gridsearch/ .


# Run locally from command line
source activate nlp
cd /home/annina/scripts/great_unread_nlp/src/ && python hpo.py eng /home/annina/scripts/great_unread_nlp/data regression --testing
