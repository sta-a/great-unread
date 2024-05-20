# source activate nlp
# conda env export > ~/environment.yml
# scp ~/environment.yml stahla@euler.ethz.ch:


#ssh stahla@euler.ethz.ch
# Copying a file from the cluster to your PC (current directory)
rm -r /home/annina/scripts/great_unread_nlp/data/nested_gridsearch && cd /home/annina/scripts/great_unread_nlp/data/ && scp -r stahla@euler.ethz.ch:/cluster/scratch/stahla/data/nested_gridsearch/ .


# Run locally from command line
source activate nlp
cd /home/annina/scripts/great_unread_nlp/src/ && python run_prediction.py eng /home/annina/scripts/great_unread_nlp/data regression --testing
