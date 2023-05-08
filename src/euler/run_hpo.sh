#!/bin/bash

# source activate nlp
# [[-d /cluster/scratch/stahla/data/nested_gridsearch]] && rm -r /cluster/scratch/stahla/data/nested_gridsearch


sbatch -n 30 --time=4:00:00 --mem-per-cpu=1024 --output="output_eng.txt" --open-mode=truncate --error="errors_eng.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src/hpo.py eng /cluster/scratch/stahla/data regression"
sbatch -n 30 --time=4:00:00 --mem-per-cpu=1024 --output="output_ger.txt" --open-mode=truncate --error="errors_ger.txt" --open-mode=truncate --wrap="python /cluster/scratch/stahla/src/hpo.py ger /cluster/scratch/stahla/data regression"
