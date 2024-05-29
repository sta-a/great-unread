# %%
# %load_ext autoreload
# %autoreload 2



'''

This script runs the loops from the run_predictions script repeatedly to test if the cv splits produce the same indices every time.
'''
from_commandline = True

import warnings
warnings.simplefilter('once', RuntimeWarning)
import argparse
import time
import os
import pandas as pd
import numpy as np
from prediction.prediction_functions import get_data, CustomGroupKFold, get_task_params, run_gridsearch


start = time.time()
if from_commandline:
    parser = argparse.ArgumentParser()
    parser.add_argument('language')
    parser.add_argument('data_dir')
    parser.add_argument('tasks', nargs="*", type=str)
    # If --testing flag is set, testing is set to bool(True). 
    # If --no-testing flag is set, testing is set to bool(False)
    parser.add_argument('--testing', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    language = args.language
    data_dir = args.data_dir
    tasks = args.tasks
    testing = args.testing
else:
    # Don't use defaults because VSC interactive mode can't handle command line arguments
    language = 'eng'
    data_dir = '../data'
    tasks = ['regression-importances']
    testing = True
print(language, data_dir, tasks, testing )
n_outer_folds = 5



# Use full features set for classification to avoid error with underrepresented classes
sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
metadata_dir = os.path.join(data_dir, 'metadata', language)
canonscores_dir = os.path.join(data_dir, 'canonscores')
features_dir = os.path.join(data_dir, 'features', language)
gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
if not os.path.exists(gridsearch_dir):
    os.makedirs(gridsearch_dir, exist_ok=True)

numiter = 2
fold_train_indices = {}
for niter in range(numiter):
    for task in tasks:
        task_params = get_task_params(task, testing, language)
        print(task_params['param_grid'])
        for label_type in task_params['labels']:
            # for features in task_params['features']:
            for features in ['cacb']:
                print(f'Task: {task}, Label_type: {label_type}, Features: {features}\n')
                X, y = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)

                # Run grid search for nested cv
                cv_outer = CustomGroupKFold(n_splits=5, stratified=task_params['stratified']).split(X, y.values.ravel())
                X_ = X.copy(deep=True)
                y_ = y.copy(deep=True)


                for outer_fold, (train_idx, test_idx) in enumerate(cv_outer):
                    if outer_fold not in fold_train_indices:
                        fold_train_indices[outer_fold] = []
                    fold_train_indices[outer_fold].append(train_idx)

# Create DataFrames for each fold number
dfs = {}
for fold_num, train_indices_list in fold_train_indices.items():
    df = pd.DataFrame(train_indices_list)

    dfs[fold_num] = df

output_dir = '/home/annina/scripts/great_unread_nlp/src/test'
# Save DataFrames to files
for fold_num, df in dfs.items():
    filename = f"{output_dir}/train_indices_fold_{fold_num}.csv"
    df.to_csv(filename, index=False)
# %%
