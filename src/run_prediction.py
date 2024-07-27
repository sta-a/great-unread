# %%
# %load_ext autoreload
# %autoreload 2
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
parser = argparse.ArgumentParser()
parser.add_argument('language')
parser.add_argument('data_dir')
parser.add_argument('tasks', nargs="*", type=str)
# If --testing flag is set, testing is set to bool(True). 
# If --no-testing flag is set, testing is set to bool(False)
parser.add_argument('--testing', action=argparse.BooleanOptionalAction)
parser.add_argument('--features', type=str, help='Specify the features to use if you want to run the cv for only one feature.')
parser.add_argument('--fold', type=int, help='Specify the fold to run (0-based index) if you want to run the cv for only one fold.')
args = parser.parse_args()
language = args.language
data_dir = args.data_dir
tasks = args.tasks
testing = args.testing
features_arg = args.features
fold_arg = args.fold

print(language, data_dir, tasks, testing )
n_outer_folds = 5

if 'data_author' in data_dir:
    by_author = True
else:
    by_author = False
print('by author', by_author)



# Use full features set for classification to avoid error with underrepresented classes
sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
metadata_dir = os.path.join(data_dir, 'metadata', language)
canonscores_dir = os.path.join(data_dir, 'canonscores')
features_dir = os.path.join(data_dir, 'features', language)
gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
idxs_dir = os.path.join(data_dir, 'fold_idxs', language)
if not os.path.exists(gridsearch_dir):
    os.makedirs(gridsearch_dir, exist_ok=True)
if not os.path.exists(idxs_dir):
    os.makedirs(idxs_dir, exist_ok=True)




current_working_dir = os.path.basename(os.getcwd())
print(current_working_dir)
def write_idxs_to_file(task, label_type, features, outer_fold, train_idx, test_idx):
    train_filename = os.path.join(idxs_dir, f'{task}_{label_type}_{features}_fold_{outer_fold}_trainidx.txt')
    test_filename = os.path.join(idxs_dir, f'{task}_{label_type}_{features}_fold_{outer_fold}_testidx.txt')

    # Write train indices to file
    with open(train_filename, 'w') as train_file:
        train_file.write(','.join(map(str, train_idx)))

    # Write test indices to file
    with open(test_filename, 'w') as test_file:
        test_file.write(','.join(map(str, test_idx)))



for task in tasks:
    task_params = get_task_params(task, testing, language)
    print(task_params['param_grid'])
    for label_type in task_params['labels']:

        if features_arg:
            # If features are passed as an argument, use them
            features = [features_arg]
        else:
            # Otherwise, use the features from task_params
            features = task_params['features']

        for features in features:
            print(f'Task: {task}, Label_type: {label_type}, Features: {features}\n')
            X, y = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir, by_author)

            # Run grid search for nested cv
            cv_outer = CustomGroupKFold(n_splits=5, stratified=task_params['stratified']).split(X, y.values.ravel())
            X_ = X.copy(deep=True)
            y_ = y.copy(deep=True)

            for outer_fold, (train_idx, test_idx) in enumerate(cv_outer):
                print('Outer fold: ', outer_fold)
                write_idxs_to_file(task, label_type, features, outer_fold, train_idx, test_idx)

                # If fold_arg is provided, only run the specified fold
                if fold_arg is not None and outer_fold != fold_arg:
                    continue

                X_train_outer, X_test_outer = X_.iloc[train_idx], X_.iloc[test_idx]
                y_train_outer, y_test_outer = y_.iloc[train_idx], y_.iloc[test_idx]
                
                gridsearch_object = run_gridsearch(
                    gridsearch_dir=gridsearch_dir, 
                    language=language, 
                    task=task, 
                    label_type=label_type, 
                    features=features, 
                    fold=outer_fold, 
                    columns_list=task_params['columns_list'], 
                    task_params=task_params, 
                    X_train=X_train_outer,
                    y_train=y_train_outer)

                estimator = gridsearch_object.best_estimator_

                y_pred = estimator.predict(X_test_outer)
                y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
                y_pred['fold'] = outer_fold
                y_pred.to_csv(
                    os.path.join(gridsearch_dir, f'y-pred_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
                    index=False)
                y_true = pd.DataFrame(y_test_outer).rename({'y': 'y_true'}, axis=1).reset_index().rename({'index': 'file_name'})
                y_true['fold'] = outer_fold              
                y_true.to_csv(
                    os.path.join(gridsearch_dir, f'y-true_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
                    index=False)

# %%
