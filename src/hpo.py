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
from hpo_helpers import get_data, CustomGroupKFold, get_task_params, run_gridsearch, make_cv_splits
from scipy.stats import pearsonr


start = time.time()
if from_commandline:
    parser = argparse.ArgumentParser()
    parser.add_argument('languages')
    parser.add_argument('data_dir')
    parser.add_argument('tasks', nargs="*", type=str)
    # If --testing flag is set, testing is set to bool(True). 
    # If --no-testing flag is set, testing is set to bool(False)
    parser.add_argument('--testing', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    languages = [args.languages]
    data_dir = args.data_dir
    tasks = args.tasks
    testing = args.testing
else:
    # Don't use defaults because VSC interactive mode can't handle command line arguments
    languages = ['eng', 'ger']
    data_dir = '/home/annina/scripts/great_unread_nlp/data'
    tasks = ['regression', 'binary', 'library', 'multiclass'] # 
    testing = False
print(languages, data_dir, tasks, testing )
n_outer_folds = 5


for language in languages:

    # Use full features set for classification to avoid error with underrepresented classes
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores', language)
    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
    gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
    cv_dir = os.path.join(data_dir, 'cv', language)
    if not os.path.exists(gridsearch_dir):
        os.makedirs(gridsearch_dir, exist_ok=True)
    if not os.path.exists(cv_dir):
        os.makedirs(cv_dir, exist_ok=True)
    print(cv_dir)

    columns_list = [
    ['average_sentence_embedding', 'doc2vec_chunk_embedding'],
    ['average_sentence_embedding', 'doc2vec_chunk_embedding', 'pos']]
    if language == 'eng':
        columns_list.extend([
            ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->'], 
            ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->', 'pos']])
    # columns_list.extend([['passthrough']])
    # if testing:
    #     columns_list = [columns_list[-1]]

    for task in tasks:
        task_params = get_task_params(task, testing)
        for label_type in task_params['labels']:
            for features in task_params['features']:
                print(f'\n\n##################################\n Task: {task}, label_type {label_type}, features {features}\n')
                with open(os.path.join(gridsearch_dir, f'best-models-in-refit.csv'), 'a')as f:
                    f.write(f'\n{language},{task},{label_type},{features}\n')
                X, y = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
                  
                # Run grid search for nested cv
                for outer_fold in range(0, n_outer_folds):
                    with open(os.path.join(gridsearch_dir, f'best-models-in-refit.csv'), 'a')as f:
                        f.write(f'fold_{outer_fold}')

                    train_idxs_path = os.path.join(cv_dir, f'train_{language}_{task}_{label_type}_fold-{outer_fold}.csv')
                    test_idxs_path = os.path.join(cv_dir, f'test_{language}_{task}_{label_type}_fold-{outer_fold}.csv')

                    X_ = X.copy(deep=True)
                    y_ = y.copy(deep=True)
                    
                    # Load cv idxs
                    if not os.path.exists(train_idxs_path):
                        make_cv_splits(language, task, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir, cv_dir, n_outer_folds, task_params)
                    train_idxs = [line.strip() for line in open(train_idxs_path, 'r')]
                    test_idxs = [line.strip() for line in open(test_idxs_path, 'r')]

                    X_train_outer = X_.loc[X_.index.isin(train_idxs)]
                    X_test_outer = X_.loc[X_.index.isin(test_idxs)]
                    y_train_outer = y_.loc[y_.index.isin(train_idxs)]
                    y_test_outer = y_.loc[y_.index.isin(test_idxs)]
                    print(f'\nX_train_outer{X_train_outer.shape}, X_test_outer{X_test_outer.shape},  y_train_outer{y_train_outer.shape}, y_test_outer{y_test_outer.shape}')
                    print(f'\nX_train_outer: {X_train_outer.index.nunique()}, X_test_outer: {X_test_outer.index.nunique()}, y_train_outer: {y_train_outer.index.nunique()}, y_test_outer: {y_test_outer.index.nunique()}')

                    gridsearch_object = run_gridsearch(
                        gridsearch_dir=gridsearch_dir, 
                        language=language, 
                        task=task, 
                        label_type=label_type, 
                        features=features, 
                        fold=outer_fold, 
                        columns_list=columns_list, 
                        task_params=task_params, 
                        X_train=X_train_outer,
                        y_train=y_train_outer)

                    estimator = gridsearch_object.best_estimator_
                    best_parameters = pd.DataFrame(gridsearch_object.best_params_)

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

                    corr, pval = pearsonr(y_true['y_true'], y_pred['y_pred'])
                    best_parameters['outer_corr'] = corr
                    best_parameters['outer_corr_pval'] = pval
                    print('testing corr', corr, pval)

                    best_parameters.to_csv(
                        os.path.join(gridsearch_dir, f'best-params_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
                        index=False)

 # %%
