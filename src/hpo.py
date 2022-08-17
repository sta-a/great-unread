# %%
%load_ext autoreload
%autoreload 2
from_commandline = False

import warnings
warnings.simplefilter('once', RuntimeWarning)
import argparse
import time
import os
import pandas as pd
from copy import deepcopy
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hpo_helpers import get_data, ColumnTransformer, CustomGroupKFold, apply_harmonic_pvalue, get_task_params

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
    languages = ['eng', 'ger']
    data_dir = '/home/annina/scripts/great_unread_nlp/data'
    tasks = ['regression', 'binary', 'library', 'multiclass'] # 
    testing = False
print(languages, data_dir, tasks, testing )


def run_gridsearch(X, y, task_params, features, fold, columns_list):
    print(f'Inner CV: X {X.shape}, y {y.shape}')
    # Get data, set 'file_name' column as index
    cv = CustomGroupKFold(n_splits=5, stratified=task_params['stratified']).split(X, y.values.ravel())
    
    ## Parameter Grid
    # Params that are constant between grids
    print('\ncolumns list', columns_list, '\n')
    constant_param_grid = {'drop_columns__columns_to_drop': columns_list}
    print(constant_param_grid)
    param_grid = deepcopy(task_params['param_grid'])  
    [d.update(constant_param_grid) for d in param_grid]
    print('param grid', param_grid)

    ## Pipeline
    pipe = Pipeline(steps=[
        ('drop_columns', ColumnTransformer()),
        ('scaler', StandardScaler()),
        #('dimred', SelectPercentile()),
        ('clf', SVR())
        ])

    gridsearch = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=task_params['scoring'],
        n_jobs=-1,
        refit=task_params['refit'],
        cv=cv,
        verbose=1,
        error_score='raise', #np.nan
        return_train_score=False
    )

    gridsearch.fit(X, y.values.ravel())

    # print('Best estimator: ', gridsearch.best_estimator_)
    # print('Best params: ' , gridsearch.best_params_)
    cv_results = pd.DataFrame(gridsearch.cv_results_)
    cv_results.insert(0, 'features', features)
    cv_results.insert(0, 'fold', fold)

    if task == 'regression':
        cv_results['harmonic_pvalue'] = cv_results.apply(apply_harmonic_pvalue, axis=1)

    cv_results.to_csv(os.path.join(gridsearch_dir, f'inner-cv-result_{language}_{task}_{label_type}_{features}_fold-{fold}.csv'), index=False, na_rep='NaN')
    with open(os.path.join(gridsearch_dir, f'gridsearch-object_{language}_{task}_{label_type}_{features}_fold-{fold}.pkl'), 'wb') as f:
        pickle.dump(gridsearch, f, -1)

    best_estimator = gridsearch.best_estimator_
    # feature_importances = best_estimator.named_steps["clf"].feature_importances_
    # print(feature_importances)

    return best_estimator


for language in languages:

    # Use full features set for classification to avoid error with underrepresented classes
    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
    gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores', language)
    if not os.path.exists(gridsearch_dir):
        os.makedirs(gridsearch_dir, exist_ok=True)

    columns_list = [
    ['average_sentence_embedding', 'doc2vec_chunk_embedding'],
    ['average_sentence_embedding', 'doc2vec_chunk_embedding', 'pos']]
    if language == 'eng':
        columns_list.extend([
            ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->'], 
            ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->', 'pos']])
    columns_list.extend([['passthrough']])
    # if testing:
    #     columns_list = [columns_list[-1]]

    for task in tasks:
        print('Task: ', task)
        task_params = get_task_params(task, testing)
        for label_type in task_params['labels']:
            for features in task_params['features']:
                print('features', features)
                X, y = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)

                # Run grid search on full data for final model
                # estimator = run_gridsearch(
                #     X=deepcopy(X), 
                #     y=deepcopy(y), 
                #     task_params=task_params, 
                #     features=features, 
                #     fold='fulldata', 
                #     columns_list=columns_list)

                # Run grid search for nested cv
                cv_outer = CustomGroupKFold(n_splits=5, stratified=task_params['stratified']).split(X, y.values.ravel())
                y_tests = []
                y_preds = []
                for outer_fold, (train_idx, test_idx) in enumerate(cv_outer):
                    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
                    y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
                    print(f'X_train_outer{X_train_outer.shape}, X_test_outer{X_test_outer.shape},  y_train_outer{y_train_outer.shape}, y_test_outer{y_test_outer.shape}')
                    estimator = run_gridsearch(X_train_outer, y_train_outer, task_params, features, outer_fold, columns_list)
                    y_pred = estimator.predict(X_test_outer)
                    y_tests.append(y_test_outer)
                    y_preds.extend(y_pred.tolist())
                y_tests = pd.concat(y_tests)
                y_tests['y_pred'] = y_preds
                outer_cv_result = y_tests.rename({'y': 'y_true'}, axis=1)
                outer_cv_result.to_csv(os.path.join(gridsearch_dir, f'outer-cv-predicted_{language}_{task}_{label_type}_{features}.csv'), index=True, na_rep='NaN')
 # %%
