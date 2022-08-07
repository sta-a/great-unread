# %%
%load_ext autoreload
%autoreload 2
from_commandline = False

from tokenize import group
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
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models_helpers import get_data, permute_params, ColumnTransformer, CustomGroupKFold, apply_harmonic_pvalue, get_model, average_chunk_features, analyze_cv, get_document_features

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
    languages = ['eng'] # ['eng', 'ger']
    data_dir = '/home/annina/scripts/great_unread_nlp/data'
    tasks = ['multiclass'] #['regression', 'binary', 'library', 'multiclass']
    testing = True #True, False
print(languages, data_dir, tasks, testing )


for language in languages:

    # Use full features set for classification to avoid error with underrepresented classes
    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
    gridsearch_dir = os.path.join(data_dir, 'gridsearch', language)
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
    if testing:
        columns_list = [columns_list[-1]]

    for task in tasks:
        print('Task: ', task)
        model = get_model(task, testing)
        print('stratified', model['stratified'])
        for features_type in model['features']:

            # Copy param grid for every combination 
            for label_type in model['labels']: ##########################
                # Get data, set 'file_name' column as index
                X, y = get_data(task, language, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
                cv = CustomGroupKFold(n_splits=5, stratified=model['stratified']).split(X, y.values.ravel())
                # analyze_cv(X, cv)
                
                ## Parameter Grid
                # Params that are constant between grids
                # Permute_params() to create separate instance of transformers for every parameter combination
                constant_param_grid = {
                    'data_subset': 
                        [FunctionTransformer(get_document_features)] + # For all models, run without chunk features
                        [FunctionTransformer(average_chunk_features)] + # For all models, run with averaged chunk features
                        model['param_data_subset'], # Add model-specific feature levels # For regression, also run chunk features
                    'drop_columns__columns_to_drop': 
                        columns_list,
                    'dimred': # Pass transformers as parameters because of 'passthrough' option
                        ['passthrough'] + 
                        permute_params(SelectPercentile, percentile=list(range(10, 60, 10)), score_func=[f_regression, mutual_info_regression]) +  # try between 10 and 100 % of features
                        permute_params(PCA, n_components=[None] + list(np.arange(0.81, 1.05, 0.06).tolist())) # try None and explaining between 81 % to 99 % of the variance
                }

                #if testing:
                constant_param_grid['dimred'] = ['passthrough'] #########################

                param_grid = deepcopy(model['param_grid'])  
                [d.update(constant_param_grid) for d in param_grid]
                print('Param grid, ', param_grid, 'nr_combinations: ', len(param_grid))


                ## Pipeline
                pipe = Pipeline(steps=[
                    ('data_subset', ColumnTransformer()),
                    ('drop_columns', ColumnTransformer()),
                    ('scaler', StandardScaler()),
                    ('dimred', SelectPercentile()),
                    ('clf', SVR())
                    ])    
                
                ## Grid Search
                # gs = GridSearchCV(
                #     estimator=pipe,
                #     param_grid=param_grid,
                #     scoring=model['scoring'],
                #     n_jobs=-1, ############################################################
                #     refit=model['refit'],
                #     cv=cv,
                #     verbose=3,
                #     return_train_score=False
                #     #pre_dispatch= 
                # )
                
                n_iter = 500
                randsearch = RandomizedSearchCV(
                    estimator=pipe, 
                    param_distributions=param_grid, 
                    n_iter=n_iter, 
                    scoring=model['scoring'],
                    n_jobs=-1, ############################################################
                    refit=model['refit'],
                    cv=cv,
                    verbose=2,
                    random_state=4,
                    return_train_score=False
                )
                randsearch.fit(X, y.values.ravel())

                print('Best estimator: ', randsearch.best_estimator_)
                print('Best params: ' , randsearch.best_params_)
                cv_results = pd.DataFrame(randsearch.cv_results_)

                if task == 'regression':
                    cv_results['harmonic_pvalue'] = cv_results.apply(apply_harmonic_pvalue, axis=1)
                else:
                    print(f'Best CV score={randsearch.best_score_}')

                if not testing:
                    cv_results.to_csv(os.path.join(gridsearch_dir, f'cv-results_{language}_{task}_{label_type}_niter-{n_iter}.csv'), index=False)
                    with open(os.path.join(gridsearch_dir, f'randsearch-object_{language}_{task}_{label_type}_niter-{n_iter}.pkl'), 'wb') as f:
                        pickle.dump(randsearch, f, -1)
                    print(f'Time for running regression for 1 language: {time.time()-start}') # 10985s for multiclass

# %%
