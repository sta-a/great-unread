# %%
%load_ext autoreload
%autoreload 2
from_commandline = False

import argparse
import os
from pyclbr import Function
import warnings
import pandas as pd
from copy import deepcopy
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Lasso
warnings.simplefilter(action='ignore', category=FutureWarning) # Suppress FutureWarning ####################################
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler
from models_functions import get_data, score_regression, get_min_fold_size, permute_params, XGBClassifierMulticlassImbalanced, ColumnTransformer, CustomGroupKFold, score_binary, score_multiclass, get_pvalue, get_model, average_chunk_features

if from_commandline:
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='eng')
    parser.add_argument('--task_type', default='regression')
    # if --testing flag is set, testing is set to bool(True). 
    # If --no-testing flag is set, testing is set to bool(False).
    parser.add_argument('--testing', action=argparse.BooleanOptionalAction)
    parser.add_argument('--data_dir', default='/cluster/scratch/stahla/data/')
    args = parser.parse_args()
    language = args.language
    task_type = args.task_type
    testing = args.testing
    data_dir = args.data_dir
else:
    # Don't use defaults because VSC interactive can't handle command line arguments
    language = 'eng' #'eng', 'ger'
    task_type = 'regression' #'regression', 'binary', 'library', 'multiclass'
    testing = True #True, False
    data_dir = '/home/annina/scripts/great_unread_nlp/data'

# Use full features set for classification to avoid error with underrepresented classes
features_dir = features_dir = os.path.join(data_dir, 'features_None', language)  ############################3
results_dir = os.path.join(data_dir, 'results', language)
sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
metadata_dir = os.path.join(data_dir, 'metadata', language)
canonscores_dir = os.path.join(data_dir, 'canonscores', language)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir, exist_ok=True)


for language in ['eng']: # ['eng', 'ger']:
    columns_list = [
    ['average_sentence_embedding', 'doc2vec_chunk_embedding'],
    ['average_sentence_embedding', 'doc2vec_chunk_embedding', 'pos']]
    if language == 'eng':
        columns_list.extend([
            ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->'], 
            ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->', 'pos']])
    if testing:
        columns_list = [columns_list[-1]]
        print(columns_list)

    for task_type in ['regression']: #['regression', 'binary', 'library', 'multiclass']:
        print('Task type: ', task_type)
        cv_results_dfs = []
        model = get_model(task_type, testing)
        print('Model: ', model)


        # Copy param grid for every combination 
        for label_type in model['labels']:
            # Get data, set 'file_name' column as index
            X, y = get_data(task_type, language, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
            cv = CustomGroupKFold(n_splits=5, stratified=model['stratified']).split(X, y)

            
            ## Parameter Grid
            # Params that are constant between grids
            # Permute_params() to create separate instance of transformers for every parameter combination
            constant_param_grid = {
                'data_subset': 
                    [ColumnTransformer(columns_to_drop=['_chunk'])] + [FunctionTransformer(average_chunk_features)] + model['param_data_subset'], # Use book features
                'drop_columns__columns_to_drop': 
                    columns_list,
                'dimred': # Pass transformers as parameters because of 'passthrough' option
                    ['passthrough'] + 
                    permute_params(SelectPercentile, percentile=list(range(10, 100, 10)), score_func=[f_regression, mutual_info_regression]) +  # try between 10 and 100 % of features
                    permute_params(PCA, n_components=[None] + list(np.arange(0.81, 1.01, 0.02).tolist())) # try None and explaining between 81 % to 99 % of the variance
            }

            if testing:
                constant_param_grid['dimred'] = ['passthrough']

            param_grid = deepcopy(model['param_grid'])  
            [d.update(constant_param_grid) for d in param_grid]
            print('Param grid, ', param_grid)


            ## Pipeline
            pipe = Pipeline(steps=[
                ('data_subset', ColumnTransformer()),
                ('drop_columns', ColumnTransformer()),
                ('scaler', StandardScaler()),
                ('dimred', SelectPercentile()),
                ('clf', SVR())
                ])    
            
            ## Grid Search
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=model['scoring'],
                n_jobs=1,
                refit=model['refit'],
                cv=cv,
                verbose=1,
                #pre_dispatch= #####################################,
                error_score='raise', #np.nan
                return_train_score=False
            )
            gs.fit(X, y.values.ravel()) # Don't pass file_name column, dtype error
            #print(f'Best CV score={gs.best_score_}')
            print('Best estimator: ', gs.best_estimator_)
            print('Best params: ' , gs.best_params_)
            df = pd.DataFrame(gs.cv_results_)
            cv_results_dfs.append(df)

            results = pd.concat(cv_results_dfs)
            if task_type == 'regression':
                df['harmonic_pvalue'] = df.apply(get_pvalue, axis=1)
            results.to_csv(os.path.join(results_dir, f'results_{language}_{task_type}.csv'), index=False)



# # %%
# results = pd.read_csv(results_path, sep='\t')
# results
# # %%
# if task_type == 'regression':
#     results = results[results['validation_corr_pvalue']<=0.1]
# eval_metric = 'validation_' + models['eval_metric']
# best_model = results.sort_values(by=eval_metric, axis=0)


# %%
