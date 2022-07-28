# %%
# %load_ext autoreload
# %autoreload 2
# from_commandline = False
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
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Lasso
warnings.simplefilter(action='ignore', category=FutureWarning) # Suppress FutureWarning ####################################
from xgboost import XGBRegressor, XGBClassifier
from models_functions import get_data, _score_regression, get_min_fold_size, permute_params, XGBClassifierMulticlassImbalanced, ColumnTransformer, AuthorCV, get_averaged_chunk_features

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
    testing = False #True, False
    data_dir = '/home/annina/scripts/great_unread_nlp/data'

features_dir = features_dir = os.path.join(data_dir, 'features_150', language)  ############################3
results_dir = os.path.join(data_dir, 'results', language)
sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
metadata_dir = os.path.join(data_dir, 'metadata', language)
canonscores_dir = os.path.join(data_dir, 'canonscores', language)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir, exist_ok=True)


# All paramters
# Separate grids for conditional parameters
param_grid_regression = [
    {'clf': (SVR(),),
    'clf__C': [0.1, 1],
    'clf__epsilon': [0.1, 1]},
    {'clf': (Lasso(),),
    'clf__alpha': [1,10]},
    {'clf': (XGBRegressor(objective='reg:squarederror', random_state=7),),
    'clf__max_depth': [2,4,6,8, 20],
    'clf__learning_rate': [None, 0.01, 0.033, 0.1],
    'clf__colsample_bytree': [0.33, 0.60, 0.75]}, 
    ]

param_grid_binary = [
    {'clf': (SVC(class_weight='balanced'),),
    'clf__C': [0.1, 1, 10, 100, 1000]},
    {'clf': (XGBClassifierMulticlassImbalanced(objective='binary:logistic', random_state=7, use_label_encoder=False),),
    'clf__max_depth': [2,4,6,8, 20],
    'clf__learning_rate': [None, 0.01, 0.033, 0.1],
    'clf__colsample_bytree': [0.33, 0.60, 0.75]}, 
    ]

param_grid_multiclass= [
    {'clf': (SVC(class_weight='balanced'),),
    'clf__C': [0.1, 1, 10, 100, 1000]},
    {'clf': (XGBClassifierMulticlassImbalanced(objective='multi:softmax', random_state=7, use_label_encoder=False),),
    'clf__max_depth': [2,4,6,8, 20],
    'clf__learning_rate': [None, 0.01, 0.033, 0.1],
    'clf__colsample_bytree': [0.33, 0.60, 0.75]}, 
    ]

if testing:
    print('Use testing param grid.')
    param_grid_regression = [
        # {'clf': (SVR(),),
        # 'clf__C': [0.1],
        # 'clf__epsilon': [0.1]},
        # {'clf': (Lasso(),),
        # 'clf__alpha': [1]},
        {'clf': (XGBRegressor(objective='reg:squarederror', random_state=7),),
        'clf__max_depth': [20],
        'clf__learning_rate': [0.1],
        'clf__colsample_bytree': [0.75]}, 
        ]

    param_grid_binary = [
        # {'clf': (SVC(class_weight='balanced'),),
        # 'clf__C': [1]},
        {'clf': (XGBClassifierMulticlassImbalanced(objective='binary:logistic', random_state=7, use_label_encoder=False),),
        'clf__max_depth': [20],
        'clf__learning_rate': [0.1],
        'clf__colsample_bytree': [0.75]}, 
        ]

    param_grid_multiclass= [
        # {'clf': (SVC(class_weight='balanced'),),
        # 'clf__C': [1]},
        {'clf': (XGBClassifierMulticlassImbalanced(objective='multi:softmax', random_state=7, use_label_encoder=False),),
        'clf__max_depth': [20],
        'clf__learning_rate': [0.1],
        'clf__colsample_bytree': [0.75]}, 
        ]

models = {
    'regression': {
        'model': ['xgboost'], 
        'labels': ['textblob', 'sentiart', 'combined'],
        'eval_metric': 'corr',
        'stratified': False,
        'param_grid': param_grid_regression,
        'param_data_subset': ['passthrough', ColumnTransformer(columns_to_drop=['_fulltext'])],
        },
    'binary': {
        'model': ['svc', 'xgboost'], 
        'labels': ['binary'],
        'eval_metric': 'balanced_acc',
        'stratified': True,
        'param_grid': param_grid_binary,
        'param_data_subset': [],
        }
    }

models['library'] = models['binary']
models['library']['labels'] = ['library']
models['multiclass'] = models['binary']
models['multiclass']['labels'] = ['multiclass']
models['multiclass']['eval_metric'] = ['f1']
models['multiclass']['param_grid'] = param_grid_multiclass

# Overwrite for testing 
if testing:
    models['regression']['model'] = ['xgboost']
    models['regression']['labels'] = ['sentiart']

    models['binary']['model'] = ['xgboost']

    models['library']['model'] = ['xgboost']

    models['multiclass']['model'] = ['xgboost']

# %%
for task_type in ['regression']: #['regression', 'binary', 'library', 'multiclass']:
    model = models[task_type]
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

        # Copy param grid for every combination 
        for label_type in model['labels']:

            X, y = get_data(task_type, language, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
            X_full = deepcopy(X)
            print('NaN in X: ', X.isnull().values.any())
            print('NaN in y: ', y.isnull().values.any())
            print('X, y shape', X.shape, y.shape)
            cv = AuthorCV(df=X, n_folds=5, stratified=model['stratified']).get_folds()
            
            ## Parameter Grid
            # Find number of components/features
            min_nr_columns_fulltext = ColumnTransformer(columns_to_drop=columns_list[-1] + ['_fulltext']).transform(X, y).shape[1]
            min_nr_columns_chunk = ColumnTransformer(columns_to_drop=columns_list[-1] + ['_chunk']).transform(X, y).shape[1]
            max_k = min(min_nr_columns_fulltext, min_nr_columns_chunk)
            min_fold_size = get_min_fold_size(cv) # Smallest number of samples in a split
            max_components = min(max_k, min_fold_size) # max n_components for PCA

            ## Parameter Grid
            data_subset_params = model['param_data_subset'] + \
                    [ColumnTransformer(columns_to_drop=['_chunk'])] + \
                    [FunctionTransformer(get_averaged_chunk_features, kw_args={'X_full': X_full})]
            # Params that are constant between grids
            constant_param_grid = {
                'data_subset': data_subset_params,
                'drop_columns__columns_to_drop': columns_list,
                'dimred': # Pass transformers as parameters because of 'passthrough' option
                    ['passthrough'] + 
                    # Create separate instance of transformers for every parameter combination
                    permute_params(SelectKBest, k=list(range(10, max_k, 5)), score_func=[f_regression, mutual_info_regression]) +  
                    permute_params(PCA, n_components=[None] + list(range(2, max_components, 5))) 
            }

            if testing:
                constant_param_grid['dimred'] = ['passthrough']

            param_grid = deepcopy(model['param_grid'])  
            [d.update(constant_param_grid) for d in param_grid]


            ## Pipeline
            pipe = Pipeline(steps=[
                ('data_subset', ColumnTransformer()),
                ('drop_columns', ColumnTransformer()),
                ('dimred', SelectKBest()),
                ('clf', SVR())
                ])    
            
            ## Grid Search
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=_score_regression,
                n_jobs=1,
                refit=model['eval_metric'],
                cv=cv,
                verbose=1,
                #pre_dispatch= #####################################,
                error_score='raise', #np.nan
                return_train_score=False
            )
            gs.fit(X, y.values.ravel()) # Don't pass file_name column, dtype error
            print("Best parameter (CV score=%0.3f):" % gs.best_score_)
            print(gs.best_params_)
            df = pd.DataFrame(gs.cv_results_)
            df.to_csv('test')





# %%
# results = pd.DataFrame()
# for model in models[task_type]['model']:
#     for features_type in models[task_type]['features']:
#         df = deepcopy(features_mapping[features_type])
#         for label_type in models[task_type]['labels']:
#             labels = deepcopy(labels_mapping[label_type])
#             for dimensionality_reduction in models[task_type]['dimensionality_reduction']:
#                 for drop_columns in models[task_type]['drop_columns']:
#                     print(task_type, model, label_type, features_type, dimensionality_reduction, len(drop_columns))
                    
#                     cv_info = {'language': language,
#                             'task_type': task_type,
#                             'model': model,
#                             'features_type': features_type,
#                             'label_type': label_type,
#                             'dimensionality_reduction': dimensionality_reduction,
#                             'drop_columns': drop_columns}

#                     #info_string = f'{cv_params['language']}_{cv_params['task_type']}_{cv_params['model']}_label-{cv_params['label_type']}_feat-{self.features_type}_dimred-{self.dimensionality_reduction}_drop-{len(self.drop_columns)}'
#                     cv_params = {**cv_info, 
#                                 'df': df, 
#                                 'labels': labels, 
#                                 'results_dir': results_dir,
#                                 'verbose': True,
#                                 'eval_metric': models[task_type]['eval_metric']}
#                     cv_info['drop_columns'] = len(cv_info['drop_columns'])
#                     info_string = f'{"_".join(str(value) for value in cv_info.values())}'
#                     cv_params['info_string'] = info_string

#                     if task_type == 'regression':
#                         experiment = Regression(**cv_params)
#                     elif (task_type == 'binary') or (task_type == 'library'):
#                         experiment = BinaryClassification(**cv_params)
#                     elif task_type == 'multiclass':
#                         experiment = MulticlassClassification(**cv_params)

#                     returned_values = experiment.run()
#                     result = pd.concat(objs=[pd.DataFrame([cv_info])]+returned_values, axis=1)
                    
#                     with open(f'{results_dir}results-{language}-{task_type}-log.csv', 'a') as f:
#                         result.to_csv(f, header=False)
#                     results = pd.concat([results, result], axis=0)
#                     print('\n-----------------------------------------------------------\n')

# results_path = os.path.join(results_dir, f'results-{language}-{task_type}-final.csv')
# results.to_csv(results_path, index=False, sep='\t')


# # %%
# results = pd.read_csv(results_path, sep='\t')
# results
# # %%
# if task_type == 'regression':
#     results = results[results['validation_corr_pvalue']<=0.1]
# eval_metric = 'validation_' + models['eval_metric']
# best_model = results.sort_values(by=eval_metric, axis=0)


# %%
