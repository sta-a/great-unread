# %%
# %load_ext autoreload
# %autoreload 2
from_commandline = True

import argparse
import os
import pandas as pd
from copy import deepcopy
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor, XGBClassifier
from models_functions import drop_columns_transformer, get_data, custom_pca

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
    data_dir = '../data/'

print('testing', testing)
features_dir = features_dir = os.path.join(data_dir, 'features', language)  ############################3
results_dir = os.path.join(data_dir, 'results', language)
sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
metadata_dir = os.path.join(data_dir, 'metadata', language)
canonscores_dir = os.path.join(data_dir, 'canonscores', language)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir, exist_ok=True)

# import time
# start = time.time()
# task_type = 'book'
# book_df = pd.read_csv(os.path.join(features_dir, f'{task_type}_features.csv')).rename(columns = {'book_name': 'file_name'})
# print('time to read data: ', time.time()-start)

# start = time.time()
# x = deepcopy(book_df)
# print(time.time() - start)


# %%

# All paramters
'''
models: 'svr', 'lasso', 'xgboost', 'svc'
dimensionality_reduction: 'ss_pca_0_95', 'k_best_f_reg_0_10', 'k_best_mutual_info_0_10', 'rfe', None
'''
# features_mapping = {
#     'book': book_df,
#     'chunk': chunk_df,
#     'baac': book_and_averaged_chunk_df,
#     'cacb': chunk_and_copied_book_df}
# labels_mapping = {
#     'canon': canon_labels,
#     'textblob': textblob_labels,
#     'sentiart': sentiart_labels,
#     'combined': combined_labels,
#     'binary': binary_labels,
#     'multiclass': multiclass_labels,
#     'library': library_labels}
drop_columns_list = [
    ['average_sentence_embedding', 'doc2vec_chunk_embedding'],
    ['average_sentence_embedding', 'doc2vec_chunk_embedding', 'pos']]
if language == 'eng':
    drop_columns_list.extend([
        ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->'], 
        ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->', 'pos']])


param_dict = {
    'regression': {
        'model': ['xgboost'], 
        'features': ['book', 'chunk', 'baac', 'cacb'],
        'labels': ['textblob', 'sentiart', 'combined'],
        'dimensionality_reduction': [None], 
        'drop_columns': drop_columns_list,
        'eval_metric': 'corr'},
    'binary': {
        'model': ['svc', 'xgboost'], 
        'features': ['book', 'baac'],
        'labels': ['binary'],
        'dimensionality_reduction': [None], 
        'drop_columns': drop_columns_list,
        'eval_metric': 'balanced_acc'},
    'multiclass': {
        'model': ['svc', 'xgboost'], 
        'features': ['book', 'baac'],
        'labels': ['multiclass'],
        'dimensionality_reduction': [None],
        'drop_columns': drop_columns_list,
        'eval_metric': 'f1'}}

param_dict['library'] = param_dict['binary']
param_dict['library']['labels'] = ['library']

# Overwrite for testing 
if testing == True:
    for task_type, d in param_dict.items():
        d['drop_columns'] = [drop_columns_list[-1]]

    param_dict['regression']['model'] = ['xgboost']
    param_dict['regression']['features'] = ['book', 'chunk']
    param_dict['regression']['labels'] = ['sentiart']

    param_dict['binary']['model'] = ['xgboost']
    param_dict['binary']['features'] = ['book']

    param_dict['library']['model'] = ['xgboost']
    param_dict['library']['features'] = ['book']

    param_dict['multiclass']['model'] = ['xgboost']
    param_dict['multiclass']['features'] = ['book']

# %%
import pandas as pd
import statistics
import os
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def custom_pca(X):
    print('----------------------------------------', task_type, features_type, label_type, drop_columns)
    X = StandardScaler().fit_transform(X)
    print('################################3 X before pca: ', X.shape)
    for i in range(5, X.shape[1], int((X.shape[1] - 5) / 10)):
        pca = PCA(n_components=i)
        X_trans = pca.fit_transform(X)
        if pca.explained_variance_ratio_.sum() >= 0.95:
            break
    print('##################################Nr dim after pca: ',X_trans.shape)
    return X_trans

for task_type in ['regression']: #['regression', 'binary', 'library', 'multiclass']:
    for language in ['eng']: # ['eng', 'ger']:
        for features_type in param_dict[task_type]['features']:
            print('Features: ', features_type)
            for label_type in param_dict[task_type]['labels']:
                X, y = get_data(task_type, language, features_type, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
                print('X before transformer ', X.shape)
                for drop_columns in param_dict[task_type]['drop_columns']:
                    pipe = Pipeline(steps=[
                        ('drop_columns', FunctionTransformer(drop_columns_transformer, kw_args={'drop_columns': drop_columns})),
                        ('dimred', FunctionTransformer(custom_pca)),
                        # Set any estimator, different options for estimators are passed as parameters
                        ('clf', SVR)
                        ])    

                    # Separate dicts for conditional parameters
                    all_param_grid = {'reduce_dim': ['passthrough', FunctionTransformer(custom_pca)]}
                    param_grid = [
                        {'clf': (XGBRegressor(objective='reg:squarederror', random_state=42),)}, 
                        {'clf': (SVR(),)}
                        ]
                    #[d.update(all_param_grid) for d in param_grid]
                    print('param grid\n', param_grid)
                    # param_grid = dict(
                    #     svr__C=[0.1, 1]
                    # )
                    
                    gs = GridSearchCV(pipe, param_grid, n_jobs=1) #############
                    print('NaN in X: ', X.isnull().values.any())
                    print('NaN in y: ', y.isnull().values.any())
                    print('X, y shape', X.shape, y.shape)
                    gs.fit(X, y.values.ravel())
                    print("Best parameter (CV score=%0.3f):" % gs.best_score_)
                    print(gs.best_params_)





# %%
# results = pd.DataFrame()
# for model in param_dict[task_type]['model']:
#     for features_type in param_dict[task_type]['features']:
#         df = deepcopy(features_mapping[features_type])
#         for label_type in param_dict[task_type]['labels']:
#             labels = deepcopy(labels_mapping[label_type])
#             for dimensionality_reduction in param_dict[task_type]['dimensionality_reduction']:
#                 for drop_columns in param_dict[task_type]['drop_columns']:
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
#                                 'eval_metric': param_dict[task_type]['eval_metric']}
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
# eval_metric = 'validation_' + param_dict['eval_metric']
# best_model = results.sort_values(by=eval_metric, axis=0)


# %%
