# %%
%load_ext autoreload
%autoreload 2
from_commandline = False

import argparse
import sys
import os
import pandas as pd
from copy import deepcopy

from utils import get_labels
from cross_validation.cross_validation import Regression, BinaryClassification, MulticlassClassification 

if from_commandline:
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='eng')
    parser.add_argument('--task_type', default='regression')
    parser.add_argument('--testing', default='False')
    args = parser.parse_args()
    language = args.language
else:
    # Don't use defaults because VSC interactive can't handle command line arguments
    language = 'eng' #'eng', 'ger'
    task_type = 'regression' #'regression', 'binary', 'library', 'multiclass'
    testing = 'True' #True, False

features_dir = f'../data/features_jcls_conference/{language}/' ###########################
results_dir = f'../data/results/{language}/'
sentiscores_dir = '../data/sentiscores/'
metadata_dir = '../data/metadata/'
canonscores_dir = '../data/canonscores/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# %%
'''
Get data
'''
canon_labels = get_labels('canon', language, canonscores_dir, None, None)
textblob_labels = get_labels('textblob', language, None, sentiscores_dir, metadata_dir)
sentiart_labels = get_labels('sentiart', language, None, sentiscores_dir, metadata_dir)
combined_labels = get_labels('combined', language, None, sentiscores_dir, metadata_dir)
binary_labels = get_labels('binary', language, None, sentiscores_dir, metadata_dir)
multiclass_labels = get_labels('multiclass', language, None, sentiscores_dir, metadata_dir)
library_labels = get_labels('library', language, None, sentiscores_dir, metadata_dir)
#binary_labels['y'].plot.hist(grid=True, bins=50)
           
book_df = pd.read_csv(f'{features_dir}book_df.csv').rename(columns = {'book_name': 'file_name'})
book_and_averaged_chunk_df = pd.read_csv(f'{features_dir}book_and_averaged_chunk_df.csv').rename(columns = {'book_name': 'file_name'})
chunk_df = pd.read_csv(f'{features_dir}chunk_df.csv').rename(columns = {'book_name': 'file_name'})
chunk_and_copied_book_df = pd.read_csv(f'{features_dir}chunk_and_copied_book_df.csv').rename(columns = {'book_name': 'file_name'})

# %%

# All paramters
'''
models: 'svr', 'lasso', 'xgboost', 'svc'
dimensionality_reduction: 'ss_pca_0_95', 'k_best_f_reg_0_10', 'k_best_mutual_info_0_10', 'rfe', None
labels: 'canon', textblob', 'sentiart', 'combined', 'binary', 'multiclass', 'library'
'''
features_list = ['book', 'chunk', 'baac', 'cacb']
features_mapping = {'book': book_df, 'chunk': chunk_df, 'baac': book_and_averaged_chunk_df, 
                 'cacb': chunk_and_copied_book_df}
labels_mapping = {'canon': canon_labels, 'textblob': textblob_labels, 'sentiart': sentiart_labels, 'combined': combined_labels, 
          'binary': binary_labels, 'multiclass': multiclass_labels, 'library': library_labels}
drop_columns_list = [
    ['average_sentence_embedding', 'doc2vec_chunk_embedding'],
    ['average_sentence_embedding', 'doc2vec_chunk_embedding', 'pos']]
if language == 'eng':
    drop_columns_list.extend([
        ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->'], 
        ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->', 'pos']])


# %%
'''
Link parameters to models
'''
regression_dict = {
    'model': ['xgboost'], 
    'dimensionality_reduction': [None], 
    'features': features_list,
    'labels': ['textblob', 'sentiart', 'combined'],
    'drop_columns': drop_columns_list}
binary_dict = {
    'model': ['svc', 'xgboost'], 
    'dimensionality_reduction': [None], 
    'features': ['book', 'baac'],
    'labels': ['binary'],
    'drop_columns': drop_columns_list}
library_dict = deepcopy(binary_dict)
library_dict['labels'] = ['library']
multiclass_dict = {
    'model': ['svc', 'xgboost'], 
    'dimensionality_reduction': [None],
    'features': ['book', 'baac'],
    'labels': ['multiclass'],
    'drop_columns': drop_columns_list}

# Test CV with only one paramter combination
testing_reg_dict = {
    'model': ['svr'], 
    'dimensionality_reduction': [None], 
    'features': ['book'],
    'labels': ['sentiart'],
    'drop_columns': [drop_columns_list[-1]]}
testing_binary_dict = {
    'model': ['svc'], #xgboost
    'dimensionality_reduction': [None], 
    'features': ['book'], #'baac' 
    'labels': ['binary'],
    'drop_columns': [drop_columns_list[-1]]}
testing_library_dict = deepcopy(testing_binary_dict)
testing_library_dict['labels'] = ['library']
testing_multiclass_dict = {
    'model': ['svc'], #xgboost
    'dimensionality_reduction': [None], 
    'features': ['book'], #'baac'
    'labels': ['multiclass'],
    'drop_columns': [drop_columns_list[-1]]}

# %%
'''
Run Cross-Validation
'''  
if task_type == 'regression':
    param_dict = regression_dict
elif task_type == 'library':
    param_dict = library_dict
elif task_type == 'binary':
    param_dict = binary_dict
elif task_type == 'multiclass':
    param_dict = multiclass_dict

# Overwrite for testing 
if testing == 'True':
    if task_type == 'regression':
        param_dict = testing_reg_dict
    if task_type == 'binary':
        param_dict = testing_binary_dict
    elif task_type == 'multiclass':
        param_dict = testing_multiclass_dict
    elif task_type == 'library':
        param_dict = testing_library_dict

# %%
results = pd.DataFrame()
for model in param_dict['model']:
    for labels_name in param_dict['labels']:
        labels = deepcopy(labels_mapping[labels_name])
        for features_name in param_dict['features']:
            df = deepcopy(features_mapping[features_name])
            for dimensionality_reduction in param_dict['dimensionality_reduction']:
                for drop_columns in param_dict['drop_columns']:
                    
                    cv_info = {'language': language,
                            'task_type': task_type,
                            'model': model,
                            'features_name': features_name,
                            'labels_name': labels_name,
                            'dimensionality_reduction': dimensionality_reduction,
                            'drop_columns': drop_columns}

                    #info_string = f'{cv_params['language']}_{cv_params['task_type']}_{cv_params['model']}_label-{cv_params['labels_name']}_feat-{self.features_name}_dimred-{self.dimensionality_reduction}_drop-{len(self.drop_columns)}'
                    cv_params = {**cv_info, 
                                'df': df, 
                                'labels': labels, 
                                'results_dir': results_dir,
                                'verbose': True}

                    cv_info['drop_columns'] = len(cv_info['drop_columns'])
                    info_string = f'{"_".join(str(value) for value in cv_info.values())}'
                    cv_params['info_string'] = info_string

                    if task_type == 'regression':
                        cv_params['eval_metric'] = 'corr'
                        experiment = Regression(**cv_params)
                    elif (task_type == 'binary') or (task_type == 'library'):
                        cv_params['eval_metric'] = 'balanced_acc'
                        experiment = BinaryClassification(**cv_params)
                    elif task_type == 'multiclass':
                        cv_params['eval_metric'] = 'f1'
                        experiment = MulticlassClassification(**cv_params)

                    returned_values = experiment.run()
                    result = pd.concat(objs=[pd.DataFrame([cv_info])]+returned_values, axis=1)
                    
                    with open(f'{results_dir}results-{language}-{task_type}-log.csv', 'a') as f:
                        result.to_csv(f, header=False)
                    results = pd.concat([results, result], axis=0)
                    print(results)

                    print(language, task_type, model, labels_name, features_name,
                            dimensionality_reduction, drop_columns, returned_values)
                    print('\n-----------------------------------------------------------\n')


results_df = pd.DataFrame(results)
results_df.to_csv(f'{results_dir}results-{language}-{task_type}-final.csv', index=False, sep='\t')
# %%
