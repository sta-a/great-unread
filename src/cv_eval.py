# %%
%load_ext autoreload
%autoreload 2
from_commandline = False

import argparse
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
    testing = 'True' #True, False
    data_dir = '../data/'

features_dir = features_dir = os.path.join(data_dir, 'features_jcls_conference', language) ###################
results_dir = os.path.join(data_dir, 'results', language)
sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
metadata_dir = os.path.join(data_dir, 'metadata', language)
canonscores_dir = os.path.join(data_dir, 'canonscores', language)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

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
book_and_averaged_chunk_df = pd.read_csv(f'{features_dir}book_and_averaged_chunk_df.csv').rename(columns = {'book_name': 'file_name'}) #€#################################################3
chunk_df = pd.read_csv(f'{features_dir}chunk_df.csv').rename(columns = {'book_name': 'file_name'})
chunk_and_copied_book_df = pd.read_csv(f'{features_dir}chunk_and_copied_book_df.csv').rename(columns = {'book_name': 'file_name'})


# All paramters
'''
models: 'svr', 'lasso', 'xgboost', 'svc'
dimensionality_reduction: 'ss_pca_0_95', 'k_best_f_reg_0_10', 'k_best_mutual_info_0_10', 'rfe', None
'''
features_mapping = {
    'book': book_df,
    'chunk': chunk_df,
    'baac': book_and_averaged_chunk_df,
    'cacb': chunk_and_copied_book_df}
labels_mapping = {
    'canon': canon_labels,
    'textblob': textblob_labels,
    'sentiart': sentiart_labels,
    'combined': combined_labels,
    'binary': binary_labels,
    'multiclass': multiclass_labels,
    'library': library_labels}
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
if testing == 'True':
    for task_type, d in param_dict.items():
        d[task_type]['drop_columns'] = [drop_columns_list[-1]]

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
results = pd.DataFrame()
for model in param_dict[task_type]['model']:
    for features_name in param_dict[task_type]['features']:
        df = deepcopy(features_mapping[features_name])
        for labels_name in param_dict[task_type]['labels']:
            labels = deepcopy(labels_mapping[labels_name])
            for dimensionality_reduction in param_dict[task_type]['dimensionality_reduction']:
                for drop_columns in param_dict[task_type]['drop_columns']:
                    print(task_type, model, labels_name, features_name, dimensionality_reduction, len(drop_columns))
                    
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
                                'verbose': True,
                                'eval_metric': param_dict[task_type]['eval_metric']}
                    cv_info['drop_columns'] = len(cv_info['drop_columns'])
                    info_string = f'{"_".join(str(value) for value in cv_info.values())}'
                    cv_params['info_string'] = info_string

                    if task_type == 'regression':
                        experiment = Regression(**cv_params)
                    elif (task_type == 'binary') or (task_type == 'library'):
                        experiment = BinaryClassification(**cv_params)
                    elif task_type == 'multiclass':
                        experiment = MulticlassClassification(**cv_params)

                    returned_values = experiment.run()
                    result = pd.concat(objs=[pd.DataFrame([cv_info])]+returned_values, axis=1)
                    
                    with open(f'{results_dir}results-{language}-{task_type}-log.csv', 'a') as f:
                        result.to_csv(f, header=False)
                    results = pd.concat([results, result], axis=0)
                    print('\n-----------------------------------------------------------\n')

results_path = f'{results_dir}results-{language}-{task_type}-final.csv'
results.to_csv(results_path, index=False, sep='\t')


# %%
results = pd.read_csv(results_path, sep='\t')
results
# %%
if task_type == 'regression':
    results = results[results['validation_corr_pvalue']<=0.1]
eval_metric = 'validation_' + param_dict['eval_metric']
best_model = results.sort_values(by=eval_metric, axis=0)


# %%
