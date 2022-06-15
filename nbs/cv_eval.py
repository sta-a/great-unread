# %%
%load_ext autoreload
%autoreload 2
# 'eng' , 'ger'
language = 'ger'
# 'regression', 'twoclass', 'library', 'multiclass'
task_type = 'library' 
# True, False
testing = 'True'

import sys
from copy import deepcopy
sys.path.insert(0, '../src/')
import os
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from utils import get_labels
from cross_validation import Regression, TwoclassClassification, MulticlassClassification

features_dir = f'../data/features/{language}/'
results_dir = f'../data/results_sentiment_testing/{language}/'
sentiscores_dir = '../data/sentiscores/'
#metadata_dir = '../data/metadata/'
#metadata_dir = '/home/annina/Downloads/JCLS2022_Modeling-and-Predicting-Lit-Reception/metadata/'
metadata_dir = '/home/annina/Downloads/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# %%
df = pd.read_csv(metadata_dir + 'GER_texts_circulating-libs.csv', sep=';')
df

# %%
'''
Get data
'''
textblob_labels = get_labels(language, sentiscores_dir, metadata_dir, 'textblob')
sentiart_labels = get_labels(language, sentiscores_dir, metadata_dir, 'sentiart')
combined_labels = get_labels(language, sentiscores_dir, metadata_dir, 'combined')
twoclass_labels = get_labels(language, sentiscores_dir, metadata_dir, 'twoclass')
multiclass_labels = get_labels(language, sentiscores_dir, metadata_dir, 'multiclass')

# %%
metadata_dir

# %%
twoclass_labels['y'].value_counts()

# %%
library_labels = get_labels(language, sentiscores_dir, metadata_dir, 'library')
#twoclass_labels['y'].plot.hist(grid=True, bins=50)
           
book_df = pd.read_csv(f'{features_dir}book_df.csv').rename(columns = {'book_name': 'file_name'})
book_and_averaged_chunk_df = pd.read_csv(f'{features_dir}book_and_averaged_chunk_df.csv').rename(columns = {'book_name': 'file_name'})
chunk_df = pd.read_csv(f'{features_dir}chunk_df.csv').rename(columns = {'book_name': 'file_name'})
chunk_and_copied_book_df = pd.read_csv(f'{features_dir}chunk_and_copied_book_df.csv').rename(columns = {'book_name': 'file_name'})

# %%
library_labels

# %%
'''
All paramters

models: 'svr', 'lasso', 'xgboost', 'svc'
dimensionality_reduction: 'ss_pca_0_95', 'k_best_f_reg_0_10', 'k_best_mutual_info_0_10', None
labels: 'textblob', 'sentiart', 'combined', 'twoclass', 'multiclass', 'library'
'''
model_params_dict = {'svr': [1], 'lasso': [1, 4], 'xgboost': [None], 'svc': [0.1, 1, 10, 100, 1000, 10000]} 
features_list = ['book', 'chunk', 'baac', 'cacb']
features_dict = {'book': book_df, 'chunk': chunk_df, 'baac': book_and_averaged_chunk_df, 
                 'cacb': chunk_and_copied_book_df}
labels_dict = {'textblob': textblob_labels, 'sentiart': sentiart_labels, 'combined': combined_labels, 
          'twoclass': twoclass_labels, 'multiclass': multiclass_labels, 'library': library_labels}
drop_columns_list = [
    ['average_sentence_embedding', 'doc2vec_chunk_embedding'],
    ['average_sentence_embedding', 'doc2vec_chunk_embedding', 'pos']]
if language == 'eng':
    drop_columns_list.extend([
        ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->'], 
        ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->', 'pos']])
    
# Model-specific column names for writing results to file
general_cols = ['language', 'task_type', 'model', 'model_param', 'labels_string', 'features_string',
    'dimensionality_reduction', 'drop_columns']
regression_cols = general_cols + ['mean_train_mse', 'mean_train_rmse', 'mean_train_mae', 'mean_train_r2', 
    'mean_train_corr', 'mean_validation_mse', 'mean_validation_rmse', 'mean_validation_mae', 
    'mean_validation_r2', 'mean_validation_corr', 'mean_p_value']
twoclass_cols = general_cols + ['mean_train_acc', 'mean_train_balanced_acc', 'mean_validation_acc', 'mean_validation_balanced_acc'] # also used for library
multiclass_cols = general_cols + ['mean_train_f1', 'mean_validation_f1']

# %%
'''
Link parameters to models
'''
regression_dict = {
    'model': ['xgboost'], 
    'dimensionality_reduction': [None], 
    'features': features_list,
    'labels': ['textblob', 'sentiart', 'combined'],
    'drop_columns': drop_columns_list,
    'model_cols': regression_cols}
twoclass_dict = {
    'model': ['svc', 'xgboost'], 
    'dimensionality_reduction': [None], 
    'features': ['book', 'baac'],
    'labels': ['twoclass'],
    'drop_columns': drop_columns_list,
    'model_cols': twoclass_cols}
library_dict = deepcopy(twoclass_dict)
library_dict['labels'] = ['library']
multiclass_dict = {
    'model': ['svc', 'xgboost'], 
    'dimensionality_reduction': [None],
    'features': ['book', 'baac'],
    'labels': ['multiclass'],
    'drop_columns': drop_columns_list,
    'model_cols': multiclass_cols}

# Test CV with only one paramter combination
testing_reg_dict = {
    'model': ['xgboost'], 
    'dimensionality_reduction': [None], 
    'features': ['book'],
    'labels': ['combined'],
    'drop_columns': [drop_columns_list[-1]],
    'model_cols': regression_cols}
testing_twoclass_dict = {
    'model': ['xgboost', 'svc'], #xgboost
    'dimensionality_reduction': [None], 
    'features': ['book'], #'baac'
    'labels': ['twoclass'],
    'drop_columns': [drop_columns_list[-1]],
    'model_cols': twoclass_cols}
testing_library_dict = deepcopy(testing_twoclass_dict)
testing_library_dict['labels'] = ['library']
testing_multiclass_dict = {
    'model': ['xgboost', 'svc'], #xgboost
    'dimensionality_reduction': [None], 
    'features': ['book'], #'baac'
    'labels': ['multiclass'],
    'drop_columns': [drop_columns_list[-1]],
    'model_cols': multiclass_cols}
testing_twoclass_dict = { # best results balanced acc
    'model': ['svc'], #xgboost
    'dimensionality_reduction': [None], 
    'features': ['book'], #'baac'
    'labels': ['twoclass'],
    'drop_columns': [drop_columns_list[0]],
    'model_cols': twoclass_cols}

# %%
'''
Run Cross-Validation
'''  
if task_type == 'regression':
    param_dict = regression_dict
elif task_type == 'library':
    param_dict = library_dict
elif task_type == 'twoclass':
    param_dict = twoclass_dict
elif task_type == 'multiclass':
    param_dict = multiclass_dict

##Overwrite for testing 
if testing == 'True':
    if task_type == 'regression':
        param_dict = testing_reg_dict
    if task_type == 'twoclass':
        param_dict = testing_twoclass_dict
    elif task_type == 'multiclass':
        param_dict = testing_multiclass_dict
    elif task_type == 'library':
        param_dict = testing_library_dict

# %%
results = []
with open(f'{results_dir}results-{language}-{task_type}-log.csv', 'a') as f:
    f.write('\t'.join(param_dict['model_cols']) + '\n')
for model in param_dict['model']:
    model_param = model_params_dict[model]
    for model_param in [10]:
    #for model_param in model_param:############################################3
        for labels_string in param_dict['labels']:
            labels = deepcopy(labels_dict[labels_string])
            print(labels.value_counts())
            for features_string in param_dict['features']:
                df = deepcopy(features_dict[features_string])
                for dimensionality_reduction in param_dict['dimensionality_reduction']:
                    for drop_columns in param_dict['drop_columns']:
                        if task_type == 'regression':
                            experiment = Regression(
                                results_dir=results_dir,
                                language=language,
                                task_type=task_type,
                                model=model,
                                model_param=model_param,
                                labels_string=labels_string,
                                labels=labels,
                                features_string=features_string,
                                df=df,
                                dimensionality_reduction=dimensionality_reduction,
                                drop_columns=drop_columns,
                                verbose=True)
                        elif (task_type == 'twoclass') or (task_type == 'library'):
                            experiment = TwoclassClassification(
                                results_dir=results_dir,
                                language=language,
                                task_type=task_type,
                                model=model,
                                model_param=model_param,
                                labels_string=labels_string,
                                labels=labels,
                                features_string=features_string,
                                df=df,
                                dimensionality_reduction=dimensionality_reduction,
                                drop_columns=drop_columns,
                                verbose=True)

                        elif task_type == 'multiclass':
                            experiment = MulticlassClassification(
                                results_dir=results_dir,
                                language=language,
                                task_type=task_type,
                                model=model,
                                model_param=model_param,
                                labels_string=labels_string,
                                labels=labels,
                                features_string=features_string,
                                df=df,
                                dimensionality_reduction=dimensionality_reduction,
                                drop_columns=drop_columns,
                                verbose=True)

                        returned_values = experiment.run()
                        all_columns = [language, task_type, model, model_param, labels_string, features_string,
                                       dimensionality_reduction, drop_columns] + returned_values
                        
                        with open(f'{results_dir}results-{language}-{task_type}-log.csv', 'a') as f:
                            f.write('\t'.join([str(x) for x in all_columns]) + '\n')
                            results.append(all_columns) 

                        print(language, task_type, model, model_param, labels_string, features_string,
                                dimensionality_reduction, drop_columns, returned_values)
                        print('\n-----------------------------------------------------------\n')


results_df = pd.DataFrame(results, columns=param_dict['model_cols'])
results_df.to_csv(f'{results_dir}results-{language}-{task_type}-final.csv', index=False, sep='\t')

# %% [markdown]
# Hoffmansthal_Hugo-von_Ein-Brief_1902
# Hoffmansthal_Hugo_Ein-Brief_1902

# %%


# %%


# %%


# %%


# %%



