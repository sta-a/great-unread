# %%
%load_ext autoreload
%autoreload 2
from_commandline = False

import argparse
import os
import pandas as pd
from copy import deepcopy
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from models_helpers import get_data, permute_params, ColumnTransformer, CustomGroupKFold, get_pvalue, get_model, average_chunk_features, CustomPCA

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
features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
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

    for task_type in ['binary']: #['regression', 'binary', 'library', 'multiclass']:
        print('Task type: ', task_type)
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
                    [ColumnTransformer(columns_to_drop=['_chunk'])] + [FunctionTransformer(average_chunk_features)] + model['param_data_subset'], # Add model-specific feature levels
                'drop_columns__columns_to_drop': 
                    columns_list,
                'dimred': # Pass transformers as parameters because of 'passthrough' option
                    ['passthrough'] + 
                    permute_params(SelectPercentile, percentile=list(range(10, 100, 10)), score_func=[f_regression, mutual_info_regression]) +  # try between 10 and 100 % of features
                    permute_params(CustomPCA, n_components=[None] + list(np.arange(0.81, 1.01, 0.02).tolist())) # try None and explaining between 81 % to 99 % of the variance
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
                n_jobs=1, ############################################################
                refit=model['refit'],
                cv=cv,
                verbose=1,
                error_score='raise', #np.nan
                return_train_score=False
                #pre_dispatch= 
            )
            gs.fit(X, y.values.ravel()) # Don't pass file_name column, dtype error
            print('Best estimator: ', gs.best_estimator_)
            print('Best params: ' , gs.best_params_)
            cv_results = pd.DataFrame(gs.cv_results_)

            if task_type == 'regression':
                cv_results['harmonic_pvalue'] = cv_results.apply(get_pvalue, axis=1)
            else:
                print(f'Best CV score={gs.best_score_}')

            cv_results.to_csv(os.path.join(results_dir, f'cv-results_{language}_{task_type}_{label_type}.csv'), index=False)
            with open(os.path.join(results_dir, f'gs-object_{language}_{task_type}_{label_type}.pkl'), 'wb') as f:
                pickle.dump(gs, f, -1)


# %%
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from tabulate import tabulate
from models_helpers import get_data, get_model
import os
import pandas as pd
import pickle

language = 'eng' #'eng', 'ger'
testing = True #True, False
data_dir = '/home/annina/scripts/great_unread_nlp/data'
# Use full features set for classification to avoid error with underrepresented classes
features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
results_dir = os.path.join(data_dir, 'results', language)
sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
metadata_dir = os.path.join(data_dir, 'metadata', language)
canonscores_dir = os.path.join(data_dir, 'canonscores', language)


# Predict with best estimator and analyze results
for language in ['eng']: # ['eng', 'ger']:
    for task_type in ['regression', 'binary']: #['regression', 'binary', 'library', 'multiclass']:
        print(f'Task type: {task_type}')
        model = get_model(task_type, testing=True) #############3
        for label_type in model['labels']:
            X, y = get_data(task_type, language, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
            with open(os.path.join(results_dir, f'gs-object_{language}_{task_type}_{label_type}.pkl'), 'rb') as f:
                gs = pickle.load(f)
                estimator = gs.best_estimator_
                y_pred = estimator.predict(X)
                if task_type == 'regression':
                    corr, corr_pvalue = pearsonr(y.values.ravel(), y_pred)
                    print(f'Correlation coefficent: {corr}, p-value: {corr_pvalue}')
                else:
                    if (task_type=='binary') or (task_type=='library'):
                        score = balanced_accuracy_score(y, y_pred)
                        print(f'Balanced accuracy: {score}')
                    else:
                        score = f1_score(y, y_pred, average='macro')
                        print(f'F1 score: {score}')

                    crosstab = pd.crosstab(index=y.values.ravel(), columns=y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
                    crosstab.to_csv(os.path.join(results_dir, f'crosstab_{language}_{task_type}_{label_type}.csv'), index=True)
                    


# %%
