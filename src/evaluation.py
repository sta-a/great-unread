# %%
%load_ext autoreload
%autoreload 2

import warnings
warnings.simplefilter('once', RuntimeWarning)
import pandas as pd
import numpy as np
from hpo_helpers import get_task_params, refit_regression
from evaluation_helpers import *

languages = ['eng'] #'eng', 'ger'
tasks = ['regression']#, 'binary', 'library', 'multiclass']
testing = False
data_dir = '/home/annina/scripts/great_unread_nlp/data'
significance_threshold = 0.1


for language in languages:

    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
    gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores', language)
    results_dir = os.path.join(data_dir, 'results', language)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)


    for task in tasks:
        print('Task: ', task)
        outer_results = []
        task_params = get_task_params(task, testing)
        if task == 'regression':
            eval_metric_col = 'mean_test_corr'
        else:
            eval_metric_col = 'mean_test_' + task_params['refit']

        for label_type in task_params['labels']:
            all_best_folds_features = []

            # Predict labels of unlabeled texts
            X_unlabeled = get_unlabeled_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
            gs_object = load_gridsearch_object(gridsearch_dir, language, task, label_type, features)




            for features in task_params['features']:
                print(language, task, label_type, features)
                # Find best model(s) for each feature level and fold
                best_models = []
                for outer_fold in range(0, 5):
                    print(outer_fold)
                    fold_results = pd.read_csv(os.path.join(gridsearch_dir, f'inner-cv-result_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), header=0)
                    # Check if there are NaN harmonic p-values
                    if task == 'regression':
                        na_correlation_coefficient = fold_results[fold_results['harmonic_pvalue'].isnull()]
                        na_correlation_coefficient.to_csv(os.path.join(results_dir, f'napvalue_{language}_{task}_{label_type}.csv'), index=False, header=True, na_rep='NaN')
                    best_model = get_best_models(fold_results, task, significance_threshold, eval_metric_col)
                    best_models.append(best_model)
                best_models = pd.concat(best_models, axis=0)
                all_best_folds_features.append(best_models)
            all_best_folds_features = pd.concat(all_best_folds_features)
            all_best_folds_features.to_csv(os.path.join(results_dir, f'best-folds-per-features_{language}_{task}_{label_type}.csv'), index=False, header=True, na_rep='NaN')

            # Find feature level with highest amean inner cv score
            mean_metric_innercv = {}
            for features, group in all_best_folds_features.groupby('features'):
                # Only keep one row per fold since all rows belonging to the same fold have the same evaluation metric (they are all best models)
                group = group.drop_duplicates('fold')
                print(group.shape)
                mean_metric = group[eval_metric_col].mean()
                mean_metric_innercv[features] = mean_metric
            print(mean_metric_innercv)
            # Keep only feature level that has the highest mean inner cv score
            best_features = max(mean_metric_innercv, key=mean_metric_innercv.get)
            best_model_across_features = all_best_folds_features.loc[all_best_folds_features['features'] == best_features]
            col_name = f'mean_{eval_metric_col}'
            # Check if best model has significant harmonic p-value
            best_model_across_features[col_name] = mean_metric_innercv[best_features]
            if task == 'regression':
                nonsignificant = best_model_across_features.loc[best_model_across_features['harmonic_pvalue'] < significance_threshold]
                if nonsignificant.shape[0] != 0:
                    print('At least one p-value in best features is not significant. This means that no model had a significant harmonic pvalue and model with smalles pvalue was returned instead.')
            best_model_across_features.to_csv(os.path.join(results_dir, f'best-model_{language}_{task}_{label_type}.csv'), index=False, header=True, na_rep='NaN')

                
            # Score outer cv
            outer_cv_result = pd.read_csv(os.path.join(gridsearch_dir, f'outer-cv-predicted_{language}_{task}_{label_type}_{best_features}.csv'), header=0)
            y_true = outer_cv_result['y_true']
            y_pred = outer_cv_result['y_pred']
            outer_scores = score_task(task, y_true, y_pred)
            outer_results_dict = {'language': language, 'task': task, 'label_type': label_type, 'best_features': best_features}.update(outer_scores)
            outer_results.append(pd.DataFrame(outer_results_dict))
            if task != 'regression':
                evaluate_classification(results_dir, language, task, label_type, best_features, y_true, y_pred)
    outer_results = pd.concat(outer_results)
    outer_results.to_csv(os.path.join(results_dir, f'outer-scores_{language}.csv'), header=0)

# %%
# Predict labels of unlabeled texts
for language in languages:
    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
    gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores', language)
    results_dir = os.path.join(data_dir, 'results', language)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    for task in tasks:
        print('Task: ', task)
        task_params = get_task_params(task, testing)
        if task == 'regression':
            eval_metric_col = 'mean_test_corr'
        else:
            eval_metric_col = 'mean_test_' + task_params['refit']

        cv_results = []
        for label_type in task_params['labels']:
            for features in task_params['features']:
                cv_result = pd.read_csv(os.path.join(gridsearch_dir, f'inner-cv-result_{language}_{task}_{label_type}_{features}_fold-fulldata.csv'), header=0)
                cv_results.append(cv_result)
        cv_results = pd.concat(cv_results)
        best_model = get_best_models(cv_results)
        gs_object = load_gridsearch_object(gridsearch_dir, language, task, label_type, features)


        X_unlabeled = get_unlabeled_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)