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
tasks = ['regression']# ['regression', '', 'library', 'multiclass']
testing = False ##############################3
data_dir = '/home/annina/scripts/great_unread_nlp/data'
significance_threshold = 0.1
n_outer_folds = 5


# Outer cv evaluation from single files
for language in languages:

    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
    gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language) ######################3
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
        for label_type in task_params['labels']:
            best_inner_models = []

            ## Find best model(s) for each feature level for each fold of the outer cv
            for features in task_params['features']:
                print(language, task, label_type, features)
                outer_scores = []
                for outer_fold in range(0, n_outer_folds):
                    y_pred = pd.read_csv(
                        os.path.join(gridsearch_dir, f'y-pred_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
                        header=0)
                    y_pred = y_pred.rename(columns={'fold': 'fold_pred'})

                    y_true = pd.read_csv(
                        os.path.join(gridsearch_dir, f'y-true_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
                        header=0)
                    y_true = y_true.rename(columns={'fold': 'fold_true'})
                    df = pd.concat([y_pred, y_true], axis=1)
                    outer_scores.append(df)
                df = pd.concat(outer_scores)
                df = df[['file_name', 'fold_true', 'fold_pred', 'y_true', 'y_pred']]
                print(pearsonr(df['y_true'], df['y_pred']))
                df.to_csv(
                    os.path.join(results_dir, 
                    f'outer-cv-predicted_{language}_{task}_{label_type}_{features}.csv'), 
                    header=True,
                    index=False)

# %%
# Find best model for each task
# For regression, each label type is a separate task
for language in languages:

    features_dir = os.path.join(data_dir, 'features_None', language)
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
            best_inner_models = []

            ## Find best model(s) for each feature level for each fold of the outer cv
            for features in task_params['features']:
                print('\n##############################\n', language, task, label_type, features)
                best_models = []
                for outer_fold in range(0, n_outer_folds):
                    print('Fold ', outer_fold)
                    fold_results = pd.read_csv(
                        os.path.join(gridsearch_dir, f'inner-cv_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
                        header=0)
                    # Check if there are NaN harmonic p-values
                    if task == 'regression':
                        na_pvalue = fold_results[fold_results['harmonic_pvalue'].isnull()]
                        if na_pvalue.shape[0] != 0:
                            print(f'{na_pvalue.shape[0]} models have non-significant harmonic p-value.')
                            na_pvalue.to_csv(
                                os.path.join(results_dir, f'napvalue_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
                                index=False, 
                                header=True, 
                                na_rep='NaN')
                    best_model = get_best_models(fold_results, task, significance_threshold, eval_metric_col)
                    best_models.append(best_model)
                best_models = pd.concat(best_models, axis=0)
                best_inner_models.append(best_models)
            best_inner_models = pd.concat(best_inner_models)
            best_inner_models.to_csv(
                os.path.join(results_dir, f'best-inner-models-per-feature_{language}_{task}_{label_type}.csv'), 
                index=False, 
                header=True, 
                na_rep='NaN')

            ## Keep only feature level with highest mean inner cv score across folds of outer cv          
            best_model_across_features, best_features = get_best_model_across_features(
                task, 
                best_inner_models, 
                eval_metric_col, 
                n_outer_folds, 
                significance_threshold)
            print('best features', best_features)
            best_model_across_features.to_csv(
                os.path.join(results_dir, f'best-inner-model_{language}_{task}_{label_type}.csv'), 
                index=False, 
                header=True, 
                na_rep='NaN')

            # Score outer cv
            outer_cv_result = pd.read_csv(
                os.path.join(results_dir, 
                f'outer-cv-predicted_{language}_{task}_{label_type}_{best_features}.csv'), 
                header=0)
            
            y_true = outer_cv_result['y_true']
            y_pred = outer_cv_result['y_pred']
            outer_scores = score_task(task, y_true, y_pred)
            outer_results_dict = {
                'language': language, 
                'task': task, 
                'label_type': label_type, 
                'best_features': best_features}
            outer_results_dict.update(outer_scores)
            outer_results.append(pd.DataFrame.from_dict([outer_results_dict]))
            if task != 'regression':
                evaluate_classification(results_dir, language, task, label_type, best_features, y_true, y_pred)
    outer_results = pd.concat(outer_results)
    outer_results.to_csv(
        os.path.join(results_dir, f'outer-scores_{language}.csv'), 
        header=True, 
        index=False)

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
                cv_result = pd.read_csv(os.path.join(gridsearch_dir, f'inner-cv_{language}_{task}_{label_type}_{features}_fold-fulldata.csv'), header=0)
                cv_results.append(cv_result)
        cv_results = pd.concat(cv_results)
        best_model = get_best_models(cv_results)
        gs_object = load_gridsearch_object(gridsearch_dir, language, task, label_type, features)


        X_unlabeled = get_unlabeled_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
# %%
