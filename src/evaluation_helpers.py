from tabulate import tabulate
import os
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
import pandas as pd
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from hpo_helpers import get_data


def get_best_model_across_features(task, best_inner_models, eval_metric_col, n_outer_folds, significance_threshold):
    # Find feature level which has the highest mean inner cv score across folds of outer cv
    mean_score_innercv = {}
    for features, group in best_inner_models.groupby('features'):
        # Only keep one model per inner fold since all rows belonging to the same fold have the same mean inner cv score (they are all best models)
        group = group.drop_duplicates('fold')
        mean_metric = group[eval_metric_col].mean()
        mean_score_innercv[features] = mean_metric
    print('mean_score_innercv', mean_score_innercv)

    best_features = max(mean_score_innercv, key=mean_score_innercv.get)
    best_model_across_features = best_inner_models.loc[best_inner_models['features'] == best_features]
    # Store mean inner cv score in column
    col_name = f'mean_{eval_metric_col}'
    best_model_across_features = best_model_across_features.copy(deep=True) # Make copy to avoid chained assignment warning
    best_model_across_features[col_name] = mean_score_innercv[best_features]

    # Check if there are multiple best models
    if not best_model_across_features.shape[0] == n_outer_folds:
        print('At least one model among the best models is not the sinlge best model. \
                Nr best models from {n_outer_folds} folds: {best_model_across_features.shape[0]}.')

    # Check if best model has significant harmonic p-value
    if task == 'regression':
        nonsignificant = best_model_across_features.loc[best_model_across_features['harmonic_pvalue'] >= significance_threshold]
        if nonsignificant.shape[0] != 0:
            print(f'{nonsignificant.shape[0]} best inner models have a non-significant harmonic p-value. \
                This means that no model had a significant harmonic pvalue and model with smalles pvalue was returned instead.')

    return best_model_across_features, best_features


def load_gridsearch_object(gridsearch_dir, language, task, label_type, features):
    with open(os.path.join(gridsearch_dir, f'gridsearch-object_{language}_{task}_{label_type}_{features}.pkl'), 'rb') as f:
        gridsearch_object = pickle.load(f)
    return gridsearch_object

def get_unlabeled_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir):
    X_labeled, y_labeled = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
    X = pd.read_csv(os.path.join(features_dir, f'{features}_features.csv')).set_index('file_name', drop=True)
    X = X.loc[X.index not in X_labeled.index]
    print(f'X labeled {X_labeled.shape}, X unlabeled {X.shape}')
    return X


def evaluate_classification(results_dir, language, task, label_type, best_features, y_true, y_pred):
    crosstab = pd.crosstab(index=y_true, columns=y_pred, rownames=['True'], colnames=['Predicted'], margins=True) # .values.ravel()
    crosstab.to_csv(os.path.join(results_dir, f'crosstab_{language}_{task}_{label_type}_{best_features}.csv'), index=True)
    print('crosstab', crosstab)
    crosstab_latex = tabulate(crosstab, tablefmt='latex_booktabs')
    with open(os.path.join(results_dir, f'crosstab-latex_{language}_{task}_{label_type}_{best_features}.csv'), 'w') as f:
        f.write(crosstab_latex)
    report = classification_report(y_true.values.ravel(), y_pred.values.ravel())
    with open(os.path.join(results_dir, f'classification-report{language}_{task}_{label_type}_{best_features}.csv'), 'w') as f:
        f.write(report)

def score_task(task, y_true, y_pred):
    if task == 'regression':
        corr, corr_pvalue = pearsonr(y_true, y_pred)
        return {'corr': corr, 
                'corr_pvalue': corr_pvalue}
    elif (task == 'binary') or (task == 'library'):
        return {'balanced_acc': balanced_accuracy_score(y_true, y_pred)}
    else:
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        return {'f1_macro': f1_macro,
                'f1_weighted': f1_weighted}
                

def get_best_models(cv_results, task, significance_threshold, eval_metric_col):
    # Return models with highest evaluation metric

    if task == 'regression':

        model_smallest_pval = cv_results[cv_results['harmonic_pvalue'] == cv_results['harmonic_pvalue'].min()] # Return if no model has significant harmonic pvalue
        # Adapted from to refit_regression()
        cv_results = cv_results[~cv_results['harmonic_pvalue'].isna()]
        cv_results = cv_results[cv_results['harmonic_pvalue']<significance_threshold]

        # Check if there is any model with significant correlation coefficient
        if cv_results.shape[0] == 0:
            print(f'No model has a significant hamonic p-value. Model with the smalles p-value is returned.')
            return model_smallest_pval

    # Check how many models have the highest correlation coefficent
    max_metric = cv_results[eval_metric_col].max()

    # Find index of maximum correlation
    best_models = cv_results.loc[cv_results[eval_metric_col] == max_metric]
    if best_models.shape[0] > 1:
        print('Number of models that have the highest correlation coefficient and significant p-value: ', best_models.shape[0])

    return best_models

def load_outer_scores(gridsearch_dir, language, task, label_type, features, outer_fold):
    print('load outer scorespath', os.path.join(gridsearch_dir, f'y-pred_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'))
    y_pred = pd.read_csv(
        os.path.join(gridsearch_dir, f'y-pred_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
        header=0)
    y_pred = y_pred.rename(columns={'fold': 'fold_pred'})

    y_true = pd.read_csv(
        os.path.join(gridsearch_dir, f'y-true_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
        header=0)
    y_true = y_true.rename(columns={'fold': 'fold_true'})
    df = pd.concat([y_pred, y_true], axis=1)
    return df