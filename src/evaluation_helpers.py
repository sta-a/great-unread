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


def load_gridsearch_object(gridsearch_dir, language, task, label_type, features):
    with open(os.path.join(gridsearch_dir, f'gridsearch-object_{language}_{task}_{label_type}_{features}.pkl'), 'rb') as f:
        gridsearch_object = pickle.load(f)
    return gridsearch_object

def get_unlabeled_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
    X_labeled, y_labeled = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
    X = pd.read_csv(os.path.join(features_dir, f'{features}_features.csv')).set_index('file_name', drop=True)
    X = X.loc[X.index not in X_labeled.index]
    print(f'X labeled {X_labeled.shape}, X unlabeled {X.shape}')
    return X


def evaluate_classification(results_dir, language, task, label_type, best_features, y_true, y_pred):
    crosstab = pd.crosstab(index=y_true, columns=y_pred, rownames=['True'], colnames=['Predicted'], margins=True) # .values.ravel()
    crosstab.to_csv(os.path.join(results_dir, f'crosstab_{language}_{task}_{label_type}_{best_features}.csv'), index=True)
    print(crosstab)
    crosstab_latex = tabulate(crosstab, tablefmt='latex_booktabs')
    crosstab_latex.to_csv(os.path.join(results_dir, f'crosstab-latex_{language}_{task}_{label_type}_{best_features}.csv'), index=True)
    print(crosstab_latex)
    report = pd.DataFrame(classification_report(y_true, y_pred))
    report.to_csv(os.path.join(results_dir, f'classification-report{language}_{task}_{label_type}_{best_features}.csv'), index=True)
    print(report)


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
        model_smallest_pval = cv_results['harmonic_pvalue'].idxmin() # Return if no model has significant harmonic pvalue
        # Adapted from to refit_regression()
        cv_results = cv_results[~cv_results['harmonic_pvalue'].isna()]
        cv_results = cv_results[cv_results['harmonic_pvalue']<significance_threshold]

        # Check if there is any model with significant correlation coefficient
        if cv_results.shape[0] == 0:
            print(f'No model has a significant hamonic p-value. Model with the highest correlation coefficient is returned.')
            return model_smallest_pval

    # Check how many models have the highest correlation coefficent
    max_metric = cv_results[eval_metric_col].max()
    nr_max_metric = (cv_results[eval_metric_col].values == max_metric).sum()
    print('Number of models that have the highest correlation coefficient and significant p-value: ', nr_max_metric)

    # Find index of maximum correlation
    best_models = cv_results.loc[cv_results[eval_metric_col] == max_metric] 
    best_models.to_csv('bestmodelsfunction') #######################################################################################
    return best_models
