# %%
%load_ext autoreload
%autoreload 2

import warnings
warnings.simplefilter('once', RuntimeWarning)
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from tabulate import tabulate
from decimal import Decimal
import os
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from models_helpers import get_data, CustomGroupKFold, get_model, get_author_groups, score_regression, score_binary, score_multiclass


languages = ['eng'] #'eng', 'ger'
tasks = ['regression', 'binary', 'library', 'multiclass']
testing = True
data_dir = '/home/annina/scripts/great_unread_nlp/data'



pvalues = []
def get_harmonic_pvalue(pvalues):
    denominator = sum([Decimal(1)/Decimal(x) for x in pvalues])
    harmonic_pval = len(pvalues)/denominator
    return harmonic_pval

def get_correlation(estimator, X, y):
    y_pred = estimator.predict(X)
    corr, corr_pvalue = pearsonr(y, y_pred)  
    pvalues.append(corr_pvalue)
    return corr

# %%
# Predict with best estimator and analyze results
n_iter = 500
for language in languages:

    # Use full features set for classification to avoid error with underrepresented classes
    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
    gridsearch_dir = os.path.join(data_dir, 'gridsearch', language)
    results_dir = os.path.join(data_dir, 'results', language)
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores', language)
    if not os.path.exists(gridsearch_dir):
        os.makedirs(results_dir, exist_ok=True)

    for task in tasks:
        print(f'Task type: {task}')
        model = get_model(task, testing=True) #############3
        for label_type in model['labels']:
            X, y = get_data(task, language, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
            kfold_cv = CustomGroupKFold(n_splits=5, stratified=model['stratified']).split(X, y)
            
            with open(os.path.join(gridsearch_dir, f'randsearch-object_{language}_{task}_{label_type}_niter-{n_iter}.pkl'), 'rb') as f:
                randsearch = pickle.load(f)
                best_estimator = randsearch.best_estimator_
                y_pred = best_estimator.predict(X)

                mask = best_estimator.named_steps
                print(mask)

                # mask = best_estimator.named_steps['clf'].support_
                # print(mask)
                validate_cv = False
                if validate_cv:
                # Validate grid search cv for best estimator
                    if task == 'regression':
                        scoring = get_correlation
                    elif (task == 'binary') or (task == 'library'):
                        scoring = 'balanced_accuracy'
                    else:
                        scoring = 'f1_macro'
                    scores = cross_val_score(
                        estimator=deepcopy(randsearch.best_estimator_), 
                        X=X,
                        y=y,
                        scoring=scoring,
                        cv=kfold_cv,
                        n_jobs=-1,
                        verbose=3,
                    )
                    print('Cross validation score: ', np.mean(scores))
                    if task == 'regression':
                        print(get_harmonic_pvalue(pvalues))

                # Final result
                author_groups = get_author_groups(X)
                group_results = []
                for train_index, test_index in LeaveOneGroupOut().split(X, y, author_groups):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    group_test = author_groups.iloc[test_index]
                    estimator = deepcopy(randsearch.best_estimator_).fit(X_train, y_train)
                    y_group = deepcopy(y_test)
                    y_group['y_pred'] = estimator.predict(X_test)
                    scores = model['scoring'](estimator, X_test, y_test.values.ravel())
                    for metric, value in scores.items():
                        y_group[metric] = value
                    y_group['group'] = group_test
                    group_results.append(y_group)
                    print(y_group)
                group_results = pd.concat(group_results)
                group_results.to_csv(os.path.join(results_dir, f'LeaveOneGroupOut_{language}_{task}_{label_type}_niter-{n_iter}.csv'), index=True)

            if task != 'regression':
                crosstab = pd.crosstab(index=group_results.y, columns=group_results.y_pred, rownames=['True'], colnames=['Predicted'], margins=True) # .values.ravel()
                crosstab.to_csv(os.path.join(results_dir, f'crosstab_{language}_{task}_{label_type}_niter-{n_iter}.csv'), index=True)


                   

# %%
