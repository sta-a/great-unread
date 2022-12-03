# %%
%load_ext autoreload
%autoreload 2

import warnings
warnings.simplefilter('once', RuntimeWarning)
import pandas as pd
import numpy as np
import pickle
import os
from hpo_functions import get_data, CustomGroupKFold, get_task_params, run_gridsearch


if from_commandline:
    parser = argparse.ArgumentParser()
    parser.add_argument('languages')
    parser.add_argument('data_dir')
    # If --testing flag is set, testing is set to bool(True). 
    # If --no-testing flag is set, testing is set to bool(False)
    parser.add_argument('--testing', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    languages = [args.languages]
    data_dir = args.data_dir
    testing = args.testing
else:
    # Don't use defaults because VSC interactive mode can't handle command line arguments
    languages = ['eng', 'ger']
    data_dir = '../data'
    testing = False
print(languages, data_dir, testing )
n_outer_folds = 5


languages = ['eng', 'ger'] #'eng', 'ger'
task = 'regression'
label_type = 'canon'
features = 'cacb'

data_dir = '../data'
n_outer_folds = 5

for language in languages:
    print('-----------------------------', language)
    for outer_fold in range(0, n_outer_folds):################3
    #for outer_fold in range(0, 1):

    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores')
    features_dir = os.path.join(data_dir, 'features_None', language)
    importances_dir = os.path.join(data_dir, 'importances', language)
    importances_gridsearch_dir = os.path.join(importances_dir, 'nested_gridsearch', language)
    if not os.path.exists(importances_gridsearch_dir):
        os.makedirs(importances_gridsearch_dir, exist_ok=True)

    run_gs_without_dropping()
        

def get_feature_importances():
    gsobj_path = os.path.join(importances_gridsearch_dir, f'gridsearch-object_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.pkl')
    with open(gsobj_path, 'rb') as f:
        gsobj = pickle.load(f)
        drop = gsobj.best_estimator_.named_steps['drop_columns']
        #estimator = gsobj.best_estimator_.named_steps['clf'].feature_importances_
        print(drop)
    # Get feature count after dropping columns
    # X, y = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)

                
# Drop columns
                


# %%
