# %%
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import pickle
import os
from hpo_functions import get_data, ColumnTransformer


languages = ['eng', 'ger'] #'eng', 'ger'
task = 'regression-importances'
label_type = 'canon'
# Importances are calculated on cacb features
features = 'cacb'

data_dir = '../data'
n_outer_folds = 5

#for language in languages: ###############################
language = 'eng'
print('-----------------------------', language)

features_dir = os.path.join(data_dir, 'features_None', language)
gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
metadata_dir = os.path.join(data_dir, 'metadata', language)
canonscores_dir = os.path.join(data_dir, 'canonscores')
importances_dir = os.path.join(data_dir, 'importances', language)
if not os.path.exists(importances_dir):
    os.makedirs(importances_dir, exist_ok=True)

importances_list = []
for outer_fold in range(0, n_outer_folds):

    gsobj_path = os.path.join(gridsearch_dir, f'gridsearch-object_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.pkl')
    with open(gsobj_path, 'rb') as f:
        gsobj = pickle.load(f)
    estimator = gsobj.best_estimator_
    importances = estimator.named_steps['clf'].feature_importances_
    importances_list.append(importances)
    print('Nr importance: ', importances.shape)

# Feature voting
# Get feature count after dropping columns
X, y = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
X_transformed = ColumnTransformer(columns_to_drop=estimator.named_steps['drop_columns'].columns_to_drop).fit_transform(X, None)
print(f'Nr features in full dataset: {X.shape[1]}. Nr features after dropping embeddings: {X_transformed.shape[1]}.')

df = pd.DataFrame(np.row_stack(importances_list), columns=X_transformed.columns)
# nonzeros = df.astype(bool).sum(axis=0)     # For how many fold is feature important
sum_imprt = df.sum(axis=0)
sum_imprt.sort_values().plot(figsize=(8, 10)) # Curve has knee between 0.02 and 0.04
# Filter columns so that only features with total importance above threshold are kept
df_002 = df.loc[:,sum_imprt>0.02] # Threshold = 0.02. eng: 73 features, ger: 72 features
df_004 = df.loc[:,sum_imprt>0.04] # Threshold = 0.04. eng: 24 features, ger: 22 features
# nonzeros_004 = df_004.astype(bool).sum(axis=0)

# Keep 30 best features
best_features_imprt = sum_imprt.nlargest(n=30, keep='all') # all features are 'fulltext

# Use book features
#features='book'
X, y = get_data(language, task, label_type, 'book', features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
best_features = X[best_features_imprt.index.to_list()]
best_features.to_csv(
    os.path.join(importances_dir,
    f'book_features_best.csv'),
    header=True,
    index=True)

best_features_imprt = best_features_imprt.reset_index().rename({'index': 'feature', 0: 'importance'}, axis=1)
print(best_features_imprt)
best_features_imprt.to_csv(
    os.path.join(importances_dir,
    f'best_features_importances.csv'),
    header=True,
    index=False)



# %%
