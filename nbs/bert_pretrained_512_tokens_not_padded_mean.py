# %%
%load_ext autoreload
%autoreload 2

import os
import sys
sys.path.insert(0, '../src/')
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from process import BertProcessor
from utils import get_doc_paths, get_pickle_paths, read_labels

raw_docs_dir = '../data/raw_docs/'
labels_dir = '../data/labels/'


# %% [markdown]
# # Process Data

# %%
lang = 'eng'
raw_doc_paths = get_doc_paths(raw_docs_dir, lang)
bp = BertProcessor(lang=lang, pad=False)
bp.process(raw_doc_paths)

# %% [markdown]
# # Create Bert Document Vectors

# %%
lang = 'eng'
bv = BertVectorizer(lang=lang, sentence_to_doc_agg='mean')
pickle_paths = get_pickle_paths('../data/processed_bert_512_tokens_not_padded/', lang)
bv.fit(pickle_paths)
doc_vectors = bv.get_doc_vectors()

# %%
df = doc_vectors.copy()
labels = read_labels(lang)
df['y'] = df['pickle_path'].apply(lambda x: labels[x.split('/')[-1][:-7]])
df['file_name'] = df['pickle_path'].apply(lambda x: x.split('/')[-1][:-7])

# %% [markdown]
# # Cross Validation

# %%
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

X = df.drop(columns=['y', 'file_name', 'pickle_path']).values
y = df['y'].values.ravel()

all_predictions = []
all_labels = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for index, (train_indices, validation_indices) in enumerate(kf.split(X)):
    train_X = X[train_indices, :]
    train_y = y[train_indices]
    validation_X = X[validation_indices, :]
    validation_y = y[validation_indices]
    pca = PCA(n_components=30)
    train_X = pca.fit_transform(train_X)
    print(pca.explained_variance_ratio_.sum())
    validation_X = pca.transform(validation_X)
    model = SVR(C=40)
    model.fit(train_X, train_y)
    train_yhat = model.predict(train_X)
    validation_yhat = model.predict(validation_X) # np.array([train_yhat.mean()] * validation_X.shape[0]) # 
    all_labels.extend(validation_y.tolist())
    all_predictions.extend(validation_yhat.tolist())
    train_mse = mean_squared_error(train_y, train_yhat)
    train_mae = mean_absolute_error(train_y, train_yhat)
    validation_mse = mean_squared_error(validation_y, validation_yhat)
    validation_mae = mean_absolute_error(validation_y, validation_yhat)
    print(f'Fold: {index+1}, TrainMSE: {train_mse}, TrainMAE: {train_mae}, ValMSE: {validation_mse}, ValMAE: {validation_mae}')
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))
plt.scatter(all_labels, all_predictions)
plt.show();

# %%



