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
from process import Doc2VecProcessor
from vectorize import Doc2VecVectorizer
from utils import get_doc_paths, read_labels

raw_docs_dir = '../data/raw_docs/'
labels_dir = '../data/labels/'


# %% [markdown]
# # Process data

# %%
language = 'eng'
raw_doc_paths = get_doc_paths(raw_docs_dir, language)
dp = Doc2VecProcessor(language=language, processed_chunk_sentence_count=None, stride=None)
dp.process(raw_doc_paths)

# %%
language = 'ger'
raw_doc_paths = get_doc_paths(raw_docs_dir, language)
dp = Doc2VecProcessor(language=language, processed_chunk_sentence_count=None, stride=None)
dp.process(raw_doc_paths)

# %%
language = 'eng'
raw_doc_paths = get_doc_paths(raw_docs_dir, language)
dp = Doc2VecProcessor(language=language, processed_chunk_sentence_count=500, stride=500)
dp.process(raw_doc_paths)

# %%
language = 'ger'
raw_doc_paths = get_doc_paths(raw_docs_dir, language)
dp = Doc2VecProcessor(language=language, processed_chunk_sentence_count=500, stride=500)
dp.process(raw_doc_paths)

# %% [markdown]
# # Full documents + Doc2VecDMM + SVR

# %%
language = 'eng'
processed_full_doc_paths = get_doc_paths('../data/processed_doc2vec_full/', language)
d2vv = Doc2VecVectorizer(dm=1, dm_mean=1)
d2vv.fit(processed_full_doc_paths)
df = d2vv.get_doc_vectors()
labels = read_labels('eng')
df['y'] = df['doc_path'].apply(lambda x: labels[x.split('/')[-1][:-4]])
df = df.drop(columns=['doc_path'])

# %%
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

X = df.drop(columns=['y']).values
y = df['y'].values.ravel()

all_predictions = []
all_labels = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for index, (train_indices, validation_indices) in enumerate(kf.split(X)):
    train_X = X[train_indices, :]
    train_y = y[train_indices]
    validation_X = X[validation_indices, :]
    validation_y = y[validation_indices]
    
    model = SVR(C=30)
    model.fit(train_X, train_y)
    train_yhat = model.predict(train_X)
    validation_yhat = model.predict(validation_X)
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
plt.hist(all_labels)
plt.show();

# %%
plt.figure(figsize=(18, 6))
plt.scatter(all_labels, all_predictions)
plt.show();

# %% [markdown]
# # Chunks of 500 Sentences + Doc2VecDMM + SVR

# %%
language = 'eng'
processed_sc_500_st_500_doc_paths = get_doc_paths('../data/processed_doc2vec_sc_500_st_500/', language)
d2vv = Doc2VecVectorizer(dm=1, dm_mean=1)
d2vv.fit(processed_sc_500_st_500_doc_paths)
df = d2vv.get_doc_vectors()
labels = read_labels('eng')
df['y'] = df['doc_path'].apply(lambda x: labels[x.split('/')[-1][:-4].split('_pt')[0]])
df['file_name'] = df['doc_path'].apply(lambda x: x.split('/')[-1][:-4].split('_pt')[0])


# %%
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

all_predictions = []
all_labels = []

file_names = df['file_name'].unique()
file_names_splitted = np.array_split(file_names, 10)

for index, split in enumerate(file_names_splitted):
    train_X = df[~df['file_name'].isin(split)].drop(columns=['y', 'doc_path', 'file_name']).values
    train_y = df[~df['file_name'].isin(split)]['y'].values.ravel()
    validation_X = df[df['file_name'].isin(split)].drop(columns=['y', 'doc_path', 'file_name']).values
    validation_y = df[df['file_name'].isin(split)]['y'].values.ravel()
    
    model = SVR(C=30)
    model.fit(train_X, train_y)
    train_yhat = model.predict(train_X)
    validation_yhat = model.predict(validation_X)
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

# %% [markdown]
# # Full documents + Doc2VecDBOW + SVR

# %%
language = 'eng'
processed_full_doc_paths = get_doc_paths('../data/processed_doc2vec_full/', language)
d2vv = Doc2VecVectorizer(dm=0, dm_mean=0)
d2vv.fit(processed_full_doc_paths)
df = d2vv.get_doc_vectors()
labels = read_labels('eng')
df['y'] = df['doc_path'].apply(lambda x: labels[x.split('/')[-1][:-4]])
df = df.drop(columns=['doc_path'])

# %%
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

X = df.drop(columns=['y']).values
y = df['y'].values.ravel()

all_predictions = []
all_labels = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for index, (train_indices, validation_indices) in enumerate(kf.split(X)):
    train_X = X[train_indices, :]
    train_y = y[train_indices]
    validation_X = X[validation_indices, :]
    validation_y = y[validation_indices]
    
    model = MLPRegressor(hidden_layer_sizes=(80, 50, 30, 10), activation='relu', max_iter=50)
    model.fit(train_X, train_y)
    train_yhat = model.predict(train_X)
    validation_yhat = model.predict(validation_X)
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



