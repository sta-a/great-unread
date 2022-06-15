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

raw_docs_dir = '../data/raw_docs/'
labels_dir = '../data/labels/'


def get_doc_paths(docs_dir, lang):
    doc_paths = [os.path.join(docs_dir, lang, doc_name) for doc_name in os.listdir(os.path.join(docs_dir, lang)) if doc_name[-4:] == '.txt']
    return doc_paths

def get_pickle_paths(pickles_dir, lang):
    pickle_paths = [os.path.join(pickles_dir, lang, pickle_name) for pickle_name in os.listdir(os.path.join(pickles_dir, lang)) if pickle_name[-7:] == '.pickle']
    return pickle_paths

def read_labels(lang):
    labels_df = pd.read_csv(os.path.join(labels_dir, lang, 'canonisation_scores.csv'), sep=';')
    if lang == 'eng':
        file_name_mapper = {
            'The Wild Irish Girl': 'Owenson_Sydney_The-Wild-Irish-Girl_1806',
            'Somerville-Ross_The-Real-Charlotte_1894': 'Somerville-Ross_Edith-Martin_The-Real-Charlotte_1894',
            'LeFanu_Joseph-Sheridan_Schalken-the-Painter_1851.txt': 'LeFanu_Joseph-Sheridan_Schalken-the-Painter_1851',
        }

        for key, value in file_name_mapper.items():
            labels_df['file_name'][labels_df['file_name'] == key] = value
        
        extra_file_names = [
            'Austen_Jane_Northanger-Abbey_1818',
            'Cleland_John_Fanny-Hill_1748',
            'Defoe_Daniel_Roxana_1724',
            'Fielding_Henry_Amelia_1752',
            'Kingsley_Charles_The-Water-Babies_1863',
            'Le-Queux_William_The-Invasion-of-1910_1906',
            'Surtees_Robert_Jorrocks-Jaunts-and-Jollities_1831'
        ]
        labels = dict(labels_df[~labels_df['file_name'].isin(extra_file_names)][['file_name', 'percent']].values)
    elif lang == 'ger':
        file_name_mapper = {
            'Ebner-Eschenbach_Marie-von_Bozena_1876': 'Ebner-Eschenbach_Marie_Bozena_1876',
            'Ebner-Eschenbach_Marie-von_Das-Gemeindekind_1887': 'Ebner-Eschenbach_Marie_Das-Gemeindekind_1887',
            'Ebner-Eschenbach_Marie-von_Der-Kreisphysikus_1883': 'Ebner-Eschenbach_Marie_Der-Kreisphysikus_1883',
            'Ebner-Eschenbach_Marie-von_Der-Muff_1896': 'Ebner-Eschenbach_Marie_Der-Muff_1896',
            'Ebner-Eschenbach_Marie-von_Die-Freiherren-von-Gemperlein_1889': 'Ebner-Eschenbach_Marie_Die-Freiherren-von-Gemperlein_1889',
            'Ebner-Eschenbach_Marie-von_Die-Poesie-des-Unbewussten_1883': 'Ebner-Eschenbach_Marie_Die-Poesie-des-Unbewussten_1883',
            'Ebner-Eschenbach_Marie-von_Die-Resel_1883': 'Ebner-Eschenbach_Marie_Die-Resel_1883',
            'Ebner-Eschenbach_Marie-von_Ein-kleiner-Roman_1887': 'Ebner-Eschenbach_Marie_Ein-kleiner-Roman_1887',
            'Ebner-Eschenbach_Marie-von_Krambabuli_1883': 'Ebner-Eschenbach_Marie_Krambabuli_1883',
            'Ebner-Eschenbach_Marie-von_Lotti-die-Uhrmacherin_1874': 'Ebner-Eschenbach_Marie_Lotti-die-Uhrmacherin_1874',
            'Ebner-Eschenbach_Marie-von_Rittmeister-Brand_1896': 'Ebner-Eschenbach_Marie_Rittmeister-Brand_1896',
            'Ebner-Eschenbach_Marie-von_Unsuehnbar_1890': 'Ebner-Eschenbach_Marie_Unsuehnbar_1890',
            'Hunold_Christian-Friedrich_Adalie_1702': 'Hunold_Christian_Friedrich_Die-liebenswuerdige-Adalie_1681'
        }
        for key, value in file_name_mapper.items():
            labels_df['file_name'][labels_df['file_name'] == key] = value
        labels = dict(labels_df[['file_name', 'percent']].values)
    return labels


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
from vectorizer import BertVectorizer

lang = 'eng'
bv = BertVectorizer(lang=lang, sentence_to_doc_agg='first')
pickle_paths = get_pickle_paths('../data/processed_bert_512_tokens_not_padded/', lang)
bv.fit(pickle_paths)
df = bv.get_doc_vectors()
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
    pca = PCA(n_components=150)
    train_X = pca.fit_transform(train_X)
    print(pca.explained_variance_ratio_.sum())
    validation_X = pca.transform(validation_X)
    model = SVR(C=20)
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



