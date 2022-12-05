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
features = 'book'

data_dir = '../data'
canonscores_dir = os.path.join(data_dir, 'canonscores')
n_outer_folds = 5



# %%
# Nr reviewed texts after contradicting labels were removed: 191
for language in languages:
    # Use full features set for classification to avoid error with underrepresented classes
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
    gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
    if not os.path.exists(gridsearch_dir):
        os.makedirs(gridsearch_dir, exist_ok=True)


    ## Sentiscores
    # Review year 
    if language == 'eng':
        sentiscores_file = 'ENG_reviews_senti_FINAL.csv'
    else:
        sentiscores_file = 'GER_reviews_senti_FINAL.csv'
    sentiscores_df = pd.read_csv(os.path.join(sentiscores_dir, sentiscores_file), header=0, sep=';')

    # Nr of different texts that were reviewed
    nr_reviewed_texts = len(sentiscores_df.file_name.unique())

    sentiscores_df = sentiscores_df[['file_name', 'textfile', 'pub_year']]
    # ID des reviewten Texts _ Jahr der Rezension _ Kürzel der Zeitschrift _ Seitenangabe
    sentiscores_df['review_year'] = sentiscores_df['textfile'].str.split(pat='_', expand=True)[1]
    #sentiscores_df['review_year'].sort_values().to_csv(f'review-year_{language}')


    ## Library Scores
    if language == 'eng':
        lib_file = 'ENG_texts_circulating-libs.csv'
    else:
        lib_file = 'GER_texts_circulating-libs.csv'
    lib_df = pd.read_csv(os.path.join(metadata_dir, lib_file), header=0, sep=';')[['file_name', 'pub_year']]
    old_lib_df = pd.read_csv('/home/annina/scripts/great_unread_nlp/data/metadata_old/ger/GER_texts_circulating-libs.csv', header=0, sep=';')[['file_name', 'pub_year']]

    # publication year
    lib_df['publication_year'] = lib_df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
    lib_df = lib_df.loc[lib_df['pub_year'] != lib_df['publication_year']]
    lib_df = lib_df.drop(columns='publication_year')
    lib_df = lib_df.sort_values(by='file_name')
    lib_df.to_csv(f'publication-year_{language}.csv', index=False)


    old_lib_df['publication_year'] = old_lib_df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
    old_lib_df = old_lib_df.loc[old_lib_df['pub_year'] != old_lib_df['publication_year']]
    old_lib_df = old_lib_df.drop(columns='publication_year')
    old_lib_df = old_lib_df.sort_values(by='file_name')
    old_lib_df.to_csv(f'publication-year-olddata_{language}.csv', index=False)

    if language == 'ger':
        difference = lib_df.merge(old_lib_df, how='outer', on='file_name')
        difference.to_csv(f'difference-old-new-libdata_{language}.csv', header=True, index=False)

    
    new_file_names = pd.read_csv('/home/annina/scripts/great_unread_nlp/src/new_file_names.csv', header=0)[['file_name', 'file_name_new']]
    print(new_file_names)



# %%
# Plot canon scores distribution
import matplotlib.pyplot as plt

for language in languages:
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)

    X, y = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)
    threshold = 0.5
    above_threshold = y.loc[y['y'] >= threshold]
    print(f'Nr texts with canon score above {threshold}: {above_threshold.shape}')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)# Set the title of plot
    ax.plot(y['y'].sort_values(), label=language)
    ax.set_title(f'Canon scores distribution {language}')
    plt.show()
# %%
