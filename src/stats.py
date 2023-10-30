# %%
# %load_ext autoreload
# %autoreload 2

'''
Functions and classes for calculating and plotting essential metrics of the corpus.
'''


import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import os
from utils import DataHandler, get_filename_from_path, get_files_in_dir, DataLoader
#from hpo_functions import get_data, ColumnTransformer

class TextStatistics(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='svg')
        self.text_tokenized_dir = self.text_raw_dir.replace('/text_raw', '/text_tokenized') 

    def get_longest_shortest_text(self):
        text_lengths= {}
        # tokenized_paths = get_files_in_dir(self.text_tokenized_dir)
        for tokenized_path in self.doc_paths: ##############################3
            with open(tokenized_path, 'r') as f:
                text = f.read().split()
                text_lengths[get_filename_from_path(tokenized_path)] = len(text)
                

        text_lengths= dict(sorted(text_lengths.items(), key=lambda item: item[1]))

        key_max = max(text_lengths.keys(), key=(lambda k: text_lengths[k]))
        key_min = min(text_lengths.keys(), key=(lambda k: text_lengths[k]))

        print('Maximum Value: ', key_max, text_lengths[key_max])
        print('Minimum Value: ', key_min, text_lengths[key_min])

        # Print the shortest 10 texts and their lengths
        print('Shortest 10 Texts:')
        shortest_texts = list(text_lengths.keys())[:10]  # Get the list of shortest text names
        print(shortest_texts)
        for key in shortest_texts:
            print(key, text_lengths[key])


        title = (
            f'Maximum Value: {key_max}, {text_lengths[key_max]}\n'
            f'Minimum Value: {key_min}, {text_lengths[key_min]}\n'
        )
        self.plot_words_per_text(text_lengths, title)

    def plot_words_per_text(self, text_lengths, title):
        plt.figure(figsize=(10, 6))
        # x_values = range(len(text_lengths)) 
        plt.bar(text_lengths.keys(), text_lengths.values())
        plt.xlabel('Document')
        plt.ylabel('Number of Words')
        plt.title('Number of Words per Text')
        plt.xticks(rotation=45)
        plt.xticks([])  # Remove x-axis labels
        plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        plt.tight_layout()
        plt.text(0.9, 0.95, title, transform=plt.gca().transAxes, va='top',
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
        # plt.show()
        self.save_data(data=plt, file_name='words-per-text')
        plt.close()



class PlotCanonscores(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='svg')
        self.df = DataLoader(self.language).prepare_metadata(type='canon')

    
    def plot(self):
        df = self.df.sort_values(by='canon')

        min_score = df['canon'].min()
        max_score = df['canon'].max()

        print(f"Minimum Score: {min_score}")
        print(f"Maximum Score: {max_score}")

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df)), df['canon'])
        plt.title('Canon Scores Distribution')
        plt.xlabel('Texts')
        plt.ylabel('Scores')
        
        print(df)
        x_ticks = range(0, len(df), 100)
        # plt.xticks(x_ticks, df.iloc[x_ticks]['file_name'], rotation=45, ha='right')
        plt.xticks(x_ticks, [str(i) for i in x_ticks], rotation=45, ha='right')

        self.save_data(data=plt, data_type='svg', file_name=f'canon-scores')
        plt.close()


class PlotFeatureDist(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='svg')


    def plot(self):
        df = DataLoader(self.language).prepare_features()
        fig, axes = plt.subplots(nrows=13, ncols=7, figsize=(30, 40))

        # Flatten the 2D array of axes for easier indexing
        axes = axes.flatten()

        for i, column in enumerate(df.columns):
            self.df[column].hist(ax=axes[i], bins=20)
            axes[i].set_title(column)

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Save the figure
        self.save_data(data=plt, data_type='svg', file_name=f'feature-dist')
        plt.close()




# for language in ['eng', 'ger']:
# #     ts = TextStatistics(language)
# #     ts.get_longest_shortest_text()

#     # pc = PlotCanonscores(language)
#     # pc.plot()

#     pfd = PlotFeatureDist(language)
#     pfd.plot()











# %%
task = 'regression-importances'
label_type = 'canon'
# Importances are calculated on cacb features
features = 'book'

data_dir = '../data'
canonscores_dir = os.path.join(data_dir, 'canonscores')
n_outer_folds = 5



# %%
# Nr reviewed texts after contradicting labels were removed: 191
# for language in languages:
#     # Use full features set for classification to avoid error with underrepresented classes
#     sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
#     metadata_dir = os.path.join(data_dir, 'metadata', language)
#     features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
#     gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
#     if not os.path.exists(gridsearch_dir):
#         os.makedirs(gridsearch_dir, exist_ok=True)


#     ## Sentiscores
#     # Review year 
#     if language == 'eng':
#         sentiscores_file = 'ENG_reviews_senti_FINAL.csv'
#     else:
#         sentiscores_file = 'GER_reviews_senti_FINAL.csv'
#     sentiscores_df = pd.read_csv(os.path.join(sentiscores_dir, sentiscores_file), header=0, sep=';')

#     # Nr of different texts that were reviewed
#     nr_reviewed_texts = len(sentiscores_df.file_name.unique())

#     sentiscores_df = sentiscores_df[['file_name', 'textfile', 'pub_year']]
#     # ID des reviewten Texts _ Jahr der Rezension _ Kürzel der Zeitschrift _ Seitenangabe
#     sentiscores_df['review_year'] = sentiscores_df['textfile'].str.split(pat='_', expand=True)[1]
#     #sentiscores_df['review_year'].sort_values().to_csv(f'review-year_{language}')


#     ## Library Scores
#     if language == 'eng':
#         lib_file = 'ENG_texts_circulating-libs.csv'
#     else:
#         lib_file = 'GER_texts_circulating-libs.csv'
#     lib_df = pd.read_csv(os.path.join(metadata_dir, lib_file), header=0, sep=';')[['file_name', 'pub_year']]
#     old_lib_df = pd.read_csv('/home/annina/scripts/great_unread_nlp/data/metadata_old/ger/GER_texts_circulating-libs.csv', header=0, sep=';')[['file_name', 'pub_year']]

#     # publication year
#     lib_df['publication_year'] = lib_df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
#     lib_df = lib_df.loc[lib_df['pub_year'] != lib_df['publication_year']]
#     lib_df = lib_df.drop(columns='publication_year')
#     lib_df = lib_df.sort_values(by='file_name')
#     lib_df.to_csv(f'publication-year_{language}.csv', index=False)


#     old_lib_df['publication_year'] = old_lib_df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
#     old_lib_df = old_lib_df.loc[old_lib_df['pub_year'] != old_lib_df['publication_year']]
#     old_lib_df = old_lib_df.drop(columns='publication_year')
#     old_lib_df = old_lib_df.sort_values(by='file_name')
#     old_lib_df.to_csv(f'publication-year-olddata_{language}.csv', index=False)

#     if language == 'ger':
#         difference = lib_df.merge(old_lib_df, how='outer', on='file_name')
#         difference.to_csv(f'difference-old-new-libdata_{language}.csv', header=True, index=False)

    
#     new_file_names = pd.read_csv('/home/annina/scripts/great_unread_nlp/src/new_file_names.csv', header=0)[['file_name', 'file_name_new']]
#     print(new_file_names)


