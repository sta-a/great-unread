# %%
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import os
from utils import DataHandler, get_doc_paths, get_filename_from_path
from feature_extraction.process_rawtext import TextChunker
#from hpo_functions import get_data, ColumnTransformer

class DataChecker(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='png')
        self.text_raw_dir = os.path.join(self.data_dir, 'text_raw', language)
        self.text_tokenized_dir = self.text_raw_dir.replace('/text_raw', '/text_tokenized') 
        self.doc_paths = get_doc_paths(self.text_raw_dir)

    def get_longest_shortest_text(self):
        d = {}
        for doc_path in self.doc_paths:
            with open(doc_path, 'r') as f:
                text = f.read().split()
                d[get_filename_from_path(doc_path)] = len(text)

        d = dict(sorted(d.items(), key=lambda item: item[1]))

        key_max = max(d.keys(), key=(lambda k: d[k]))
        key_min = min(d.keys(), key=(lambda k: d[k]))

        print('Maximum Value: ', key_max, d[key_max])
        print('Minimum Value: ', key_min, d[key_min])

        title = (
            f'Maximum Value: {key_max}, {d[key_max]}\n'
            f'Minimum Value: {key_min}, {d[key_min]}\n'
        )
        self.plot_words_per_text(d, title)

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
        plt.text(0.05, 0.95, title, transform=plt.gca().transAxes, va='top',
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
        # plt.show()
        self.save_data(data=plt, file_name='words-per-text')
        plt.close()


    def count_chunks_per_doc(self):
        # Count chunks
        nr_chunks_per_doc = {}

        # Nr chunks per doc
        for doc_path in self.doc_paths:
            text_tokenized_path = doc_path.replace('/text_raw', '/text_tokenized') 
            with open(text_tokenized_path, 'r') as f:
                nr_chunks = sum(1 for _ in f)
                nr_chunks_per_doc[get_filename_from_path(doc_path)] = nr_chunks
        print("Number of Chunks per Document:", nr_chunks_per_doc)

        # Calculate total number of chunks
        total_nr_chunks = sum(nr_chunks_per_doc.values())
        print('Total nr chunks:', total_nr_chunks)

        # Chunk count distribution
        chunk_count_freq = Counter(nr_chunks_per_doc.values())
        print("Chunk count freq:", chunk_count_freq)

        min_chunks_doc = min(nr_chunks_per_doc, key=nr_chunks_per_doc.get)
        max_chunks_doc = max(nr_chunks_per_doc, key=nr_chunks_per_doc.get)
        print("Document with minimum chunks:", min_chunks_doc, "Chunks:", nr_chunks_per_doc[min_chunks_doc])
        print("Document with maximum chunks:", max_chunks_doc, "Chunks:", nr_chunks_per_doc[max_chunks_doc])

        title = (
            f"Total nr chunks: {total_nr_chunks}\n"
            f"Document with minimum chunks: {min_chunks_doc}, Chunks: {nr_chunks_per_doc[min_chunks_doc]}\n"
            f"Document with maximum chunks: {max_chunks_doc}, Chunks: {nr_chunks_per_doc[max_chunks_doc]}"
        )

        self.plot_chunks_per_doc(chunk_count_freq, title)


    def plot_chunks_per_doc(self, chunk_count_freq, title):
        #chunk_count_freq = dict(sorted(list(chunk_count_freq.items()))[:10])
        plt.figure(figsize=(20, 6))
        
        # Calculate the bar width to avoid overlapping
        num_categories = len(chunk_count_freq)
        max_xrange = max(chunk_count_freq.keys()) - min(chunk_count_freq.keys())
        bar_width = max_xrange / (num_categories * 2)  # Adjust the divisor as needed
        

        plt.bar(chunk_count_freq.keys(), chunk_count_freq.values(), color='red', width=bar_width)
        plt.xlabel('Chunk Count')
        plt.ylabel('Frequency')
        plt.title('Chunk Frequency Distribution')
        plt.xticks(rotation=45)

        # Set y-axis ticks to integers
        max_freq = max(chunk_count_freq.values())
        print('max freq', max_freq)
        plt.yticks(range(max_freq + 1))
        # plt.tight_layout()
        # plt.show()
        # Set the title string into a box at the top right corner of the plot
        plt.text(0.95, 0.95, title, transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                va='top', ha='right')

        self.save_data(data=plt, file_name='chunks-per-doc')
        plt.close()

    def count_tokens_per_chunk(self):
        textchunker = TextChunker(language=self.language, tokens_per_chunk=self.tokens_per_chunk)

        counts = {}
        for doc_path in self.doc_paths:
            chunks = textchunker.load_data(file_name=get_filename_from_path(doc_path), doc_path=doc_path) # load chunks as a list of chunks
            chunks = ' '.join(chunks)
            l = len(chunks.split())
            counts[get_filename_from_path(doc_path)] = l

        # self.save_data(data=plt, file_name='tokens-per-chunk')
        self.plot_tokens_per_chunk(counts)


    def plot_tokens_per_chunk(self, counts):
        plt.figure(figsize=(8, 6))
        plt.bar(counts.keys(), counts.values())
        plt.xlabel('Text')
        plt.ylabel('Tokens per chunk')
        plt.title('Tokens per chunk')
        plt.xticks(rotation=45)
        self.save_data(data=plt, file_name='tokens-per-chunk')
        plt.close()


for language in ['eng', 'ger']:
    ts = DataChecker(language)
    # ts.get_longest_shortest_text()
    ts.count_chunks_per_doc()
    ts.count_tokens_per_chunk()





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
# Get # tokens in texts

# %%
