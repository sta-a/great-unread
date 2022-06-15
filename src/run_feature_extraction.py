# %%
# %load_ext autoreload
# %autoreload 2
from_commandline = False

import argparse
import sys
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from feature_extraction.doc2vec_chunk_vectorizer import Doc2VecChunkVectorizer
from feature_extraction.doc_based_feature_extractor import DocBasedFeatureExtractor
from feature_extraction.corpus_based_feature_extractor import CorpusBasedFeatureExtractor
from utils import get_doc_paths

if from_commandline:
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='eng')
    args = parser.parse_args()
    language = args.language
else:
    # Don't use defaults because VSC interactive can't handle command line arguments
    language = 'eng'

raw_docs_dir = f'../data/raw_docs/{language}/'
features_dir = f'../data/features_test/{language}/' ################################
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

doc_paths = get_doc_paths(raw_docs_dir)[:2] #######################
sentences_per_chunk = 200

# %%
# Create doc2vec embeddings
for curr_sentences_per_chunk in [sentences_per_chunk, None]:
    doc2vec_chunk_embeddings_dir = raw_docs_dir.replace('/raw_docs', f'/doc2vec_chunk_embeddings_spc_{sentences_per_chunk}')
    print(doc2vec_chunk_embeddings_dir)
    if not os.path.exists(doc2vec_chunk_embeddings_dir):
        d2vcv = Doc2VecChunkVectorizer(language, curr_sentences_per_chunk)
        d2vcv.fit_transform(doc_paths)

# %%
# Document-based features
document_chunk_features = []
document_book_features = [] 
for doc_path in tqdm(doc_paths):
    fe = DocBasedFeatureExtractor(language, doc_path, sentences_per_chunk)
    chunk_features, book_features = fe.get_all_features()  
    document_chunk_features.extend(chunk_features)
    document_book_features.append(book_features)
print(len(document_book_features), len(document_chunk_features))
with open(features_dir + 'document_chunk_features' + '.pkl', 'wb') as f:
    pickle.dump(document_chunk_features, f, -1)
with open(features_dir + 'document_book_features' + '.pkl', 'wb') as f:
    pickle.dump(document_book_features, f, -1)

# %%
# Recalculate the chunk features for the whole book, which is treated as one chunk
document_chunk_features_fulltext = []
for doc_path in tqdm(doc_paths):
    fe = DocBasedFeatureExtractor(language, doc_path, sentences_per_chunk=None)###!!!
    chunk_features_fulltext, _ = fe.get_all_features()
    document_chunk_features_fulltext.extend(chunk_features_fulltext)
print(len(document_chunk_features_fulltext))
with open(features_dir + 'document_chunk_features_fulltext' + '.pkl', 'wb') as f:
    pickle.dump(document_chunk_features_fulltext, f, -1)

# %%
# Corpus-based features
cbfe = CorpusBasedFeatureExtractor(language, doc_paths, sentences_per_chunk, nr_features=2) ########
corpus_chunk_features, corpus_book_features = cbfe.get_all_features()
print(corpus_book_features)
with open(features_dir + 'corpus_chunk_features' + '.pkl', 'wb') as f:
    pickle.dump(corpus_chunk_features, f, -1)
with open(features_dir + 'corpus_book_features' + '.pkl', 'wb') as f:
    pickle.dump(corpus_book_features, f, -1)
print('corpus book features after running', corpus_book_features, '½½½½½½½½½½½½½½½½½½½½½½½½')
with open(features_dir + 'corpus_book_features' + '.pkl', 'rb') as f:
    corpus_book_features = pickle.load(f)
print('after reload###########################', corpus_book_features)

# %%
# Recalculate the chunk features for the whole book, which is considered as one chunk
cbfe_fulltext = CorpusBasedFeatureExtractor(language, doc_paths, sentences_per_chunk=None, nr_features=100)
with open(features_dir + 'cbfe' + '.pkl', 'wb') as f:
    pickle.dump(cbfe, f, -1)
corpus_chunk_features_fulltext, _ = cbfe_fulltext.get_all_features()
with open(features_dir + 'corpus_chunk_features_fulltext' + '.pkl', 'wb') as f:
    pickle.dump(corpus_chunk_features_fulltext, f, -1)

# %%
# Load
with open(features_dir + 'document_chunk_features' + '.pkl', 'rb') as f:
    document_chunk_features = pickle.load(f)
with open(features_dir + 'document_book_features' + '.pkl', 'rb') as f:
    document_book_features = pickle.load(f)
with open(features_dir + 'document_chunk_features_fulltext' + '.pkl', 'rb') as f:
    document_chunk_features_fulltext = pickle.load(f)
with open(features_dir + 'corpus_chunk_features' + '.pkl', 'rb') as f:
    corpus_chunk_features = pickle.load(f)
with open(features_dir + 'corpus_book_features' + '.pkl', 'rb') as f:
    corpus_book_features = pickle.load(f)
with open(features_dir + 'corpus_chunk_features_fulltext' + '.pkl', 'rb') as f:
    corpus_chunk_features_fulltext = pickle.load(f)
print('after reload###########################', corpus_book_features, '#########################3')

# %%
# Book features
document_book_features = pd.DataFrame(document_book_features)
print(document_book_features)
document_chunk_features_fulltext = pd.DataFrame(document_chunk_features_fulltext)
book_df = document_book_features\
            .merge(right=document_chunk_features_fulltext, on='file_name', how='outer', validate='one_to_one')\
            .merge(right=corpus_book_features, on='file_name', validate='one_to_one')\
            .merge(right=corpus_chunk_features_fulltext, on='file_name', validate='one_to_one')

# Chunk features
document_chunk_features = pd.DataFrame(document_chunk_features)
chunk_df = document_chunk_features.merge(right=corpus_chunk_features, on='file_name', how='outer', validate='one_to_one')
# Remove chunk id from file_name
chunk_df['file_name'] = chunk_df['file_name'].str.split('_').str[:4].str.join('_')

# Combine book features and averages of chunksaveraged chunk features
book_and_averaged_chunk_df = book_df.merge(chunk_df.groupby('file_name').mean().reset_index(drop=False), on='file_name')
chunk_and_copied_book_df = chunk_df.merge(right=book_df, on='file_name', how='outer', validate='many_to_one')

dfs = {'book_df': book_df, 'book_and_averaged_chunk_df': book_and_averaged_chunk_df, 'chunk_df': chunk_df, 'chunk_and_copied_book_df': chunk_and_copied_book_df}
for name, df in dfs.items():
    df = df.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
    df.to_csv(f'{features_dir}{name}.csv', index=False)
    
    #print(df.isnull().values.any())
    #print(df.columns[df.isna().any()].tolist())
# %%
