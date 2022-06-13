# %%

%load_ext autoreload
%autoreload 2
lang = 'eng'

import os
import sys
sys.path.insert(0, '../src/')
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from feature_extraction.doc2vec_chunk_vectorizer import Doc2VecChunkVectorizer
from feature_extraction.doc_based_feature_extractor import DocBasedFeatureExtractor
from feature_extraction.corpus_based_feature_extractor import CorpusBasedFeatureExtractor

from utils import get_doc_paths

raw_docs_dir = f'../data/raw_docs/{lang}/'
features_dir = f'../data/features/{lang}/'

if not os.path.exists(features_dir):
    os.makedirs(features_dir)


doc_paths = get_doc_paths(raw_docs_dir)

sentences_per_chunk = 200

## %%
# Create doc2vec embeddings
# for sentences_per_chunk in [None]: #, 200
#     d2vcv = Doc2VecChunkVectorizer(lang, sentences_per_chunk)
#     d2vcv.fit_transform(doc_paths)


# %%
## Document-based features
document_chunk_features = []
document_book_features = [] 

for doc_path in tqdm(doc_paths):
    fe = DocBasedFeatureExtractor(lang, doc_path, sentences_per_chunk)
    chunk_features, book_features = fe.get_all_features()  
    document_chunk_features.extend(chunk_features)
    document_book_features.append(book_features)
print(len(document_book_features), len(document_chunk_features))


with open(features_dir + 'document_chunk_features' + '.pkl', 'wb') as f:
    pickle.dump(document_chunk_features, f, -1)
with open(features_dir + 'document_book_features' + '.pkl', 'wb') as f:
    pickle.dump(document_book_features, f, -1)

# %%
# Recalculate the chunk features for the whole book, which is considered as one chunk
document_chunk_features_fulltext = []

for doc_path in tqdm(doc_paths):
    fe = DocBasedFeatureExtractor(lang, doc_path, sentences_per_chunk=None)
    chunk_features_fulltext, _ = fe.get_all_features()
    document_chunk_features_fulltext.extend(chunk_features_fulltext)
print(len(document_chunk_features_fulltext))

with open(features_dir + 'document_chunk_features_fulltext' + '.pkl', 'wb') as f:
    pickle.dump(document_chunk_features_fulltext, f, -1)


## %%
## Corpus-based features
cbfe = CorpusBasedFeatureExtractor(lang, doc_paths, sentences_per_chunk, nr_features=100) 

## %%
corpus_chunk_features, corpus_book_features = cbfe.get_all_features()

with open(features_dir + 'corpus_chunk_features' + '.pkl', 'wb') as f:
    pickle.dump(corpus_chunk_features, f, -1)

with open(features_dir + 'corpus_book_features' + '.pkl', 'wb') as f:
    pickle.dump(corpus_book_features, f, -1)

## %%
# Recalculate the chunk features for the whole book, which is considered as one chunk
cbfe_fulltext = CorpusBasedFeatureExtractor(lang, doc_paths, sentences_per_chunk=None, nr_features=100)

## %%
corpus_chunk_features_fulltext, _ = cbfe_fulltext.get_all_features()
with open(features_dir + 'corpus_chunk_features_fulltext' + '.pkl', 'wb') as f:
    pickle.dump(corpus_chunk_features_fulltext, f, -1)


## %%
# # Load document-based features  
# with open(features_dir + 'document_chunk_features' + '.pkl', 'rb') as f:
#     document_chunk_features = pickle.load(f)

# with open(features_dir + 'document_book_features' + '.pkl', 'rb') as f:
#     document_book_features = pickle.load(f)

# with open(features_dir + 'document_chunk_features_fulltext' + '.pkl', 'rb') as f:
#     document_chunk_features_fulltext = pickle.load(f)

# # Load corpus-based features  
# with open(features_dir + 'corpus_chunk_features' + '.pkl', 'rb') as f:
#     corpus_chunk_features = pickle.load(f)

# with open(features_dir + 'corpus_book_features' + '.pkl', 'rb') as f:
#     corpus_book_features = pickle.load(f)

# with open(features_dir + 'corpus_chunk_features_fulltext' + '.pkl', 'rb') as f:
#     corpus_chunk_features_fulltext = pickle.load(f)

## %%
# Book features
document_book_features = pd.DataFrame(document_book_features)
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


# for name, df in dfs.items():
#     df = df.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
#     df.to_csv(f'{features_dir}{name}.csv', index=False)
    
    #print(df.isnull().values.any())
    #print(df.columns[df.isna().any()].tolist())

# %%


# %%


# %%


# %%


# %%


# %%


## %%


## %%


## %%


## %%


## %%


