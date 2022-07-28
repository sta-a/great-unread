# %%
# %load_ext autoreload
# %autoreload 2
from_commandline = True

import argparse
import chunk
import os
import numpy as np
import pandas as pd
import pickle
import time
from itertools import repeat
from multiprocessing import Pool, cpu_count
from feature_extraction.doc2vec_chunk_vectorizer import Doc2VecChunkVectorizer
from feature_extraction.doc_based_feature_extractor import DocBasedFeatureExtractor
from feature_extraction.corpus_based_feature_extractor import CorpusBasedFeatureExtractor
from utils import get_doc_paths


if from_commandline:
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='eng')
    parser.add_argument('--data_dir', default='../data')
    args = parser.parse_args()
    language = args.language
    data_dir = args.data_dir
else:
    # Don't use defaults because VSC interactive can't handle command line arguments
    language = 'eng'
    data_dir = '../data/'

nr_texts = 150    ################################################
raw_docs_dir = os.path.join(data_dir, 'raw_docs', language)
features_dir = os.path.join(data_dir, f'features_{nr_texts}', language)
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

print(features_dir)
if os.path.exists('/home/annina/scripts/great_unread_nlp/data/wordstat/eng/word_statistics.pkl'):
    os.remove('/home/annina/scripts/great_unread_nlp/data/wordstat/eng/word_statistics.pkl')
doc_paths = get_doc_paths(raw_docs_dir)[:nr_texts]  # errors in doc [13:14]
sents_per_chunk = 200


def _create_doc2vec_embeddings():
    for curr_sentences_per_chunk in [sents_per_chunk, None]:
        doc2vec_chunk_embeddings_dir = raw_docs_dir.replace('/raw_docs', f'/doc2vec_chunk_embeddings_spc_{curr_sentences_per_chunk}')
        if not os.path.exists(doc2vec_chunk_embeddings_dir):
            d2vcv = Doc2VecChunkVectorizer(language, curr_sentences_per_chunk)
            d2vcv.fit_transform(doc_paths)

def _get_doc_features_helper(doc_path, language, sentences_per_chunk):
    fe = DocBasedFeatureExtractor(language, doc_path, sentences_per_chunk=sentences_per_chunk)
    chunk_features, book_features = fe.get_all_features() 
    return [chunk_features, book_features]
    
def _get_doc_features(sentences_per_chunk, chunk_features_filename, book_features_filename):      
    all_chunk_features = []
    all_book_features = [] 

    nr_processes = max(cpu_count() - 2, 1)
    with Pool(processes=nr_processes) as pool:
        res = pool.starmap(_get_doc_features_helper, zip(doc_paths, repeat(language), repeat(sentences_per_chunk)))
        for doc_features in res:
            all_chunk_features.extend(doc_features[0])
            all_book_features.append(doc_features[1])
            
    print(len(all_chunk_features), len(all_book_features))
    with open(os.path.join(features_dir, f'{chunk_features_filename}.pkl'), 'wb') as f:
        pickle.dump(all_chunk_features, f, -1)
    # Save book features only once (not when running with fulltext chunks)
    if sentences_per_chunk != None:
        with open(os.path.join(features_dir, f'{book_features_filename}.pkl'), 'wb') as f:
            pickle.dump(all_book_features, f, -1)
    return all_chunk_features, all_book_features

def _get_corpus_features(sentences_per_chunk, chunk_features_filename, book_features_filename):
    cbfe = CorpusBasedFeatureExtractor(language, doc_paths, sentences_per_chunk=sentences_per_chunk, nr_features=100)
    chunk_features, book_features = cbfe.get_all_features()

    with open(os.path.join(features_dir, f'{chunk_features_filename}.pkl'), 'wb') as f:
        pickle.dump(chunk_features, f, -1)
    # Save book features only once (not when running with fulltext chunks)
    if sentences_per_chunk != None:
        with open(os.path.join(features_dir, f'{book_features_filename}.pkl'), 'wb') as f:
            pickle.dump(book_features, f, -1)
    return chunk_features, book_features


def _merge_features(doc_chunk_features, 
                    doc_book_features, 
                    doc_chunk_features_fulltext, 
                    corpus_chunk_features, 
                    corpus_book_features, 
                    corpus_chunk_features_fulltext):
    # Book features
    doc_book_features = pd.DataFrame(doc_book_features)
    doc_chunk_features_fulltext = pd.DataFrame(doc_chunk_features_fulltext)

    print('doc_book_features: ', doc_book_features.shape, 'doc_chunk_features_fulltext: ', doc_chunk_features_fulltext.shape, 'corpus_book_features: ', corpus_book_features.shape, 'corpus_chunk_features_fulltext: ', corpus_chunk_features_fulltext.shape)

    book_df = doc_book_features\
                .merge(right=doc_chunk_features_fulltext, on='file_name', how='outer', validate='one_to_one')\
                .merge(right=corpus_book_features, on='file_name', validate='one_to_one')\
                .merge(right=corpus_chunk_features_fulltext, on='file_name', validate='one_to_one')
    book_df.columns = [col + '_fulltext' if col != 'file_name' else col for col in book_df.columns]

    # Chunk features
    doc_chunk_features = pd.DataFrame(doc_chunk_features)
    chunk_df = doc_chunk_features.merge(right=corpus_chunk_features, on='file_name', how='outer', validate='one_to_one')
    # Remove chunk id from file_name
    chunk_df['file_name'] = chunk_df['file_name'].str.split('_').str[:4].str.join('_')
    chunk_df.columns = [col + '_chunk' if col != 'file_name' else col for col in chunk_df.columns]

    # Combine book features and averages of chunksaveraged chunk features
    # baac = book_and_averaged_chunk
    baac_df = book_df.merge(chunk_df.groupby('file_name').mean().reset_index(drop=False), on='file_name', validate='one_to_many')
    # cacb = chunk_and_copied_book_df
    cacb_df = chunk_df.merge(right=book_df, on='file_name', how='outer', validate='many_to_one')
    print(book_df.shape, chunk_df.shape, baac_df.shape, cacb_df.shape)

    dfs = {'book': book_df, 'baac': baac_df, 'chunk': chunk_df, 'cacb': cacb_df}
    # cols = [df.columns.tolist() for df in dfs.values()]
    # for name, df in dfs.items():
    #     with open(name, 'w') as f:
    #         f.write('\n'.join(df.columns.tolist()))

    for name, df in dfs.items():
        print(name)
        df = df.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
        file_path = os.path.join(features_dir, f'{name}_features.csv')
        df.to_csv(file_path, index=False)
    return dfs

            
# %%
if __name__ == '__main__':

    # #Load
    # with open(os.path.join(features_dir, 'doc_chunk_features.pkl'), 'rb') as f:
    #     doc_chunk_features = pickle.load(f)
    # with open(os.path.join(features_dir, 'doc_book_features.pkl'), 'rb') as f:
    #     doc_book_features = pickle.load(f)
    # with open(os.path.join(features_dir, 'doc_chunk_features_fulltext.pkl'), 'rb') as f:
    #     doc_chunk_features_fulltext = pickle.load(f)
    # with open(os.path.join(features_dir, 'corpus_chunk_features.pkl'), 'rb') as f:
    #     corpus_chunk_features = pickle.load(f)
    # with open(os.path.join(features_dir, 'corpus_book_features.pkl'), 'rb') as f:
    #     corpus_book_features = pickle.load(f)
    # with open(os.path.join(features_dir, 'corpus_chunk_features_fulltext.pkl'), 'rb') as f:
    #     corpus_chunk_features_fulltext = pickle.load(f)

    start = time.time()
    _create_doc2vec_embeddings()
    # doc-based features
    doc_chunk_features, doc_book_features = _get_doc_features(sents_per_chunk, 'doc_chunk_features', 'doc_book_features')
    # Recalculate the chunk features for the whole book, which is treated as one chunk
    doc_chunk_features_fulltext, _ = _get_doc_features(None, 'doc_chunk_features_fulltext', None)
    
    # Corpus-based features
    corpus_chunk_features, corpus_book_features = _get_corpus_features(sents_per_chunk, 'corpus_chunk_features', 'corpus_book_features')
    # Recalculate the chunk features for the whole book, which is considered as one chunk
    corpus_chunk_features_fulltext, _ = _get_corpus_features(None, 'corpus_chunk_features_fulltext', None)

    dfs = _merge_features(doc_chunk_features, 
                    doc_book_features, 
                    doc_chunk_features_fulltext, 
                    corpus_chunk_features, 
                    corpus_book_features, 
                    corpus_chunk_features_fulltext)

    runtime = time.time() - start
    print('Runtime with multiprocessing for all texts:', runtime)
    with open('runtime_tracker.txt', 'a') as f:
        #f.write(f'nr_texts,runtime\n')
        f.write(f'{nr_texts},{round(runtime, 2)}\n')

# %%
features_dir = os.path.join(data_dir, f'features_{nr_texts}', language)

from_file = True
if from_file:
    df = pd.read_csv(os.path.join(features_dir, 'cacb_features.csv'))
    # Check if df has missing values
    print('NaN in df: ', df.isna().values.any())
    print('Cols with NaNs in them: ', df.columns[df.isna().any()].tolist())
    # Number unique values in columns
    unique_vals_per_col = df.nunique(axis=0)
    unique_vals_per_col.sort_values().to_csv('unique_vals_per_col')
    print('Unique values per col: ', unique_vals_per_col)
    print('Frequency of unique values per col: ', unique_vals_per_col.value_counts(),)
    # constant columns
    print('Constant cols: ', df.loc[:,df.apply(pd.Series.nunique) == 1])

# %%
df[['doc2vec_stepwise_distance']].count_values()
# %%

# %%
df[['S->NP_PP_NP_fulltext']]
# %%
