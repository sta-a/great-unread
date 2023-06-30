# %%
%load_ext autoreload
%autoreload 2

import os
from feature_extraction.doc2vec_chunk_vectorizer import Doc2VecChunkVectorizer
from utils import get_doc_paths
from feature_extraction.process import Tokenizer
import os
import sys
sys.path.append("..")
from utils import load_list_of_lines, get_bookname
import collections
import time


language = 'eng'
data_dir = '../data/'
raw_docs_dir = os.path.join(data_dir, 'raw_docs', language)
doc_paths = get_doc_paths(raw_docs_dir)
tokens_per_chunk = 500

# chunk_counter = 0
# texts = []
# for doc_path in doc_paths:
#     print(doc_path)
#     tokenized_words_path = doc_path.replace('/raw_docs', '/tokenized_words') 
#     with open(tokenized_words_path, 'r') as f:
#         # # Split to obtain list of words
#         # text = f.read().split()
#         # texts.append(text)
#         nr_chunks = sum(1 for _ in f)
#         chunk_counter += nr_chunks
#         print(f'Nr chunks: {nr_chunks}')
# print('nr chunks', chunk_counter)

# %%
# Tag each chunk with document name, use document vector associated with tag
d2vcv = Doc2VecChunkVectorizer(language=language, doc_paths=doc_paths, mode='doc_tags')
d2vcv.fit()
d2vcv.transform()

# %%
# Tag each chunk with unique chunk id, infer document vectors
# Doesn't work because token limit of 10'000 also applies to inferred documents
# https://github.com/RaRe-Technologies/gensim/issues/2583
# As infer_vector() uses the same optimized Cython functions as training behind-the-scenes, it also suffers from the same fixed-token-buffer size as training, where texts with more than 10000 tokens have all overflow tokens ignored.
# start = time.time()
# d2vcv = Doc2VecChunkVectorizer(language=language, doc_paths=doc_paths, mode='chunk_tags')
# d2vcv.fit()
# d2vcv.transform()
# print(f'{time.time()-start}s for calculation')

# Use both chunk id tags and document name tag as a  list
start = time.time()
d2vcv = Doc2VecChunkVectorizer(language=language, doc_paths=doc_paths, mode='both_tags')
d2vcv.fit()
d2vcv.transform()
print(f'{time.time()-start}s for calculation')

# Save document vectors for each chunk to use them as features for prediction
start = time.time()
d2vcv = Doc2VecChunkVectorizer(language=language, doc_paths=doc_paths, mode='chunk_features')
d2vcv.fit()
d2vcv.transform()
print(f'{time.time()-start}s for calculation')








# %%
import numpy as np
doc_path = doc_paths[0]

# Compare vectors of both to individual experiments
features_path = doc_path.replace('/raw_docs', f'/doc2vec_chunk_embeddings_tpc_{tokens_per_chunk}') + '.npz'
features = np.load(features_path)
both_path = '/home/annina/scripts/great_unread_nlp/data/doc2vec_bothtags_tpc_500/eng/dv.npz'
both = np.load(both_path)

for doc_path in doc_paths:
    doctag_path = doc_path.replace('/raw_docs', f'/doc2vec_doctags_tpc_{tokens_per_chunk}') + '.npz'
    doctag = load_list_of_lines(doctag_path, 'np')
    chunktag_path = doc_path.replace('/raw_docs', f'/doc2vec_chunktags_tpc_{tokens_per_chunk}') + '.npz'
    chunktag = load_list_of_lines(chunktag_path, 'np')

    # Compare dv of book 
    book_name = get_bookname(doc_path)
    dv_both = both[book_name].tolist()
    print(dv_both, '\n', doctag, '\n-------------------------')

    # Compare dv of chunks
    book_name = get_bookname(doc_path)
    dv_both = both[f'{book_name}_0'].tolist()
    print(dv_both, '\n', chunktag, '\n-------------------------')



# %%
with open('test.txt', 'w') as f:
    for i in both:
        f.write(i + '\n')


# %%
d2vcv.assess_docs()


# %%
for xx in d2vcv.model.dv.index_to_key:
    print(xx)


