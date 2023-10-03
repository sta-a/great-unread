import sys
sys.path.append("..")
import numpy as np
from utils import get_filename_from_path
import time
from .ngrams import NgramCounter
from sklearn.feature_extraction.text import CountVectorizer


class Chunk():
    def __init__(self,
        language,
        doc_path,
        as_chunk,
        tokens_per_chunk,
        chunk_idx,
        sent_counter,
        text,
        sentences,
        sbert_embeddings, 
        d2v_embeddings, 
        ngrams,
        get_ngrams=True,
        get_char_counts=True):

        self.language = language
        self.doc_path = doc_path
        self.as_chunk = as_chunk
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_idx = chunk_idx
        self.sent_counter = sent_counter
        self.text = text
        self.sentences = sentences
        self.sbert_embeddings = sbert_embeddings
        self.d2v_embeddings = d2v_embeddings
        self.ngrams = ngrams
        self.get_ngrams = get_ngrams
        self.get_char_counts = get_char_counts

        assert len(self.sentences) == len(self.sbert_embeddings)

        self.chunkname = self.get_chunkname()
        # self.sbert_embeddings = self.load_sbert()
        self.d2v_embeddings = self.d2v_embeddings[self.chunkname]

        if self.get_ngrams:
            self.unigram_counts, self.bigram_counts, self.trigram_counts = self.load_ngrams()
        if self.get_char_counts:
            self.char_counts = self.count_chars()

    def get_chunkname(self):
        fn = get_filename_from_path(self.doc_path)
        if self.as_chunk:
            fn = f'{fn}_{self.chunk_idx}'
        return fn
    
    
    # def load_sbert(self):
    #     if not self.as_chunk:
    #         print(self.as_chunk, type(self.sbert_embeddings), self.sbert_embeddings.ndim)
    #         print(self.as_chunk, np.vstack(self.sbert_embeddings).shape)
    #         return np.vstack(self.sbert_embeddings)
    #     else:
    #         print(self.as_chunk, type(self.sbert_embeddings), self.sbert_embeddings.shape, self.sbert_embeddings[0].shape)
    #         return self.sbert_embeddings
    

    # def load_ngrams(self):
    #     nc = NgramCounter(self.language)
    #     uc = nc.load_values_for_chunk(file_name=self.chunkname, data_dict=self.ngrams['unigram'], values_only=False)
        
    #     uctime = time.time()
    #     uc = {ngram: count for ngram, count in uc.items() if count != 0}
    #     print(f'overhead {time.time()-uctime}')
    #     # bc = nc.load_values_for_chunk(file_name=self.chunkname, data_dict=self.ngrams['bigram'], values_only=True)
    #     # tc = nc.load_values_for_chunk(file_name=self.chunkname, data_dict=self.ngrams['trigram'], values_only=True)
    #     bc = []
    #     tc = [] #####################################
    #     return uc, bc, tc


    def count_chars(self):
        # Use raw text with punctuation but without capitalization
        char_counts = {}
        for character in self.text.lower():
            if character in char_counts.keys():
                char_counts[character] += 1
            else:
                char_counts[character] = 1
        return char_counts
    

    def load_ngrams(self):
        unigram_counts = self.create_ngrams('unigram')
        bigram_counts = self.create_ngrams('bigram')
        trigram_counts = self.create_ngrams('trigram')
        return unigram_counts, bigram_counts, trigram_counts

    def create_ngrams(self, ntype):
        if ntype == 'unigram':
            ngram_range = (1, 1)
            # CV's default regex pattern only matches strings that are at least 2 chars long.
            # This pattern also matches strings that are only 1 char long (a, I...).
            cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=ngram_range, dtype=np.int32)
        else:
            if ntype == 'bigram':
                ngram_range = (2, 2)
            else:
                ngram_range = (3, 3)
            # cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=ngram_range, dtype=np.int32, max_features=2000)
            cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=ngram_range, dtype=np.int32)

        dtm = cv.fit_transform([self.text])
        if ntype == 'unigram':
            words = cv.get_feature_names_out()
        else:
            words = None

        data_dict = {
            'counts': dtm.toarray()[0].tolist(),
            'words': words,
        }
        return data_dict