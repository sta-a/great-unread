import sys
sys.path.append("..")
import numpy as np
from utils import get_filename_from_path
import time
from .ngrams import NgramCounter

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
        self.sbert_embeddings = self.load_sbert()
        self.d2v_embeddings = self.d2v_embeddings[self.chunkname]

        if self.get_ngrams:
            self.unigram_counts, self.bigram_counts, self.trigram_counts = self.load_ngrams()
        if self.get_char_counts:
            countstart = time.time()
            self.char_counts = self.count_chars()
            # if as_chunk == False:
            #     print(f'{time.time()-countstart}s to calculate char counts.')

        # if self.get_ngrams:
        #     for my_dict in [self.unigram_counts, self.bigram_counts, self.trigram_counts]: ######################3
        #         first_10_elements = {k: my_dict[k] for k in list(my_dict.keys())[:10]}
        #         print(first_10_elements)

        # print('d2v')
        # print(self.d2v_embeddings)
        # print('sbert')
        # print(self.sbert_embeddings)

    def get_chunkname(self):
        fn = get_filename_from_path(self.doc_path)
        if self.as_chunk:
            fn = f'{fn}_{self.chunk_idx}'
        return fn
    
    
    def load_sbert(self):
        if not self.as_chunk:
            return np.concatenate(self.sbert_embeddings, axis=0)
        else:
            return self.sbert_embeddings
    

    def load_ngrams(self):
        nc = NgramCounter(self.language)
        uc = nc.load_values_for_chunk(file_name=self.chunkname, data_dict=self.ngrams['unigram'], values_only=False)
        uc = {ngram: count for ngram, count in uc.items() if count != 0}
        bc = nc.load_values_for_chunk(file_name=self.chunkname, data_dict=self.ngrams['bigram'], values_only=True)
        tc = nc.load_values_for_chunk(file_name=self.chunkname, data_dict=self.ngrams['trigram'], values_only=True)
        return uc, bc, tc


    def count_chars(self):
        # Use raw text with punctuation but without capitalization
        char_counts = {}
        for character in self.text.lower():
            if character in char_counts.keys():
                char_counts[character] += 1
            else:
                char_counts[character] = 1
        return char_counts