import sys
sys.path.append("..")
print(sys.path)
from utils import get_filename_from_path
import re
from .process_rawtext import Postprocessor
from .ngrams import NgramCounter

class Chunk():
    def __init__(self,
        language,
        tokens_per_chunk,
        doc_path, 
        chunk_id,
        text,
        sentences,
        sbert_embeddings, 
        d2v_embedding, 
        unigram_counts=False, 
        bigram_counts=False, 
        trigram_counts=False, 
        char_unigram_counts=False):

        self.language = language
        self.tokens_per_chunk = tokens_per_chunk
        self.doc_path = doc_path
        self.chunk_id = chunk_id
        self.chunkname = self.get_chunkname()
        print('chunkname', self.chunkname)
        self.text = text
        self.sentences = sentences
        self.sbert_embeddings = sbert_embeddings
        self.d2v_embedding = d2v_embedding
        self.unigram_counts, self.bigram_counts, self.trigram_counts = self.load_ngrams()
        self.char_unigram_counts = char_unigram_counts


        # if unigram_counts == True:
        #     self.unigram_counts = self.__find_unigram_counts()
        # if bigram_counts == True:
        #     self.bigram_counts = self.__find_bigram_counts()
        # if self.trigram_counts == True:
        #     self.trigram_counts = self.__find_trigram_counts()
        # if self.char_unigram_counts == True:
        #     self.char_unigram_counts = self.__find_char_unigram_counts()

    def get_chunkname(self):
        fn = get_filename_from_path(self.doc_path)
        if self.chunk_id is not None:
            fn = f'{fn}_{self.chunk_id}'
        return fn


    def load_ngrams(self):
        c = NgramCounter(self.language)
        if self.chunk_id is None:
            c.load_data(file_name='unigram_full')
            uc = c.load_values_for_chunk(self.chunkname)
            c.load_data(file_name='bigram_full')
            bc = c.load_values_for_chunk(self.chunkname)
            c.load_data(file_name='trigram_full')
            tc = c.load_values_for_chunk(self.chunkname)
        else:
            c.load_data(file_name='unigram_chunk')
            uc = c.load_values_for_chunk(self.chunkname)
            c.load_data(file_name='bigram_chunk')
            bc = c.load_values_for_chunk(self.chunkname)
            c.load_data(file_name='trigram_chunk')
            tc = c.load_values_for_chunk(self.chunkname)

        return uc, bc, tc


    def __find_char_unigram_counts(self):
        # Use raw text with punctuation but without capitalization
        char_unigram_counts = {}
        for character in self.text:
            if character in char_unigram_counts.keys():
                char_unigram_counts[character] += 1
            else:
                char_unigram_counts[character] = 1
        return char_unigram_counts