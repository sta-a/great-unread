import sys
sys.path.append("..")
print(sys.path)
from utils import get_bookname
import re
from .process_rawtext import Tokenizer, Postprocessor

class Chunk():
    def __init__(self, 
        tokens_per_chunk,
        doc_path, 
        chunk_id, 
        text_tokenized, 
        sbert_embeddings, 
        d2v_chunk_embedding, 
        unigram_counts=False, 
        bigram_counts=False, 
        trigram_counts=False, 
        char_unigram_counts=False):

        self.tokens_per_chunk = tokens_per_chunk
        self.doc_path = doc_path
        self.chunk_id = chunk_id
        self.text_tokenized = text_tokenized # raw
        self.sbert_embeddings = sbert_embeddings
        self.d2v_chunk_embedding = d2v_chunk_embedding
        self.unigram_counts = unigram_counts
        self.bigram_counts = bigram_counts
        self.trigram_counts = trigram_counts
        self.char_unigram_counts = char_unigram_counts

        self.file_name = get_bookname(self.doc_path)
        self.text_tokenized_pp = Postprocessor(remove_punct=True, lower=True).postprocess_text(self.text_tokenized)
        self.sentences = self.__split_into_sentences()

        if self.unigram_counts == True:
            self.unigram_counts = self.__find_unigram_counts()
        if self.bigram_counts == True:
            self.bigram_counts = self.__find_bigram_counts()
        if self.trigram_counts == True:
            self.trigram_counts = self.__find_trigram_counts()
        if self.char_unigram_counts == True:
            self.char_unigram_counts = self.__find_char_unigram_counts()

    def __split_into_sentences(self):
        # Split postprocessed text into sentences
        terminating_chars = r'\. | \: | \; | \? | \! | \) | \] | \...'
        sentences = re.split(terminating_chars, self.text_tokenized_pp)
        return sentences
    

    def __find_unigram_counts(self):
        words = self.text_tokenized_pp.split()
        unigram_counts = {}
        for unigram in words:
            if unigram in unigram_counts.keys():
                unigram_counts[unigram] += 1
            else:
                unigram_counts[unigram] = 1
        return unigram_counts

    def __find_bigram_counts(self):
        processed_text = '<BOS> ' + ' <EOS> <BOS> '.join(self.sentences) + ' <EOS>'
        processed_text_split = processed_text.split()
        bigram_counts = {}
        for i in range(len(processed_text_split) - 1):
            current_bigram = processed_text_split[i] + ' ' + processed_text_split[i+1]
            if current_bigram in bigram_counts:
                bigram_counts[current_bigram] += 1
            else:
                bigram_counts[current_bigram] = 1
        return bigram_counts

    def __find_trigram_counts(self):
        processed_text = '<BOS> <BOS> ' + ' <EOS> <EOS> <BOS> <BOS> '.join(self.sentences) + ' <EOS> <EOS>'
        processed_text_split = processed_text.split()
        trigram_counts = {}
        for i in range(len(processed_text_split) - 2):
            current_trigram = processed_text_split[i] + ' ' + processed_text_split[i+1] + ' ' + processed_text_split[i+2]
            if current_trigram in trigram_counts.keys():
                trigram_counts[current_trigram] += 1
            else:
                trigram_counts[current_trigram] = 1
        return trigram_counts


    def __find_char_unigram_counts(self):
        # Use raw text with punctuation but without capitalization
        char_unigram_counts = {}
        for character in self.text_tokenized:
            if character in char_unigram_counts.keys():
                char_unigram_counts[character] += 1
            else:
                char_unigram_counts[character] = 1
        return char_unigram_counts