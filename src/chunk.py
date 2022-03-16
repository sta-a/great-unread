import re
import numpy as np
import json
from utils import unidecode_custom


class Chunk():
    def __init__(self, 
    sentences_per_chunk,
    doc_path, 
    book_name, 
    chunk_id, 
    tokenized_sentences, 
    sbert_sentence_embeddings, 
    doc2vec_chunk_embedding, 
    raw_text=None,
    unidecoded_raw_text=None,
    processed_sentences=None,
    unigram_counts=None,
    bigram_counts=None,
    trigram_counts=None,
    char_unigram_counts=None):

        self.sentences_per_chunk = sentences_per_chunk
        self.doc_path = doc_path
        self.book_name = book_name
        self.chunk_id = chunk_id
        self.tokenized_sentences = tokenized_sentences
        self.sbert_sentence_embeddings = sbert_sentence_embeddings # list of arrays, one array per sentence
        self.doc2vec_chunk_embedding = doc2vec_chunk_embedding

        if raw_text is None:
            self.raw_text = " ".join(tokenized_sentences)
        else:
            self.raw_text = raw_text
        if unidecoded_raw_text is None:
            self.unidecoded_raw_text = unidecode_custom(self.raw_text)
        else:
            self.unidecoded_raw_text = unidecoded_raw_text
        if processed_sentences is None:
            self.processed_sentences = self.__preprocess_sentences()
        else:
            self.processed_sentences = processed_sentences
        if unigram_counts is None:
            self.unigram_counts = self.__find_unigram_counts()
        else:
            self.unigram_counts = unigram_counts
        if bigram_counts is None:
            self.bigram_counts = self.__find_bigram_counts()
        else:
            self.bigram_counts = bigram_counts
        if trigram_counts is None:
            self.trigram_counts = self.__find_trigram_counts()
        else:
            self.trigram_counts = trigram_counts
        if char_unigram_counts is None:
            self.char_unigram_counts = self.__find_char_unigram_counts()
        else:
            self.char_unigram_counts = char_unigram_counts

    def __preprocess_sentences(self):
        def __preprocess_sentences_helper(text):
            text = text.lower()
            text = unidecode_custom(text)
            text = re.sub("[^a-zA-Z]+", " ", text).strip()
            text = text.split()
            text = " ".join(text)
            return text
        return [__preprocess_sentences_helper(sentence) for sentence in self.tokenized_sentences]

    def __find_unigram_counts(self):
        unigram_counts = {}
        for processed_sentence in self.processed_sentences:
            for unigram in processed_sentence.split():
                if unigram in unigram_counts.keys():
                    unigram_counts[unigram] += 1
                else:
                    unigram_counts[unigram] = 1
        return unigram_counts

    def __find_bigram_counts(self):
        processed_text = "<BOS> " + " <EOS> <BOS> ".join(self.processed_sentences) + " <EOS>"
        processed_text_split = processed_text.split()
        bigram_counts = {}
        for i in range(len(processed_text_split) - 1):
            current_bigram = processed_text_split[i] + " " + processed_text_split[i+1]
            if current_bigram in bigram_counts:
                bigram_counts[current_bigram] += 1
            else:
                bigram_counts[current_bigram] = 1
        return bigram_counts

    def __find_trigram_counts(self):
        processed_text = "<BOS> <BOS> " + " <EOS> <EOS> <BOS> <BOS> ".join(self.processed_sentences) + " <EOS> <EOS>"
        processed_text_split = processed_text.split()
        trigram_counts = {}
        for i in range(len(processed_text_split) - 2):
            current_trigram = processed_text_split[i] + " " + processed_text_split[i+1] + " " + processed_text_split[i+2]
            if current_trigram in trigram_counts.keys():
                trigram_counts[current_trigram] += 1
            else:
                trigram_counts[current_trigram] = 1
        return trigram_counts

    def __find_char_unigram_counts(self):
        char_unigram_counts = {}
        for character in self.unidecoded_raw_text:
            if character in char_unigram_counts.keys():
                char_unigram_counts[character] += 1
            else:
                char_unigram_counts[character] = 1
        return char_unigram_counts

    def __eq__(self, other):
        return (self.sentences_per_chunk == other.sentences_per_chunk) and \
                (self.doc_path == other.doc_path) and \
                (self.book_name == other.book_name) and \
                (self.chunk_id == other.chunk_id) and \
                (self.tokenized_sentences == other.tokenized_sentences) and \
                (all([np.array_equal(x,y) for x,y in zip(self.sbert_sentence_embeddings, other.sbert_sentence_embeddings)])) and \
                (np.array_equal(self.doc2vec_chunk_embedding, other.doc2vec_chunk_embedding)) and \
                (self.raw_text == other.raw_text) and \
                (self.unidecoded_raw_text == other.unidecoded_raw_text) and \
                (self.processed_sentences == other.processed_sentences) and \
                (self.unigram_counts == other.unigram_counts) and \
                (self.bigram_counts == other.bigram_counts) and \
                (self.trigram_counts == other.trigram_counts) and \
                (self.char_unigram_counts == other.char_unigram_counts)

    def to_json(self):

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, dict):
                    return vars(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [x.tolist() if isinstance(x, np.ndarrray) else x for x in obj]
                return json.JSONEncoder.default(self, obj)
            
        return json.dumps(vars(self), cls=NumpyEncoder)