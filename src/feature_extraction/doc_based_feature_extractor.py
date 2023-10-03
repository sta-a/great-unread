import os
import numpy as np
import textstat
import logging
import time
import sys
sys.path.append("..")
from utils import get_filename_from_path
from scipy.stats import entropy
from .process_rawtext import ChunkHandler
from .embeddings import SbertProcessor, D2vProcessor
from .analyze_chunk import Chunk
from .ngrams import NgramCounter
import pickle


class DocBasedFeatureExtractor():
    '''
    Extract features that can be generated from a single document.
    '''
    def __init__(self, 
        language,
        doc_path, 
        as_chunk,
        tokens_per_chunk,
        ngrams=None,
        get_ngrams=True,
        get_char_counts=True):

        self.logger = logging.getLogger(__name__)
        self.language = language
        self.doc_path = doc_path
        self.bookname = get_filename_from_path(self.doc_path)
        self.as_chunk = as_chunk
        self.tokens_per_chunk = tokens_per_chunk
        self.ngrams = ngrams
        self.get_ngrams = get_ngrams
        self.get_char_counts = get_char_counts

        self.chunks = self.get_chunks()


    def load_data_for_chunks(self):
        chunks_text = ChunkHandler(self.language, self.tokens_per_chunk).load_data(file_name=self.bookname, remove_punct=False, lower=False, as_chunk=self.as_chunk, as_sent=False)
        chunks_sents = ChunkHandler(self.language, self.tokens_per_chunk).load_data(file_name=self.bookname, remove_punct=False, lower=False, as_chunk=self.as_chunk, as_sent=True)
        assert len(chunks_text) == len(chunks_sents)
        
        sbert_embeddings = SbertProcessor(self.language, self.tokens_per_chunk).load_data(file_name=self.bookname, doc_path=self.doc_path)

        if self.as_chunk:
            mode = 'chunk'
        else:
            mode = 'full'
        d2v_embeddings = D2vProcessor(language=self.language, tokens_per_chunk=self.tokens_per_chunk).load_data(file_name=self.bookname, mode=mode, subdir=True)

        if self.ngrams is None and self.get_ngrams:
            nc = NgramCounter(self.language)
            ngrams = nc.load_all_ngrams(as_chunk=self.as_chunk)
        else:
            ngrams = self.ngrams

        return chunks_text, chunks_sents, sbert_embeddings, d2v_embeddings, ngrams

    def get_chunks(self):
        chunks_text, chunks_sents, sbert_embeddings, d2v_embeddings, ngrams = self.load_data_for_chunks()

        chunks = []
        chunk_idx_counter = 0
        sent_counter = 0
        for text, sentences in zip(chunks_text, chunks_sents):
            chunks.append(Chunk(
                language=self.language,
                doc_path = self.doc_path,
                as_chunk = self.as_chunk,
                tokens_per_chunk = self.tokens_per_chunk,
                chunk_idx = chunk_idx_counter,
                sent_counter = sent_counter,
                text = text,
                sentences = sentences,
                sbert_embeddings = sbert_embeddings[sent_counter:sent_counter+len(sentences)],
                d2v_embeddings = d2v_embeddings,
                ngrams = ngrams,
                get_ngrams = self.get_ngrams,
                get_char_counts = self.get_char_counts))

            chunk_idx_counter += 1
            sent_counter += len(sentences)
        return chunks


    def get_all_features(self):
        chunk_feature_mapping = {
            'ratio_of_punctuation_marks': self.get_ratio_of_punctuation_marks,
            'ratio_of_whitespaces': self.get_ratio_of_whitespaces,
            'ratio_of_exclamation_marks': self.get_ratio_of_exclamation_marks,
            'ratio_of_question_marks': self.get_ratio_of_question_marks,
            'ratio_of_commas': self.get_ratio_of_commas,
            'ratio_of_uppercase_letters': self.get_ratio_of_uppercase_letters,
            'average_number_of_words_in_sentence': self.get_average_number_of_words_in_sentence,
            'maximum_number_of_words_in_sentence': self.get_maximum_number_of_words_in_sentence,
            'nr_chars': self.get_nr_chars,
            'nr_words': self.get_nr_words,
            'nr_sentences': self.get_nr_sents,
            'longest_word_length': self.get_longest_word_length,
            'average_word_length': self.get_average_word_length,
            'unigram_entropy': self.get_unigram_entropy, # second order redundancy
            'bigram_entropy': self.get_bigram_entropy,
            'trigram_entropy': self.get_trigram_entropy,
            'type_token_ratio': self.get_type_token_ratio,
            'flesch_reading_ease_score': self.get_flesch_reading_ease_score,
            0: self.get_average_sbert_embeddings, 
            1: self.get_d2v_embeddings
            #'ratio_of_digits': self.get_ratio_of_digits,
            # 'average_paragraph_length': self.get_average_paragraph_length, # structural features
        }

        book_feature_mapping = {
            'd2v_intra_textual_variance': self.get_d2v_intra_textual_variance,
            'sbert_intra_textual_variance': self.get_sbert_intra_textual_variance,
            'd2v_stepwise_distance': self.get_d2v_stepwise_distance,
            'sbert_stepwise_distance': self.get_sbert_stepwise_distance
        }

        # extract chunk based features
        chunk_features = []
        for chunk in self.chunks:
            current_features = {'file_name': chunk.chunkname}
            for feature_name, feature_function in chunk_feature_mapping.items():
                if isinstance(feature_name, int):
                    current_features.update(feature_function(chunk))
                else:
                    current_features[feature_name] = feature_function(chunk)
            chunk_features.append(current_features)

        # extract book based features
        book_features = None
        if self.as_chunk == True:
            book_features = {}
            for feature_name, feature_function in book_feature_mapping.items():
                book_features['file_name'] = self.doc_path.split('/')[-1][:-4]
                book_features[feature_name] = feature_function(self.chunks)

        return chunk_features, book_features


    def get_ratio_of_punctuation_marks(self, chunk):
        punctuations = 0
        allowed_chars = [r"'", ',', '!', '?', '-', ';', '_', '—'] # chars that are allowed in preprocessing
        for character in allowed_chars:
            punctuations += chunk.char_counts.get(character, 0)
        all_characters = sum(list(chunk.char_counts.values()))
        return punctuations / all_characters

    # def get_ratio_of_digits(self, chunk):
    #     # Digits have been replaced with tag
    #     digits = 0
    #     all_characters = 0
    #     for character in [str(i) for i in range(10)]:
    #         digits += chunk.char_counts.get(character, 0)
    #     all_characters = sum(list(chunk.char_counts.values()))
    #     return digits / all_characters

    def get_ratio_of_whitespaces(self, chunk):
        return chunk.char_counts.get(' ', 0) / sum(list(chunk.char_counts.values()))

    def get_ratio_of_exclamation_marks(self, chunk):
        return chunk.char_counts.get('!', 0) / sum(list(chunk.char_counts.values()))

    def get_ratio_of_question_marks(self, chunk):
        return chunk.char_counts.get('?', 0) / sum(list(chunk.char_counts.values()))

    def get_ratio_of_commas(self, chunk):
        return chunk.char_counts.get(',', 0) / sum(list(chunk.char_counts.values()))

    def get_ratio_of_uppercase_letters(self, chunk):
        num_upper = 0
        num_alpha = 0
        for char in chunk.text:
            if char.isalpha():
                num_alpha += 1
                if char.isupper():
                    num_upper += 1
        return num_upper / num_alpha

    # def get_average_paragraph_length(self, chunk):
    # Doesn't work because some text have poem_like structure, for example 'Ainsworth_William-Harrison_Rookwood_1834'
    #     with open(self.doc_path, 'r') as f:
    #         raw_text = f.read()
    #     split_lengths = [len(curr_split) for curr_split in chunk.raw_text.split('\n')]
    #     return np.mean(split_lengths)

    def get_average_sbert_embeddings(self, chunk):
        average_embedding_features = dict((f'average_sbert_embedding_{index+1}', embedding_part) for index, embedding_part in enumerate(np.array(chunk.sbert_embeddings).mean(axis=0)))
        return average_embedding_features

    def get_d2v_embeddings(self, chunk):
        d2v_embeddings_features = dict((f'd2v_embedding_{index+1}', embedding_part) for index, embedding_part in enumerate(chunk.d2v_embeddings))
        return d2v_embeddings_features

    def get_average_number_of_words_in_sentence(self, chunk):
        sentence_lengths = []
        for sentence in chunk.sentences:
            sentence_lengths.append(len(sentence.split()))
        return np.mean(sentence_lengths)

    def get_maximum_number_of_words_in_sentence(self, chunk):
        sentence_lengths = []
        for sentence in chunk.sentences:
            sentence_lengths.append(len(sentence.split()))
        return np.max(sentence_lengths)

    def get_nr_chars(self, chunk):
        return len(chunk.text)
    
    def get_nr_words(self, chunk):
        return len(chunk.text.split())
    
    def get_nr_sents(self, chunk):
        return len(chunk.sentences)
    
    def get_longest_word_length(self, chunk):
        return len(max(chunk.unigram_counts['words'], key=len))

    def get_average_word_length(self, chunk):
        words = chunk.unigram_counts['words']
        counts = chunk.unigram_counts['counts']
        assert len(words) == len(counts)
        total_length = sum(len(word) * count for word, count in zip(words, counts))
        total_count = sum(counts)
        return total_length / total_count

    def get_unigram_entropy(self, chunk):
        return entropy(chunk.unigram_counts['counts'])
    
    def get_bigram_entropy(self, chunk):
        return entropy(chunk.bigram_counts['counts'])
    
    def get_trigram_entropy(self, chunk):
        return entropy(chunk.trigram_counts['counts'])
    
    def get_type_token_ratio(self, chunk):
        # Type-token ratio according to Algee-Hewitt et al. (2016)
        words = chunk.unigram_counts['words']
        counts = chunk.unigram_counts['counts']
        tokens = sum(counts)
        types = len(words)
        return types/tokens

    def get_flesch_reading_ease_score(self, chunk):
        return textstat.flesch_reading_ease(chunk.text)

    # def get_gunning_fog(self, chunk):
    #     '''''''''
    #     Not implemented for German. If we can find 'easy words' in German, then we can implement it ourselves.
    #     '''
    #     return textstat.gunning_fog(chunk.text)

    def get_intra_textual_variance(self, chunks, embedding_type):
        # Euclidean distance of a novel's chunks to its centroid
        # Cannot be calculated unless there are more than 1 chunk
        if len(chunks) == 1:
            return np.nan
        chunk_embeddings = []
        for chunk in chunks:
            if embedding_type == 'd2v':
                chunk_embeddings.append(chunk.d2v_embeddings)
            elif embedding_type == 'sbert':
                 chunk_embeddings.append(chunk.sbert_embeddings.mean(axis=0)) 
            else:
                raise Exception(f'Not a valid embedding type {embedding_type}')
        average_chunk_embedding = np.array(chunk_embeddings).mean(axis=0)
        euclidean_distances = [np.linalg.norm(average_chunk_embedding - chunk_embedding) for chunk_embedding in chunk_embeddings]
        return np.mean(euclidean_distances)

    def get_d2v_intra_textual_variance(self, chunks):
        return self.get_intra_textual_variance(chunks, 'd2v')

    def get_sbert_intra_textual_variance(self, chunks):
        return self.get_intra_textual_variance(chunks, 'sbert')

    def get_stepwise_distance(self, chunks, embedding_type):
        if len(chunks) == 1:
            return np.nan
        euclidean_distances = []
        for chunk_idx in range(1, len(chunks)):
            if embedding_type == 'd2v':
                current_chunk_embedding = chunks[chunk_idx].d2v_embeddings
                previous_chunk_embedding = chunks[chunk_idx - 1].d2v_embeddings
            elif embedding_type == 'sbert':
                current_chunk_embedding = chunks[chunk_idx].sbert_embeddings.mean(axis=0)
                previous_chunk_embedding = chunks[chunk_idx - 1].sbert_embeddings.mean(axis=0)
            else:
                raise Exception(f'Not a valid embedding type {embedding_type}')
            euclidean_distances.append(np.linalg.norm(current_chunk_embedding - previous_chunk_embedding))

        mean_distance = np.sum(euclidean_distances) / len(euclidean_distances)
        return mean_distance

    def get_d2v_stepwise_distance(self, chunks):
        return self.get_stepwise_distance(chunks, 'd2v')

    def get_sbert_stepwise_distance(self, chunks):
        return self.get_stepwise_distance(chunks, 'sbert')
