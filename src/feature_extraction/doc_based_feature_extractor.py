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
from sklearn.feature_extraction.text import CountVectorizer
from .analyze_chunk import Chunk
from .ngrams import NgramCounter


class DocBasedFeatureExtractor():
    '''
    Extract features that can be generated from a single document.
    '''
    def __init__(self, 
        language,
        doc_path, 
        as_chunk,
        ngrams=None,
        tokens_per_chunk=500,
        get_ngrams=True,
        get_char_counts=True):

        self.logger = logging.getLogger(__name__)
        self.language = language
        self.doc_path = doc_path
        self.bookname = get_filename_from_path(self.doc_path)
        self.as_chunk = as_chunk
        self.ngrams = ngrams
        self.tokens_per_chunk = tokens_per_chunk
        self.get_ngrams = get_ngrams
        self.get_char_counts = get_char_counts

        self.chunks = self.__get_chunks()


    def load_data_for_chunks(self):
        start = time.time()
        chunks_text = ChunkHandler(self.language, self.tokens_per_chunk).load_data(file_name=self.bookname, remove_punct=False, lower=False, as_chunk=self.as_chunk, as_sent=False)
        chunks_sents = ChunkHandler(self.language, self.tokens_per_chunk).load_data(file_name=self.bookname, remove_punct=False, lower=False, as_chunk=self.as_chunk, as_sent=True)
        assert len(chunks_text) == len(chunks_sents)
        
        sbert_embeddings = SbertProcessor(self.language).load_data(file_name=self.bookname, doc_path=self.doc_path)
        if self.as_chunk:
            mode = 'chunk'
        else:
            mode = 'full'
        d2v_embeddings = D2vProcessor(self.language).load_data(file_name=self.bookname, mode=mode, subdir=True)

        if self.ngrams is None and self.get_ngrams:
            nc = NgramCounter(self.language)
            ngrams = nc.load_all_ngrams(as_chunk=self.as_chunk)
        else:
            ngrams = self.ngrams

        # print(f'Time to load data for 1 doc: {time.time()-start}')
        return chunks_text, chunks_sents, sbert_embeddings, d2v_embeddings, ngrams

    def __get_chunks(self):
        cstart = time.time()
        chunks_text, chunks_sents, sbert_embeddings, d2v_embeddings, ngrams = self.load_data_for_chunks()

        chunks = []
        chunk_idx_counter = 0
        for text, sentences in zip(chunks_text, chunks_sents):
            start = time.time()
            chunks.append(Chunk(
                language=self.language,
                doc_path = self.doc_path,
                as_chunk = self.as_chunk,
                tokens_per_chunk = self.tokens_per_chunk,
                chunk_idx = chunk_idx_counter,
                text = text,
                sentences = sentences,
                sbert_embeddings = sbert_embeddings,
                d2v_embeddings = d2v_embeddings,
                ngrams = ngrams,
                get_ngrams = self.get_ngrams,
                get_char_counts = self.get_char_counts))
            # print(f'Time to make chunk: {time.time()-start}')

            chunk_idx_counter += 1

            # print(f'{time.time()-cstart}s to load all chunk data for as_chunk={self.as_chunk}.')
            # object_size = sys.getsizeof(chunks) #############################a
            # Print the size in bytes
            # print(f"Size of the chunks of 1 doc: {object_size} bytes")

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
            'ratio_of_unique_unigrams': self.get_ratio_of_unique_unigrams,
            'ratio_of_unique_bigrams': self.get_ratio_of_unique_bigrams,
            'ratio_of_unique_trigrams': self.get_ratio_of_unique_trigrams,
            'nr_chars': self.get_nr_chars,
            'nr_words': self.get_nr_words,
            'longest_word_length': self.get_longest_word_length,
            'average_word_length': self.get_average_word_length,
            'unigram_entropy': self.get_unigram_entropy, # second order redundancy
            'type_token_ratio': self.get_type_token_ratio,
            'flesch_reading_ease_score': self.get_flesch_reading_ease_score,
            0: self.get_average_sbert_embeddings, 
            1: self.get_d2v_embeddings
            #'ratio_of_digits': self.get_ratio_of_digits,
            # 'average_paragraph_length': self.get_average_paragraph_length, # structural features
        }

        book_feature_mapping = {
            'bigram_entropy': self.get_bigram_entropy,
            'trigram_entropy': self.get_trigram_entropy,
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
        average_embedding = chunk.sbert_embeddings.mean(axis=0)
        average_embedding_features = dict((f'average_sbert_embedding_{index+1}', embedding_part) for index, embedding_part in enumerate(average_embedding))
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

    def get_ratio_of_unique_unigrams(self, chunk):
        return len(chunk.unigram_counts.keys()) / sum(chunk.unigram_counts.values())

    def get_ratio_of_unique_bigrams(self, chunk):
        return len(chunk.bigram_counts.keys()) / sum(chunk.bigram_counts.values())

    def get_ratio_of_unique_trigrams(self, chunk):
        return len(chunk.trigram_counts.keys()) / sum(chunk.trigram_counts.values())

    def get_nr_chars(self, chunk):
        return len(chunk.text)
    
    def get_nr_words(self, chunk):
        return len(chunk.text.split())
    
    def get_longest_word_length(self, chunk):
        word_lengths = []
        for word, count in chunk.unigram_counts.items():
            word_lengths.append(len(word))
        return max(word_lengths)

    def get_average_word_length(self, chunk):
        word_lengths = []
        for word, count in chunk.unigram_counts.items():
            word_lengths.append(len(word) * count)
        return np.mean(word_lengths)

    def get_unigram_entropy(self, chunk):
        return entropy(list(chunk.unigram_counts.values()))
    
    def __get_ngram_entropy(self, chunks, ntype):
        # Calculate nr of trigrams in the text
        # Trigram counts from CountVectorizer are filtered and do not contain all trigrams, but only those that occur in in several documents
        
        # text = ' '.join([chunk.text for chunk in chunks])
        # # nr_tokens = len(text.split())

        # # nr_trigrams = nr_tokens- 2


        # cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=(3,3), dtype=np.int32)
        # dtm = cv.fit_transform([text])
        # print(dtm.shape)
        # nr_trigrams = dtm.sum()
        # print('nr trigrams', nr_trigrams)
        nc = NgramCounter(self.language)
        data_dict = nc.load_data(file_name=f'{ntype}-full')
        dtm = data_dict['dtm']

        # Calculate the probabilities of each trigram
        trigram_probs = dtm.sum(axis=0) / dtm.sum()

        # Convert probabilities to a 1D array
        trigram_probs = np.asarray(trigram_probs).ravel()

        # Calculate the entropy for each trigram based on its probability
        # trigram_entropies = [-p * np.log2(p) if p > 0 else 0 for p in trigram_probs]
        trigram_entropies = entropy(trigram_probs)

        # Calculate the overall trigram entropy for the entire corpus
        corpus_entropy = np.mean(trigram_entropies)

        print(f"Trigram Entropy for the Document: {corpus_entropy:.4f}")


    def get_bigram_entropy(self, chunks):
        return self.__get_ngram_entropy(chunks, 'bigram')
    
    def get_trigram_entropy(self, chunks):
        return self.__get_ngram_entropy(chunks, 'trigram')
    
    def get_type_token_ratio(self, chunk):
        # Type-token ratio according to Algee-Hewitt et al. (2016)
        tokens = sum(chunk.unigram_counts.values())
        types = len(chunk.unigram_counts)
        return types/tokens

    def get_flesch_reading_ease_score(self, chunk):
        return textstat.flesch_reading_ease(chunk.text)

    # def get_gunning_fog(self, chunk):
    #     '''''''''
    #     Not implemented for German. If we can find 'easy words' in German, then we can implement it ourselves.
    #     '''
    #     return textstat.gunning_fog(chunk.text)

    # book-based features
    def __get_intra_textual_variance(self, chunks, embedding_type):
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
        return self.__get_intra_textual_variance(chunks, 'd2v')

    def get_sbert_intra_textual_variance(self, chunks):
        return self.__get_intra_textual_variance(chunks, 'sbert')

    def __get_stepwise_distance(self, chunks, embedding_type):
        if len(chunks) == 1:
            return 0
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
            print('Norm:\n', np.linalg.norm(current_chunk_embedding - previous_chunk_embedding))
            euclidean_distances.append(np.linalg.norm(current_chunk_embedding - previous_chunk_embedding))
        print('Mean\n: ', np.mean(euclidean_distances))

        # Calculate the mean by dividing by n-1
        n = len(euclidean_distances)
        mean_distance = np.sum(euclidean_distances) / (n - 1)
        return mean_distance

    def get_d2v_stepwise_distance(self, chunks):
        return self.__get_stepwise_distance(chunks, 'd2v')

    def get_sbert_stepwise_distance(self, chunks):
        return self.__get_stepwise_distance(chunks, 'sbert')
