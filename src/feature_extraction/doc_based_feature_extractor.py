import os
import numpy as np
import string
import textstat
from pathlib import Path
from scipy.stats import entropy
import sys
sys.path.append("..")
from .process_rawtext import Tokenizer
from .embeddings import SbertProcessor, D2vProcessor
from .analyze_chunk import Chunk
import sys
sys.path.append("..")
from utils import load_list_of_lines, save_list_of_lines, get_bookname, DataHandler


class DocBasedFeatureExtractor():
    '''
    Extract features that can be generated from a single document.
    '''
    def __init__(self, 
        language,
        doc_path, 
        dvs, # Document vectors from doc2vec for all documents
        use_chunks,
        tokens_per_chunk=500,
        unigram_counts=True, 
        bigram_counts=True, 
        trigram_counts=True, 
        char_unigram_counts=True):



        self.language = language
        self.doc_path = doc_path
        self.dvs = dvs
        self.use_chunks = use_chunks
        self.tokens_per_chunk = tokens_per_chunk
        self.file_name = get_bookname(self.doc_path)
        # self.doc_dvs = self.load_d2v_embeddings()
        self.sbert_embeddings = self.load_sbert_embeddings()
        # Parameters for creating chunks
        self.unigram_counts = unigram_counts
        self.bigram_counts = bigram_counts
        self.trigram_counts = trigram_counts
        self.char_unigram_counts = char_unigram_counts

        # Preprocess or load data
        self.text_tokenized = Tokenizer(self.language).create_data(doc_path, remove_punct=False, lower=False, as_chunk=True)

        self.chunks = self.__get_chunks()


    # def load_d2v_embeddings(self):
    #     doc_dvs = {}
    #     for key, vector in self.dvs:
    #             if self.use_chunks == True:
    #                 if key.startswith(self.file_name + '_'):
    #                     doc_dvs[key] = vector
    #             else:
    #                 doc_dvs[self.file_name] = self.dvs[self.file_name]
    #     print(doc_dvs.keys())
    #     return doc_dvs
    
    def load_sbert_embeddings(self):
        sbert = SbertProcessor(self.language).load_data(file_name=self.file_name + '.npz') ########
        if self.use_chunks == False:
            # If whole document is used, combine the embeddings of the chunks into one 
            all_sbert = []
            for chunk_id in sbert.keys():
                all_sbert.append(sbert[chunk_id]) #####################3
            sbert  = all_sbert
        return sbert

    def __get_chunks(self):

        if self.use_chunks == False:
            return [Chunk(
                tokens_per_chunk = self.tokens_per_chunk,
                doc_path = self.doc_path,
                chunk_id = 'None',
                text_tokenized = ' '.join(self.text_tokenized),
                sbert_embeddings = self.sbert_embeddings,
                d2v_chunk_embedding = self.dvs[self.file_name], ####################33
                unigram_counts = self.unigram_counts,
                bigram_counts = self.bigram_counts,
                trigram_counts = self.trigram_counts,
                char_unigram_counts = self.char_unigram_counts)]

        else:
            chunks = []
            chunk_id_counter = 0
            for curr_words in self.text_tokenized:
                #current_sentence_embeddings = None #self.sbert_embeddings[i:i+self.tokens_per_chunk] ###############################3
                chunks.append(Chunk(
                    tokens_per_chunk = self.tokens_per_chunk,
                    doc_path = self.doc_path,
                    chunk_id = chunk_id_counter,
                    text_tokenized = curr_words,
                    sbert_embeddings = self.sbert_embeddings[chunk_id_counter],
                    d2v_chunk_embedding = self.dvs[f'{self.file_name}_{str(chunk_id_counter)}'],
                    unigram_counts = self.unigram_counts,
                    bigram_counts = self.bigram_counts,
                    trigram_counts = self.trigram_counts,
                    char_unigram_counts = self.char_unigram_counts))
                chunk_id_counter += 1
            return chunks


    def get_all_features(self):
        chunk_feature_mapping = {
            'ratio_of_punctuation_marks': self.get_ratio_of_punctuation_marks,
            'ratio_of_whitespaces': self.get_ratio_of_whitespaces,
            #'ratio_of_digits': self.get_ratio_of_digits,
            'ratio_of_exclamation_marks': self.get_ratio_of_exclamation_marks,
            'ratio_of_question_marks': self.get_ratio_of_question_marks,
            'ratio_of_commas': self.get_ratio_of_commas,
            'ratio_of_uppercase_letters': self.get_ratio_of_uppercase_letters,
            'average_number_of_words_in_sentence': self.get_average_number_of_words_in_sentence,
            'maximum_number_of_words_in_sentence': self.get_maximum_number_of_words_in_sentence,
            'ratio_of_unique_unigrams': self.get_ratio_of_unique_unigrams,
            'ratio_of_unique_bigrams': self.get_ratio_of_unique_bigrams,
            'ratio_of_unique_trigrams': self.get_ratio_of_unique_trigrams,
            'text_length': self.get_text_length,
            'average_word_length': self.get_average_word_length,
            'bigram_entropy': self.get_bigram_entropy,
            'trigram_entropy': self.get_trigram_entropy,
            'type_token_ratio': self.get_type_token_ratio,
            'flesch_reading_ease_score': self.get_flesch_reading_ease_score,
            'unigram_entropy': self.get_unigram_entropy, # second order redundancy
            # 'average_paragraph_length': self.get_average_paragraph_length, # structural features
            # 0: self.get_average_sbert_sentence_embedding, 
            1: self.get_d2v_chunk_embedding
        }

        book_feature_mapping = {
            'd2v_intra_textual_variance': self.get_d2v_intra_textual_variance,
            # 'sbert_intra_textual_variance': self.get_sbert_intra_textual_variance,
            'd2v_stepwise_distance': self.get_d2v_stepwise_distance,
            # 'sbert_stepwise_distance': self.get_sbert_stepwise_distance
        }

        # extract chunk based features
        chunk_features = []
        for chunk in self.chunks:
            if self.use_chunks == False:
                chunk_name = chunk.file_name + '_' + str(chunk.chunk_id)
            else:
                chunk_name = chunk.file_name
            current_features = {'file_name': chunk_name}
            for feature_name, feature_function in chunk_feature_mapping.items():
                if isinstance(feature_name, int):
                    current_features.update(feature_function(chunk))
                else:
                    current_features[feature_name] = feature_function(chunk)
            chunk_features.append(current_features)

        # extract book based features
        book_features = None
        if self.use_chunks == True:
            book_features = {}
            for feature_name, feature_function in book_feature_mapping.items():
                book_features['file_name'] = self.doc_path.split('/')[-1][:-4]
                book_features[feature_name] = feature_function(self.chunks)

        #Return sbert embeddings by averageing across sentences belonging to a chunk #########################33
        return chunk_features, \
                book_features


    def get_ratio_of_punctuation_marks(self, chunk):
        punctuations = 0
        for character in string.punctuation:
            punctuations += chunk.char_unigram_counts.get(character, 0)
        all_characters = sum(list(chunk.char_unigram_counts.values()))
        return punctuations / all_characters

    def get_ratio_of_digits(self, chunk):
        digits = 0
        all_characters = 0
        for character in [str(i) for i in range(10)]:
            digits += chunk.char_unigram_counts.get(character, 0)
        all_characters = sum(list(chunk.char_unigram_counts.values()))
        return digits / all_characters

    def get_ratio_of_whitespaces(self, chunk):
        return chunk.char_unigram_counts.get(' ', 0) / sum(list(chunk.char_unigram_counts.values()))

    def get_ratio_of_exclamation_marks(self, chunk):
        return chunk.char_unigram_counts.get('!', 0) / sum(list(chunk.char_unigram_counts.values()))

    def get_ratio_of_question_marks(self, chunk):
        return chunk.char_unigram_counts.get('?', 0) / sum(list(chunk.char_unigram_counts.values()))

    def get_ratio_of_commas(self, chunk):
        return chunk.char_unigram_counts.get(',', 0) / sum(list(chunk.char_unigram_counts.values()))

    def get_ratio_of_uppercase_letters(self, chunk):
        num_upper = 0
        num_alpha = 0
        for char in chunk.text_tokenized:
            if char.isalpha():
                num_alpha += 1
                if char.isupper():
                    num_upper += 1
        return num_upper / num_alpha

    # def get_average_paragraph_length(self, chunk):
    # Doesn't work because some text have poem_like structure, for example 'Ainsworth_William-Harrison_Rookwood_1834.txt'
    #     with open(self.doc_path, 'r') as f:
    #         raw_text = f.read()
    #     split_lengths = [len(curr_split) for curr_split in chunk.raw_text.split('\n')]
    #     return np.mean(split_lengths)

    def get_average_sbert_sentence_embedding(self, chunk):
        average_sentence_embedding = np.array(chunk.sbert_embeddings).mean(axis=0)
        average_sentence_embedding_features = dict((f'average_sentence_embedding_{index+1}', embedding_part) for index, embedding_part in enumerate(average_sentence_embedding))
        return average_sentence_embedding_features

    def get_d2v_chunk_embedding(self, chunk):
        d2v_chunk_embedding_features = dict((f'd2v_chunk_embedding_{index+1}', embedding_part) for index, embedding_part in enumerate(chunk.d2v_chunk_embedding))
        return d2v_chunk_embedding_features

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

    def get_text_length(self, chunk):
        return len(chunk.text_tokenized)

    def get_average_word_length(self, chunk):
        word_lengths = []
        for word, count in chunk.unigram_counts.items():
            word_lengths.append(len(word) * count)
        return np.mean(word_lengths)

    def get_unigram_entropy(self, chunk):
        return entropy(list(chunk.unigram_counts.values()))

    def get_bigram_entropy(self, chunk):
        temp_dict = {}
        for bigram, count in chunk.bigram_counts.items():
            bigram = bigram.split()
            left = bigram[0]
            if left in temp_dict.keys():
                temp_dict[left].append(count)
            else:
                temp_dict[left] = [count]
        entropies = []
        for left, counts in temp_dict.items():
            entropies.append(entropy(counts))
        return np.mean(entropies)

    def get_trigram_entropy(self, chunk):
        temp_dict = {}
        for trigram, count in chunk.trigram_counts.items():
            trigram = trigram.split()
            left_and_middle = trigram[0] + ' ' + trigram[1]
            if left_and_middle in temp_dict.keys():
                temp_dict[left_and_middle].append(count)
            else:
                temp_dict[left_and_middle] = [count]
        entropies = []
        for left_and_middle, counts in temp_dict.items():
            entropies.append(entropy(counts))
        return np.mean(entropies)

    def get_type_token_ratio(self, chunk):
        # Type-token ratio according to Algee-Hewitt et al. (2016)
        tokens = sum(chunk.unigram_counts.values())
        types = len(chunk.unigram_counts)
        return types/tokens

    def get_flesch_reading_ease_score(self, chunk):
        return textstat.flesch_reading_ease(chunk.text_tokenized)

    # def get_gunning_fog(self, chunk):
    #     '''''''''
    #     Not implemented for German. If we can find 'easy words' in German, then we can implement it ourselves.
    #     '''
    #     return textstat.gunning_fog(chunk.text_tokenized)

    # book-based features
    def __get_intra_textual_variance(self, chunks, embedding_type):
        chunk_embeddings = []
        for chunk in chunks:
            if embedding_type == 'd2v':
                chunk_embeddings.append(chunk.d2v_chunk_embedding)
            # elif embedding_type == 'sbert':
            #     chunk_embeddings.append(np.array(chunk.sbert_embeddings).mean(axis=0)) 
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
            #print('index', chunk_idx)
            if embedding_type == 'd2v':
                current_chunk_embedding = chunks[chunk_idx].d2v_chunk_embedding
                previous_chunk_embedding = chunks[chunk_idx - 1].d2v_chunk_embedding
            # elif embedding_type == 'sbert':
            #     current_chunk_embedding = np.array(chunks[chunk_idx].sbert_embeddings).mean(axis=0)
            #     previous_chunk_embedding = np.array(chunks[chunk_idx - 1].sbert_embeddings).mean(axis=0)
            else:
                raise Exception(f'Not a valid embedding type {embedding_type}')
            #print('Norm:\n', np.linalg.norm(current_chunk_embedding - previous_chunk_embedding))
            euclidean_distances.append(np.linalg.norm(current_chunk_embedding - previous_chunk_embedding))
        #print('Mean\n: ', np.mean(euclidean_distances))
        return np.mean(euclidean_distances)

    def get_d2v_stepwise_distance(self, chunks):
        return self.__get_stepwise_distance(chunks, 'd2v')

    # def get_sbert_stepwise_distance(self, chunks):
    #     return self.__get_stepwise_distance(chunks, 'sbert')
