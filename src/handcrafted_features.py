import os
import spacy
from process import SentenceTokenizer
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from unidecode import unidecode
from multiprocessing import cpu_count
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from gensim.models import LdaMulticore
from gensim.matutils import Sparse2Corpus
from sklearn.utils import shuffle
from tqdm import tqdm
from scipy.stats import entropy, kurtosis, skew
import numpy as np
import pandas as pd
import textstat
import string
import re
import logging
from collections import Counter
from utils import load_list_of_lines, save_list_of_lines
logging.basicConfig(level=logging.DEBUG)


class Chunk(object):
    def __init__(self, sentences, sbert_sentence_embeddings, doc2vec_chunk_embedding):
        self.sentences = sentences
        self.sbert_sentence_embeddings = sbert_sentence_embeddings
        self.raw_text = " ".join(sentences)
        self.doc2vec_chunk_embedding = doc2vec_chunk_embedding
        self.unidecoded_raw_text = unidecode(self.raw_text)
        self.processed_sentences = self.__preprocess_sentences()
        self.word_unigram_counts = self.__find_word_unigram_counts()
        self.word_bigram_counts = self.__find_word_bigram_counts()
        self.word_trigram_counts = self.__find_word_trigram_counts()
        self.char_unigram_counts = self.__find_char_unigram_counts()
    
    def __preprocess_sentences(self):
        def __preprocess_sentences_helper(text):
            text = text.lower()
            text = unidecode(text)
            text = re.sub("[^a-zA-Z]+", " ", text).strip()
            text = text.split()
            text = " ".join(text)
            return text
        return [__preprocess_sentences_helper(sentence) for sentence in self.sentences]
    
    def __find_word_unigram_counts(self):
        word_unigram_counts = {}
        for processed_sentence in self.processed_sentences:
            for word_unigram in processed_sentence.split():
                if word_unigram in word_unigram_counts.keys():
                    word_unigram_counts[word_unigram] += 1
                else:
                    word_unigram_counts[word_unigram] = 1
        return word_unigram_counts

    def __find_word_bigram_counts(self):
        processed_text = "<BOS> " + " <EOS> <BOS> ".join(self.processed_sentences) + " <EOS>"
        splitted_processed_text = processed_text.split()
        word_bigram_counts = {}
        for i in range(len(splitted_processed_text) - 1):
            current_word_bigram = (splitted_processed_text[i], splitted_processed_text[i+1])
            if current_word_bigram in word_bigram_counts:
                word_bigram_counts[current_word_bigram] += 1
            else:
                word_bigram_counts[current_word_bigram] = 1
        return word_bigram_counts

    def __find_word_trigram_counts(self):
        processed_text = "<BOS> <BOS> " + " <EOS> <EOS> <BOS> <BOS> ".join(self.processed_sentences) + " <EOS> <EOS>"
        splitted_processed_text = processed_text.split()
        word_trigram_counts = {}
        for i in range(len(splitted_processed_text) - 2):
            current_word_trigram = (splitted_processed_text[i], splitted_processed_text[i+1], splitted_processed_text[i+2])
            if current_word_trigram in word_trigram_counts.keys():
                word_trigram_counts[current_word_trigram] += 1
            else:
                word_trigram_counts[current_word_trigram] = 1
        return word_trigram_counts

    def __find_char_unigram_counts(self):
        char_unigram_counts = {}
        for character in self.unidecoded_raw_text:
            if character in char_unigram_counts.keys():
                char_unigram_counts[character] += 1
            else:
                char_unigram_counts[character] = 1
        return char_unigram_counts


class Doc2VecChunkVectorizer(object):
    def __init__(self,
                 lang,
                 sentences_per_chunk=500,
                 dm=1,
                 dm_mean=1,
                 seed=42,
                 n_cores=-1):
        self.lang = lang
        if lang == "eng":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif lang == "ger":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        else:
            raise Exception(f"Not a valid language {lang}")
        self.sentences_per_chunk = sentences_per_chunk
        self.dm = dm
        self.dm_mean = dm_mean
        self.seed = seed
        if n_cores == -1 or n_cores is None:
            self.n_cores = cpu_count()
        else:
            self.n_cores = n_cores
    
    def fit_transform(self, doc_paths):
        tagged_chunks = []
        chunk_id_counter = 0
        doc_path_to_chunk_ids = {}
        logging.info("Preparing data for Doc2VecChunkVectorizer...")
        for doc_id, doc_path in tqdm(list(enumerate(doc_paths))):
            sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
            if os.path.exists(sentences_path):
                self.sentences = load_list_of_lines(sentences_path, "str")
            else:
                self.sentence_tokenizer = SentenceTokenizer(self.lang)
                self.sentences = self.sentence_tokenizer.tokenize(doc_path)
                save_list_of_lines(self.sentences, sentences_path, "str")

            if self.sentences_per_chunk is None:
                words = self.tokenizer.tokenize(" ".join(self.sentences))
                tagged_chunks.append(TaggedDocument(words=words, tags=[f'chunk_{chunk_id_counter}']))
                if doc_path in doc_path_to_chunk_ids.keys():
                    doc_path_to_chunk_ids[doc_path].append(chunk_id_counter)
                else:
                    doc_path_to_chunk_ids[doc_path] = [chunk_id_counter]
                chunk_id_counter += 1
            else:
                for i in range(0, len(self.sentences), self.sentences_per_chunk):
                    current_sentences = self.sentences[i:i+self.sentences_per_chunk]
                    if (len(current_sentences) == self.sentences_per_chunk) or (i == 0):
                        words = self.tokenizer.tokenize(" ".join(current_sentences))
                        tagged_chunks.append(TaggedDocument(words=words, tags=[f'chunk_{chunk_id_counter}']))
                        if doc_path in doc_path_to_chunk_ids.keys():
                            doc_path_to_chunk_ids[doc_path].append(chunk_id_counter)
                        else:
                            doc_path_to_chunk_ids[doc_path] = [chunk_id_counter]
                        chunk_id_counter += 1
        logging.info("Prepared data for Doc2VecChunkVectorizer.")
        
        logging.info("Fitting Doc2VecChunkVectorizer...")
        self.d2v_model = Doc2Vec(shuffle(tagged_chunks),
                                 window=10,
                                 dm=self.dm,
                                 dm_mean=self.dm_mean,
                                 workers=self.n_cores,
                                 seed=self.seed)
        logging.info("Fitted Doc2VecChunkVectorizer.")
        
        logging.info("Saving chunk vectors...")
        for doc_path in doc_paths:
            chunk_vectors = [self.d2v_model.dv[f'chunk_{chunk_id}'] for chunk_id in doc_path_to_chunk_ids[doc_path]]
            save_list_of_lines(chunk_vectors, doc_path.replace("/raw_docs", f"/processed_doc2vec_chunk_embeddings_spc_{self.sentences_per_chunk}"), "np")
        logging.info("Saved chunk vectors.")


class DocBasedFeatureExtractor(object):
    def __init__(self, lang, doc_path, sentences_per_chunk=500):
        self.lang = lang
        self.sentences_per_chunk = sentences_per_chunk
        self.doc_path = doc_path

        if self.lang == "eng":
            self.model_name = 'en_core_web_sm'
        elif self.lang == "ger":
            self.model_name = 'de_core_news_sm'
        else:
            raise Exception(f"Not a valid language {self.lang}")
        
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            logging.info(f"Downloading {self.model_name} for Spacy...")
            os.system(f"python3 -m spacy download {self.model_name}")
            logging.info(f"Downloaded {self.model_name} for Spacy.")
            self.nlp = spacy.load(self.model_name)
        
        self.stopwords = self.nlp.Defaults.stop_words
        
        ## load sentences
        sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
        if os.path.exists(sentences_path):
            self.sentences = load_list_of_lines(sentences_path, "str")
        else:
            self.sentence_tokenizer = SentenceTokenizer(self.lang)
            self.sentences = self.sentence_tokenizer.tokenize(doc_path)
            save_list_of_lines(self.sentences, sentences_path, "str")
        
        ## load sbert sentence embeddings
        sbert_sentence_embeddings_path = doc_path.replace("/raw_docs", f"/processed_sbert_sentence_embeddings") + ".npz"
        if os.path.exists(sbert_sentence_embeddings_path):
            self.sbert_sentence_embeddings = load_list_of_lines(sbert_sentence_embeddings_path, "np")
        else:
            if self.lang == "eng":
                self.sentence_encoder = SentenceTransformer('stsb-mpnet-base-v2')
            elif self.lang == "ger":
                self.sentence_encoder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
            self.sbert_sentence_embeddings = list(self.sentence_encoder.encode(self.sentences))
            save_list_of_lines(self.sbert_sentence_embeddings, sbert_sentence_embeddings_path, "np")
        
        ## load doc2vec chunk embeddings
        doc2vec_chunk_embeddings_path = doc_path.replace("/raw_docs", f"/processed_doc2vec_chunk_embeddings_spc_{sentences_per_chunk}") + ".npz"
        if os.path.exists(doc2vec_chunk_embeddings_path):
            self.doc2vec_chunk_embeddings = load_list_of_lines(doc2vec_chunk_embeddings_path, "np")
        else:
            raise Exception(f"Could not find Doc2Vec chunk embeddings for chunk size {self.sentences_per_chunk}.")
        
        self.chunks = self.__get_chunks()
        
    def __get_chunks(self):
        if self.sentences_per_chunk is None:
            return [Chunk(self.sentences, self.sbert_sentence_embeddings, self.sbert_sentence_embeddings)]
        chunks = []
        chunk_id_counter = 0
        for i in range(0, len(self.sentences), self.sentences_per_chunk):
            current_sentences = self.sentences[i:i+self.sentences_per_chunk]
            current_sentence_embeddings = self.sbert_sentence_embeddings[i:i+self.sentences_per_chunk]
            if (len(current_sentences) == self.sentences_per_chunk) or (i == 0):
                chunks.append(Chunk(current_sentences, current_sentence_embeddings, self.doc2vec_chunk_embeddings[chunk_id_counter]))
                chunk_id_counter += 1
        return chunks
    
    # def get_statistics_of_list_features(self, list_of_features, feature_name):
    #     statistics = {
    #         "max": np.max,
    #         "min": np.min,
    #         "mean": np.mean,
    #         "std": np.std,
    #         "entropy": entropy,
    #         "kurtosis": kurtosis,
    #         "skewness": skew
    #     }
    #     sorted_list_of_features = sorted(list_of_features)
    #     features = {}
    #     for statistic_name, statistic_function in statistics.items():
    #         features[f"{feature_name}_{statistic_name}"] = statistic_function(sorted_list_of_features)
    #     return features
    
    def get_all_features(self):
        chunk_based_feature_mapping = {
            "ratio_of_punctuation_marks": self.get_ratio_of_punctuation_marks,
            "ratio_of_whitespaces": self.get_ratio_of_whitespaces,
            "ratio_of_digits": self.get_ratio_of_digits,
            "ratio_of_exclamation_marks": self.get_ratio_of_exclamation_marks,
            "ratio_of_question_marks": self.get_ratio_of_question_marks,
            "ratio_of_commas": self.get_ratio_of_commas,
            "ratio_of_uppercase_letters": self.get_ratio_of_uppercase_letters,
            "average_number_of_words_in_sentence": self.get_average_number_of_words_in_sentence,
            "maximum_number_of_words_in_sentence": self.get_maximum_number_of_words_in_sentence,
            "ratio_of_unique_word_unigrams": self.get_ratio_of_unique_word_unigrams,
            "ratio_of_unique_word_bigrams": self.get_ratio_of_unique_word_bigrams,
            "ratio_of_unique_word_trigrams": self.get_ratio_of_unique_word_trigrams,
            "text_length": self.get_text_length,
            "average_word_length": self.get_average_word_length,
            "ratio_of_stopwords": self.get_ratio_of_stopwords,
            "bigram_entropy": self.get_word_bigram_entropy,
            "trigram_entropy": self.get_word_trigram_entropy,

            ## Features in the list
            
            "flesch_reading_ease_score": self.get_flesch_reading_ease_score, # readability
            "unigram_entropy": self.get_word_unigram_entropy, # second order redundancy
            "average_paragraph_length": self.get_average_paragraph_length, # structural features
            0: self.get_average_sbert_sentence_embedding,
            1: self.get_doc2vec_chunk_embedding,
            # skipped greetings since this is not e-mail(structural features)
            # skipped types of signature since this is not e-mail(structural features)
            # skipped content specific features. added BERT average sentence embedding instead.

            #######
        }
        
        book_based_feature_mapping = {
            "doc2vec_intra_textual_variance": self.get_doc2vec_intra_textual_variance,
            "sbert_intra_textual_variance": self.get_sbert_intra_textual_variance,
            "doc2vec_stepwise_distance": self.get_doc2vec_stepwise_distance,
            "sbert_stepwise_distance": self.get_sbert_stepwise_distance
        }
        
        # extract chunk based features
        chunk_based_features = []
        for chunk in self.chunks:
            current_features = {"book_name": self.doc_path.split("/")[-1][:-4]}
            for feature_name, feature_function in chunk_based_feature_mapping.items():
                if isinstance(feature_name, int):
                    current_features.update(feature_function(chunk))
                else:
                    current_features[feature_name] = feature_function(chunk)
            chunk_based_features.append(current_features)
        
        # extract book based features
        book_based_features = {}
        for feature_name, feature_function in book_based_feature_mapping.items():
            book_based_features["book_name"] = self.doc_path.split("/")[-1][:-4]
            book_based_features[feature_name] = feature_function(self.chunks)
        
        return chunk_based_features, \
               book_based_features, \
               [np.array(chunk.sbert_sentence_embeddings).mean(axis=0) for chunk in self.chunks], \
               [chunk.doc2vec_chunk_embedding for chunk in self.chunks]
    
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
        for char in chunk.unidecoded_raw_text:
            if char.isalpha():
                num_alpha += 1
                if char.isupper():
                    num_upper += 1
        return num_upper / num_alpha
    
    def get_average_paragraph_length(self, chunk):
        splitted_lengths = [len(splitted) for splitted in chunk.raw_text.split("\n")]
        return np.mean(splitted_lengths)
    
    def get_average_sbert_sentence_embedding(self, chunk):
        average_sentence_embedding = np.array(chunk.sbert_sentence_embeddings).mean(axis=0)
        average_sentence_embedding_features = dict((f"average_sentence_embedding_{index+1}", embedding_part) for index, embedding_part in enumerate(average_sentence_embedding))
        return average_sentence_embedding_features
    
    def get_doc2vec_chunk_embedding(self, chunk):
        doc2vec_chunk_embedding_features = dict((f"doc2vec_chunk_embedding_{index+1}", embedding_part) for index, embedding_part in enumerate(chunk.doc2vec_chunk_embedding))
        return doc2vec_chunk_embedding_features
    
    def get_average_number_of_words_in_sentence(self, chunk):
        sentence_lengths = []
        for processed_sentence in chunk.processed_sentences:
            sentence_lengths.append(len(processed_sentence.split()))
        return np.mean(sentence_lengths)

    def get_maximum_number_of_words_in_sentence(self, chunk):
        sentence_lengths = []
        for processed_sentence in chunk.processed_sentences:
            sentence_lengths.append(len(processed_sentence.split()))
        return np.max(sentence_lengths)
    
    def get_ratio_of_unique_word_unigrams(self, chunk):
        return len(chunk.word_unigram_counts.keys()) / sum(chunk.word_unigram_counts.values())
    
    def get_ratio_of_unique_word_bigrams(self, chunk):
        return len(chunk.word_bigram_counts.keys()) / sum(chunk.word_bigram_counts.values())
    
    def get_ratio_of_unique_word_trigrams(self, chunk):
        return len(chunk.word_trigram_counts.keys()) / sum(chunk.word_trigram_counts.values())
    
    def get_text_length(self, chunk):
        return len(chunk.unidecoded_raw_text)
    
    def get_average_word_length(self, chunk):
        word_lengths = []
        for word, count in chunk.word_unigram_counts.items():
            word_lengths.append(len(word) * count)
        return np.mean(word_lengths)
    
    def get_ratio_of_stopwords(self, chunk):
        number_of_stopwords = 0
        number_of_all_words = 0
        for word, count in chunk.word_unigram_counts.items():
            number_of_all_words += 1
            if word in self.stopwords:
                number_of_stopwords += count
        return number_of_stopwords / number_of_all_words
        
    def get_word_unigram_entropy(self, chunk):
        return entropy(list(chunk.word_unigram_counts.values()))
    
    def get_word_bigram_entropy(self, chunk):
        temp_dict = {}
        for word_bigram, count in chunk.word_bigram_counts.items():
            left = word_bigram[0]
            if left in temp_dict.keys():
                temp_dict[left].append(count)
            else:
                temp_dict[left] = [count]
        entropies = []
        for left, counts in temp_dict.items():
            entropies.append(entropy(counts))
        return np.mean(entropies)
    
    def get_word_trigram_entropy(self, chunk):
        temp_dict = {}
        for word_trigram, count in chunk.word_trigram_counts.items():
            left_and_middle = (word_trigram[0], word_trigram[1])
            if left_and_middle in temp_dict.keys():
                temp_dict[left_and_middle].append(count)
            else:
                temp_dict[left_and_middle] = [count]
        entropies = []
        for left_and_middle, counts in temp_dict.items():
            entropies.append(entropy(counts))
        return np.mean(entropies)
    
    def get_flesch_reading_ease_score(self, chunk):
        return textstat.flesch_reading_ease(chunk.unidecoded_raw_text)
    
    def get_gunning_fog(self, chunk):
        """
        Not implemented for German. If we can find "easy words" in German, then we can implement it ourselves.
        """
        return textstat.gunning_fog(chunk.unidecoded_raw_text)
    
    ######
    ## book based features
    ######
    def __get_intra_textual_variance(self, chunks, embedding_type):
        chunk_embeddings = []
        for chunk in chunks:
            if embedding_type == "doc2vec":
                chunk_embeddings.append(chunk.doc2vec_chunk_embedding)
            elif embedding_type == "sbert":
                chunk_embeddings.append(np.array(chunk.sbert_sentence_embeddings).mean(axis=0))
            else:
                raise Exception(f"Not a valid embedding type {embedding_type}")
        average_chunk_embedding = np.array(chunk_embeddings).mean(axis=0)
        euclidean_distances = [np.linalg.norm(average_chunk_embedding - chunk_embedding) for chunk_embedding in chunk_embeddings]
        return np.mean(euclidean_distances)
    
    def get_doc2vec_intra_textual_variance(self, chunks):
        return self.__get_intra_textual_variance(chunks, "doc2vec")
    
    def get_sbert_intra_textual_variance(self, chunks):
        return self.__get_intra_textual_variance(chunks, "sbert")
    
    def __get_stepwise_distance(self, chunks, embedding_type):
        euclidean_distances = []
        for chunk_idx in range(1, len(chunks)):
            if embedding_type == "doc2vec":
                current_chunk_embedding = chunks[chunk_idx].doc2vec_chunk_embedding
                previous_chunk_embedding = chunks[chunk_idx - 1].doc2vec_chunk_embedding
            elif embedding_type == "sbert":
                current_chunk_embedding = np.array(chunks[chunk_idx].sbert_sentence_embeddings).mean(axis=0)
                previous_chunk_embedding = np.array(chunks[chunk_idx - 1].sbert_sentence_embeddings).mean(axis=0)
            else:
                raise Exception(f"Not a valid embedding type {embedding_type}")
            euclidean_distances.append(np.linalg.norm(current_chunk_embedding - previous_chunk_embedding))
        return np.mean(euclidean_distances)
        
    def get_doc2vec_stepwise_distance(self, chunks):
        return self.__get_stepwise_distance(chunks, "doc2vec")
    
    def get_sbert_stepwise_distance(self, chunks):
        return self.__get_stepwise_distance(chunks, "sbert")
  

class CorpusBasedFeatureExtractor(object):
    def __init__(self, lang, doc_paths, all_average_sbert_sentence_embeddings, all_doc2vec_chunk_embeddings):
        self.lang = lang
        self.doc_paths = doc_paths
        self.word_statistics = self.__get_word_statistics()
        self.all_average_sbert_sentence_embeddings = all_average_sbert_sentence_embeddings
        self.all_doc2vec_chunk_embeddings = all_doc2vec_chunk_embeddings
        
        if self.lang == "eng":
            self.model_name = 'en_core_web_sm'
        elif self.lang == "ger":
            self.model_name = 'de_core_news_sm'
        else:
            raise Exception(f"Not a valid language {self.lang}")
        
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            logging.info(f"Downloading {self.model_name} for Spacy...")
            os.system(f"python3 -m spacy download {self.model_name}")
            logging.info(f"Downloaded {self.model_name} for Spacy.")
            self.nlp = spacy.load(self.model_name)
        self.stopwords = self.nlp.Defaults.stop_words
    
    def __find_word_unigram_counts(self, processed_sentences):
        word_unigram_counts = {}
        for processed_sentence in processed_sentences:
            for word_unigram in processed_sentence.split():
                if word_unigram in word_unigram_counts.keys():
                    word_unigram_counts[word_unigram] += 1
                else:
                    word_unigram_counts[word_unigram] = 1
        return word_unigram_counts

    def __find_word_bigram_counts(self, processed_sentences):
        processed_text = "<BOS> " + " <EOS> <BOS> ".join(processed_sentences) + " <EOS>"
        splitted_processed_text = processed_text.split()
        word_bigram_counts = {}
        for i in range(len(splitted_processed_text) - 1):
            current_word_bigram = splitted_processed_text[i] + " " + splitted_processed_text[i+1]
            if current_word_bigram in word_bigram_counts:
                word_bigram_counts[current_word_bigram] += 1
            else:
                word_bigram_counts[current_word_bigram] = 1
        return word_bigram_counts

    def __find_word_trigram_counts(self, processed_sentences):
        processed_text = "<BOS> <BOS> " + " <EOS> <EOS> <BOS> <BOS> ".join(processed_sentences) + " <EOS> <EOS>"
        splitted_processed_text = processed_text.split()
        word_trigram_counts = {}
        for i in range(len(splitted_processed_text) - 2):
            current_word_trigram = splitted_processed_text[i] + " " + splitted_processed_text[i+1] + " " + splitted_processed_text[i+2]
            if current_word_trigram in word_trigram_counts.keys():
                word_trigram_counts[current_word_trigram] += 1
            else:
                word_trigram_counts[current_word_trigram] = 1
        return word_trigram_counts
    
    def __preprocess_sentences(self, sentences):
        def __preprocess_sentences_helper(text):
            text = text.lower()
            text = unidecode(text)
            text = re.sub("[^a-zA-Z]+", " ", text).strip()
            text = text.split()
            text = " ".join(text)
            return text
        return [__preprocess_sentences_helper(sentence) for sentence in sentences]
    
    def __get_word_statistics(self):
        all_word_unigram_counts = Counter()
        all_word_bigram_counts = Counter()
        all_word_trigram_counts = Counter()
        
        for doc_path in tqdm(self.doc_paths):
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            processed_sentences = self.__preprocess_sentences(sentences)
            unigram_counts = self.__find_word_unigram_counts(processed_sentences)
            bigram_counts = self.__find_word_bigram_counts(processed_sentences)
            trigram_counts = self.__find_word_trigram_counts(processed_sentences)
            all_word_unigram_counts.update(unigram_counts)
            all_word_bigram_counts.update(bigram_counts)
            all_word_trigram_counts.update(trigram_counts)
        
        all_word_unigram_counts = dict(sorted(list(all_word_unigram_counts.items()), key=lambda x: -x[1])[:2000])
        all_word_bigram_counts = dict(sorted(list(all_word_bigram_counts.items()), key=lambda x: -x[1])[:2000])
        all_word_trigram_counts = dict(sorted(list(all_word_trigram_counts.items()), key=lambda x: -x[1])[:2000])
        
        book_name_word_unigram_mapping = {}
        book_name_word_bigram_mapping = {}
        book_name_word_trigram_mapping = {}
        for doc_path in tqdm(self.doc_paths):
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            processed_sentences = self.__preprocess_sentences(sentences)
            unigram_counts = self.__find_word_unigram_counts(processed_sentences)
            bigram_counts = self.__find_word_bigram_counts(processed_sentences)
            trigram_counts = self.__find_word_trigram_counts(processed_sentences)
            total_unigram_count = sum(unigram_counts.values())
            total_bigram_count = sum(bigram_counts.values())
            total_trigram_count = sum(trigram_counts.values())
            #relative frequencies
            book_name_word_unigram_mapping[book_name] = dict((unigram, count / total_unigram_count) for unigram, count in unigram_counts.items() if unigram in all_word_unigram_counts.keys())
            book_name_word_bigram_mapping[book_name] = dict((bigram, count / total_bigram_count) for bigram, count in bigram_counts.items() if bigram in all_word_bigram_counts.keys())
            book_name_word_trigram_mapping[book_name] = dict((trigram, count / total_trigram_count) for trigram, count in trigram_counts.items() if trigram in all_word_trigram_counts.keys())     

        word_statistics = {
            "all_word_unigram_counts": all_word_unigram_counts,
            "all_word_bigram_counts": all_word_bigram_counts,
            "all_word_trigram_counts": all_word_trigram_counts,
            "book_name_word_unigram_mapping": book_name_word_unigram_mapping,
            "book_name_word_bigram_mapping": book_name_word_bigram_mapping,
            "book_name_word_trigram_mapping": book_name_word_trigram_mapping
        }
        return word_statistics

    def get_k_most_common_word_ngram_counts(self, k, n, include_stopwords):
        if n == 1:
            dct1 = self.word_statistics["all_word_unigram_counts"]
            dct2 = self.word_statistics["book_name_word_unigram_mapping"]
        elif n == 2:
            dct1 = self.word_statistics["all_word_bigram_counts"]
            dct2 = self.word_statistics["book_name_word_bigram_mapping"]
        elif n == 3:
            dct1 = self.word_statistics["all_word_trigram_counts"]
            dct2 = self.word_statistics["book_name_word_trigram_mapping"]
        else:
            raise Exception(f"Not a valid n: {n}")
        if include_stopwords:
            most_common_k_ngrams = [ngram for ngram, count in sorted(list(dct1.items()), key=lambda x: -x[1])[:k]]
        else:
            filtered_ngrams = []
            for ngram, count in dct1.items():
                splitted_ngram = ngram.split()
                exclude = False
                for word in splitted_ngram:
                    if word in self.stopwords:
                        exclude = True
                if exclude:
                    continue
                else:
                    filtered_ngrams.append((ngram, count))
            most_common_k_ngrams = [ngram for ngram, count in sorted(filtered_ngrams, key=lambda x: -x[1])[:k]]
        result = []
        for book_name, ngram_counts in dct2.items():
            dct = dict((common_ngram, dct2[book_name].get(common_ngram, 0)) for common_ngram in most_common_k_ngrams)
            dct["book_name"] = book_name
            result.append(dct)
        return pd.DataFrame(result)
    
    def get_50_most_common_word_unigram_counts_including_stopwords(self):
        return self.get_k_most_common_word_ngram_counts(50, 1, True)
        
    def get_50_most_common_word_bigram_counts_including_stopwords(self):
        return self.get_k_most_common_word_ngram_counts(50, 2, True)
        
    def get_50_most_common_word_trigram_counts_including_stopwords(self):
        return self.get_k_most_common_word_ngram_counts(50, 3, True)
    
    def get_50_most_common_word_unigram_counts_excluding_stopwords(self):
        return self.get_k_most_common_word_ngram_counts(50, 1, False)
        
    def get_50_most_common_word_bigram_counts_excluding_stopwords(self):
        return self.get_k_most_common_word_ngram_counts(50, 2, False)
        
    def get_50_most_common_word_trigram_counts_excluding_stopwords(self):
        return self.get_k_most_common_word_ngram_counts(50, 3, False)
    
    def get_overlap_score(self, embedding_type):
        if embedding_type == "doc2vec":
            all_embeddings = self.all_doc2vec_chunk_embeddings
        elif embedding_type == "sbert":
            all_embeddings = self.all_average_sbert_sentence_embeddings
        else:
            raise Exception(f"Not a valid embedding_type {embedding_type}.")
    
        cluster_means = []
        for index, current_list_of_embeddings in enumerate(all_embeddings):
            cluster_means.append(np.array(current_list_of_embeddings).mean(axis=0))

        labels = []
        predictions = []
        for label_index, current_list_of_embeddings in tqdm(list(enumerate(all_embeddings))):
            for current_embedding in current_list_of_embeddings:
                labels.append(label_index)
                best_cluster = None
                smallest_distance = np.inf
                for prediction_index, cluster_mean in enumerate(cluster_means):
                    current_distance = np.linalg.norm(current_embedding - cluster_mean)
                    if current_distance < smallest_distance:
                        smallest_distance = current_distance
                        best_cluster = prediction_index
                predictions.append(best_cluster)
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        book_names = []
        overlap_scores = []
        for label_index, doc_path in enumerate(self.doc_paths):
            book_name = doc_path.split("/")[-1][:-4]
            indices = np.argwhere(labels == label_index).ravel()
            current_predictions = predictions[indices]
            incorrect_prediction_indices = np.argwhere(current_predictions != label_index)
            overlap_score = len(incorrect_prediction_indices) / len(current_predictions)
            book_names.append(book_name)
            overlap_scores.append(overlap_score)
        return pd.DataFrame.from_dict({"book_name": book_names, f"overlap_score_{embedding_type}": overlap_scores})
    
    def get_overlap_score_doc2vec(self):
        return self.get_overlap_score("doc2vec")
    
    def get_overlap_score_sbert(self):
        return self.get_overlap_score("sbert")
    
    def get_outlier_score(self, embedding_type):
        if embedding_type == "doc2vec":
            all_embeddings = self.all_doc2vec_chunk_embeddings
        elif embedding_type == "sbert":
            all_embeddings = self.all_average_sbert_sentence_embeddings
        else:
            raise Exception(f"Not a valid embedding_type {embedding_type}.")
    
        cluster_means = []
        for index, current_list_of_embeddings in enumerate(all_embeddings):
            cluster_means.append(np.array(current_list_of_embeddings).mean(axis=0))
        
        outlier_scores = []
        book_names = []
        for current_index, current_cluster_mean in enumerate(cluster_means):
            doc_path = self.doc_paths[current_index]
            book_name = doc_path.split("/")[-1][:-4]
            nearest_distance = np.inf
            for other_index, other_cluster_mean in enumerate(cluster_means):
                if current_index == other_index:
                    continue
                current_distance = np.linalg.norm(current_cluster_mean - other_cluster_mean)
                if current_distance < nearest_distance:
                    nearest_distance = current_distance
            outlier_scores.append(nearest_distance)
            book_names.append(book_name)
        return pd.DataFrame.from_dict({"book_name": book_names, f"outlier_score_{embedding_type}": outlier_scores})
    
    def get_outlier_score_doc2vec(self):
        return self.get_outlier_score("doc2vec")
    
    def get_outlier_score_sbert(self):
        return self.get_outlier_score("sbert")
    
    def get_lda_topic_distribution(self):
        num_topics = 10

        documents = []
        for doc_path in self.doc_paths:
            with open(doc_path, "r") as reader:
                documents.append(reader.read().strip())

        if self.lang == "eng":
            stop_words = "english"
        elif self.lang == "ger":
            stop_words = "german"
        else:
            raise Exception(f"Not a valid language {self.lang}")

        vect = CountVectorizer(min_df=20, max_df=0.2, stop_words=stop_words, 
                               token_pattern='(?u)\\b\\w\\w\\w+\\b')
        X = vect.fit_transform(documents)
        corpus = Sparse2Corpus(X, documents_columns=False)
        id_map = dict((v, k) for k, v in vect.vocabulary_.items())
        lda_model = LdaMulticore(corpus=corpus, id2word=id_map, passes=2, random_state=42, num_topics=num_topics, workers=3)

        topic_distributions = []
        book_names = []
        for doc_path, document in zip(self.doc_paths, documents):
            book_name = doc_path.split("/")[-1][:-4]
            string_input = [document]
            X = vect.transform(string_input)
            corpus = Sparse2Corpus(X, documents_columns=False)
            output = list(lda_model[corpus])[0]
            full_output = [0] * num_topics
            for topic_id, ratio in output:
                full_output[topic_id] = ratio
            topic_distributions.append(full_output)
            book_names.append(book_name)
        topic_distributions = pd.DataFrame(topic_distributions, columns=[f"lda_topic_{i+1}" for i in range(num_topics)])
        topic_distributions["book_name"] = book_names
        return topic_distributions

    def get_tfidf(self):
        # use preprocessed sentences instead of raw docs because they are preprocessed
        # get absolute word counts
        processed_sents_paths = [path.replace("/raw_docs", f"/processed_sentences") for path in self.doc_paths]
        book_names = [path.split("/")[-1][:-4] for path in processed_sents_paths]
        vect = TfidfVectorizer(input='filename')
        X = vect.fit_transform(processed_sents_paths)
        X = pd.DataFrame(X.toarray(), columns = vect.get_feature_names())
        X['book_name'] = book_names
        print(X['the'], X['and'])
        return X

    def get_all_features(self):
        result = None
        for feature_function in [#self.get_50_most_common_word_unigram_counts_including_stopwords,
                                 #self.get_50_most_common_word_bigram_counts_including_stopwords,
                                #  self.get_50_most_common_word_trigram_counts_including_stopwords,
                                #  self.get_50_most_common_word_unigram_counts_excluding_stopwords,
                                #  self.get_50_most_common_word_bigram_counts_excluding_stopwords,
                                #  self.get_50_most_common_word_trigram_counts_excluding_stopwords,
                                #  self.get_overlap_score_doc2vec,
                                #  self.get_overlap_score_sbert,
                                #  self.get_outlier_score_doc2vec,
                                #  self.get_outlier_score_sbert,
                                #  self.get_lda_topic_distribution,
                                 self.get_tfidf]:
            if result is None:
                result = feature_function()
            else:
                result = result.merge(feature_function(), on="book_name")
        return result