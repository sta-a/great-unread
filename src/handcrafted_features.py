import os
import spacy
from process import SentenceTokenizer
from sentence_transformers import SentenceTransformer
from unidecode import unidecode
from scipy.stats import entropy, kurtosis, skew
import numpy as np
import textstat
import string
import pickle
import re


class Chunk(object):
    def __init__(self, sentences):
        self.sentences = sentences
        self.raw_text = " ".join(sentences)
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


class FeatureExtractor(object):
    def __init__(self, lang, doc_path, sentences_per_chunk=500):
        self.lang = lang
        self.sentences_per_chunk = sentences_per_chunk
        self.doc_path = doc_path

        if self.lang == "eng":
            self.model_name = 'en_core_web_sm'
            self.sentence_encoder = SentenceTransformer('stsb-mpnet-base-v2')
        elif self.lang == "ger":
            self.model_name = 'de_core_news_sm'
            self.sentence_encoder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
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
        
        if os.path.exists(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle"):
            self.sentences = pickle.load(open(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle", "rb"))
        else:
            self.sentence_tokenizer = SentenceTokenizer(self.lang)
            self.sentences = self.sentence_tokenizer.tokenize(self.raw_text)
            os.makedirs("/".join(doc_path[:-4].split("/")[:-1]).replace("/raw_docs", f"/processed_sentences"), exist_ok=True)
            pickle.dump(self.sentences, open(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle", "wb"))
        
        self.chunks = self.__get_chunks()
        
    def __get_chunks(self):
        if self.sentences_per_chunk is None:
            return [Chunk(self.sentences)]
        sentence_chunks = []
        for i in range(0, len(self.sentences), self.sentences_per_chunk):
            current_sentences = self.sentences[i:i+self.sentences_per_chunk]
            if len(current_sentences) == self.sentences_per_chunk:
                sentence_chunks.append(Chunk(current_sentences))
        return sentence_chunks
    
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
        features = {
            "ratio_of_punctuation_marks": self.get_ratio_of_punctuation_marks,
            "ratio_of_whitespaces": self.get_ratio_of_whitespaces,
            "ratio_of_digits": self.get_ratio_of_digits,
            "ratio_of_exclamation_marks": self.get_ratio_of_exclamation_marks,
            "ratio_of_question_marks": self.get_ratio_of_question_marks,
            "ratio_of_commas": self.get_ratio_of_commas,
            "ratio_of_uppercase_letters": self.get_ratio_of_uppercase_letters,
            "get_average_number_of_words_in_sentence": self.get_average_number_of_words_in_sentence,
            "get_maximum_number_of_words_in_sentence": self.get_maximum_number_of_words_in_sentence,
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
            "number_of_indentations": self.get_number_of_indentations, # structural features
            "average_sentence_embedding": self.get_average_sentence_embedding, # instead of using word2vec/doc2vec, topic modeling, content specific features used this.
            # skipped greetings since this is not e-mail(structural features)
            # skipped types of signature since this is not e-mail(structural features)
            # skipped content specific features. added BERT average sentence embedding instead.
            
            #######
        }
        data = []
        for chunk in self.chunks:
            current_features = {"book_name": self.doc_path.split("/")[-1][:-4]}
            for feature_name, feature_function in features.items():
                current_features[feature_name] = feature_function(chunk)
            data.append(current_features)
        return data
    
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
    
    def get_number_of_indentations(self, chunk):
        return chunk.raw_text.count("\t")
    
    def get_average_sentence_embedding(self, chunk):
        embeddings = self.sentence_encoder.encode(chunk.sentences)
        return embeddings.mean(axis=0)
    
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