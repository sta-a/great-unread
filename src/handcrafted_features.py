import os
import spacy
from process import SentenceTokenizer
from unidecode import unidecode
from scipy.stats import entropy
import numpy as np
import textstat
import string
import pickle
import re


class FeatureExtractor(object):
    def __init__(self, lang, doc_path):
        self.lang = lang
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
        self.doc_path = doc_path
        with open(doc_path, "r") as reader:
            self.raw_text = reader.read()
        self.unidecoded_raw_text = unidecode(self.raw_text)
        self.sentence_tokenizer = SentenceTokenizer(self.lang)
        if os.path.exists(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle"):
            self.sentences = pickle.load(open(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle", "rb"))
        else:
            self.sentences = self.sentence_tokenizer.tokenize(self.raw_text)
            os.makedirs("/".join(doc_path[:-4].split("/")[:-1]).replace("/raw_docs", f"/processed_sentences"), exist_ok=True)
            pickle.dump(self.sentences, open(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle", "wb"))

        self.processed_sentences = [self.__preprocess_text(sentence) for sentence in self.sentences]
        self.word_unigram_counts = self.__find_word_unigram_counts()
        self.word_bigram_counts = self.__find_word_bigram_counts()
        self.word_trigram_counts = self.__find_word_trigram_counts()
        self.char_unigram_counts = self.__find_char_unigram_counts()

    def __preprocess_text(self, text):
        text = text.lower()
        text = unidecode(text)
        text = re.sub("[^a-zA-Z]+", " ", text).strip()
        text = text.split()
        text = " ".join(text)
        return text

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
    
    def get_all_features(self):
        return {
            "ratio_of_punctuation_marks": self.get_ratio_of_punctuation_marks(),
            "ratio_of_digits": self.get_ratio_of_digits(),
            "ratio_of_exclamation_marks": self.get_ratio_of_exclamation_marks(),
            "ratio_of_question_marks": self.get_ratio_of_question_marks(),
            "ratio_of_commas": self.get_ratio_of_commas(),
            "ratio_of_uppercase_letters": self.get_ratio_of_uppercase_letters(),
            "average_number_of_words_in_sentence": self.get_average_number_of_words_in_sentence(),
            "maximum_number_of_words_in_sentence": self.get_maximum_number_of_words_in_sentence(),
            "ratio_of_unique_word_unigrams": self.get_ratio_of_unique_word_unigrams(),
            "ratio_of_unique_word_bigrams": self.get_ratio_of_unique_word_bigrams(),
            "ratio_of_unique_word_trigrams": self.get_ratio_of_unique_word_trigrams(),
            "text_length": self.get_text_length(),
            "number_of_sentences": self.get_number_of_sentences(),
            "average_word_length": self.get_average_word_length(),
            "ratio_of_stopwords": self.get_ratio_of_stopwords(),
            "unigram_entropy": self.get_word_unigram_entropy(), # redundancy
            "bigram_entropy": self.get_word_bigram_entropy(),
            "trigram_entropy": self.get_word_trigram_entropy(),
            "flesch_reading_ease_score": self.get_flesch_reading_ease_score() # readability
        }
    
    def get_ratio_of_punctuation_marks(self):
        punctuations = 0
        for character in string.punctuation:
            punctuations += self.char_unigram_counts.get(character, 0)
        all_characters = sum(list(self.char_unigram_counts.values()))
        return punctuations / all_characters
    
    def get_ratio_of_digits(self):
        digits = 0
        all_characters = 0
        for character in [str(i) for i in range(10)]:
            digits += self.char_unigram_counts.get(character, 0)
        all_characters = sum(list(self.char_unigram_counts.values()))
        return digits / all_characters
    
    def get_ratio_of_exclamation_marks(self):
        return self.char_unigram_counts.get('!', 0) / sum(list(self.char_unigram_counts.values()))
    
    def get_ratio_of_question_marks(self):
        return self.char_unigram_counts.get('?', 0) / sum(list(self.char_unigram_counts.values()))
    
    def get_ratio_of_commas(self):
        return self.char_unigram_counts.get(',', 0) / sum(list(self.char_unigram_counts.values()))
    
    def get_ratio_of_uppercase_letters(self):
        num_upper = 0
        num_alpha = 0
        for char in self.unidecoded_raw_text:
            if char.isalpha():
                num_alpha += 1
                if char.isupper():
                    num_upper += 1
        return num_upper / num_alpha
    
    def get_average_number_of_words_in_sentence(self):
        sentence_lengths = []
        for processed_sentence in self.processed_sentences:
            sentence_lengths.append(len(processed_sentence.split()))
        return np.mean(sentence_lengths)

    def get_maximum_number_of_words_in_sentence(self):
        sentence_lengths = []
        for processed_sentence in self.processed_sentences:
            sentence_lengths.append(len(processed_sentence.split()))
        return np.max(sentence_lengths)
    
    def get_ratio_of_unique_word_unigrams(self):
        return len(self.word_unigram_counts.keys()) / sum(self.word_unigram_counts.values())
    
    def get_ratio_of_unique_word_bigrams(self):
        return len(self.word_bigram_counts.keys()) / sum(self.word_bigram_counts.values())
    
    def get_ratio_of_unique_word_trigrams(self):
        return len(self.word_trigram_counts.keys()) / sum(self.word_trigram_counts.values())
    
    def get_text_length(self):
        return len(self.unidecoded_raw_text)
    
    def get_number_of_sentences(self):
        return len(self.processed_sentences)
    
    def get_average_word_length(self):
        word_lengths = []
        for word, count in self.word_unigram_counts.items():
            word_lengths.append(len(word) * count)
        return np.mean(word_lengths)
    
    def get_ratio_of_stopwords(self):
        number_of_stopwords = 0
        number_of_all_words = 0
        for word, count in self.word_unigram_counts.items():
            number_of_all_words += 1
            if word in self.stopwords:
                number_of_stopwords += count
        return number_of_stopwords / number_of_all_words
        
    def get_word_unigram_entropy(self):
        return entropy(list(self.word_unigram_counts.values()))
    
    def get_word_bigram_entropy(self):
        temp_dict = {}
        for word_bigram, count in self.word_bigram_counts.items():
            left = word_bigram[0]
            if left in temp_dict.keys():
                temp_dict[left].append(count)
            else:
                temp_dict[left] = [count]
        entropies = []
        for left, counts in temp_dict.items():
            entropies.append(entropy(counts))
        return np.mean(entropies)
    
    def get_word_trigram_entropy(self):
        temp_dict = {}
        for word_trigram, count in self.word_trigram_counts.items():
            left_and_middle = (word_trigram[0], word_trigram[1])
            if left_and_middle in temp_dict.keys():
                temp_dict[left_and_middle].append(count)
            else:
                temp_dict[left_and_middle] = [count]
        entropies = []
        for left_and_middle, counts in temp_dict.items():
            entropies.append(entropy(counts))
        return np.mean(entropies)
    
    def get_flesch_reading_ease_score(self):
        return textstat.flesch_reading_ease(self.unidecoded_raw_text)
    
    def get_gunning_fog(self):
        """
        Not implemented for German. If we can find "easy words" in German, then we can implement it ourselves.
        """
        return textstat.gunning_fog(self.unidecoded_raw_text)