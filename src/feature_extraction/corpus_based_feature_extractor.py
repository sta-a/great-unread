import os
import spacy
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import LdaMulticore
from gensim.matutils import Sparse2Corpus
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import pickle
from collections import Counter
from utils import load_list_of_lines, unidecode_custom
from corpus_toolkit import corpus_tools as ct
logging.basicConfig(level=logging.DEBUG)


class CorpusBasedFeatureExtractor(object):
    def __init__(self, lang, doc_paths, all_average_sbert_sentence_embeddings, all_doc2vec_chunk_embeddings):
        self.lang = lang
        self.doc_paths = doc_paths
        self.word_statistics = self.__get_word_statistics()
        self.all_average_sbert_sentence_embeddings = all_average_sbert_sentence_embeddings
        self.all_doc2vec_chunk_embeddings = all_doc2vec_chunk_embeddings
        self.book_name_processed_sentences_mapping = None

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

        if self.lang == "eng":
            self.spell_checker = SpellChecker(language='en')
        elif self.lang == "ger":
            self.spell_checker = SpellChecker(language='de')
        else:
            raise Exception(f"Not a valid language {self.lang}")

        self.stopwords = self.nlp.Defaults.stop_words
        new_stopwords = []
        for stopword in self.stopwords:
            new_stopwords.append(unidecode_custom(stopword))
        self.stopwords = set(new_stopwords)

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
            text = unidecode_custom(text)
            text = re.sub("[^a-zA-Z]+", " ", text).strip()
            text = text.split()
            text = " ".join(text)
            return text
        return [__preprocess_sentences_helper(sentence) for sentence in sentences]

    def __apply_spacy_nlp_to_sentences(self, sentences):
        return [self.nlp(sentence) for sentence in sentences]

    def __apply_spacy_nlp_to_docs(self):
        for doc_path in self.doc_paths:
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", "/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            processed_sentences = self.__apply_spacy_nlp_to_sentences(sentences, "str")
            self.book_name_processed_sentences_mapping[book_name] = processed_sentences

    def __tag_books(self, tag_type, gram_type, k):
        def __tag_sentence(doc, tag_type, gram_type):
            if tag_type == "pos":
                tokens_unigram = [token.pos_ for token in doc]
            elif tag_type == "tag":
                tokens_unigram = [token.tag_ for token in doc]
            elif tag_type == "dep":
                tokens_unigram = [token.dep_ for token in doc]
            else:
                raise Exception("Not a valid tag_type")

            if gram_type == "unigram":
                return tokens_unigram
            elif gram_type == "bigram":
                tokens_bigram_temp = ["BOS"] + tokens_unigram + ["EOS"]
                tokens_bigram = ["_".join([tokens_bigram_temp[i], tokens_bigram_temp[i+1]]) for i in range(len(tokens_bigram_temp)-1)]
                return tokens_bigram
            elif gram_type == "trigram":
                tokens_trigram_temp = ["BOS", "BOS"] + tokens_unigram + ["EOS", "EOS"]
                tokens_trigram = ["_".join([tokens_trigram_temp[i], tokens_trigram_temp[i+1], tokens_trigram_temp[i+2]]) for i in range(len(tokens_trigram_temp)-2)]
                return tokens_trigram
            else:
                raise Exception("Not a valid gram_type")

        def __tag_book(processed_sentences, tag_type, gram_type):
            book_tag_counter = Counter()
            for processed_sentence in processed_sentences:
                sentence_tags = __tag_sentence(processed_sentence, tag_type, gram_type)
                book_tag_counter.update(sentence_tags)
            return book_tag_counter

        # to process sentences for only one time
        if self.book_name_processed_sentences_mapping is None:
            self.__apply_spacy_nlp_to_docs()

        tagged_books = {}
        corpus_tag_counter = Counter()
        for book_name, processed_sentences in self.book_name_processed_sentences_mapping.items():
            book_tag_counter = __tag_book(processed_sentences, tag_type, gram_type)
            tagged_books[book_name] = book_tag_counter
            corpus_tag_counter.update(book_tag_counter)
        
        # get first k tags of corpus_tag_counter
        corpus_tag_counter = sorted([(tag, count) for tag, count in corpus_tag_counter.items()], key=lambda x: -x[1])[:k]
        corpus_tag_counter = list(corpus_tag_counter.keys())
        
        data = []
        
        # get first k tags of each book_tag_counter
        for book_name, tagged_book in tagged_books.items():
            current_books_chosen_tag_counts = dict([(tag_type + "_" + gram_type + "_" + tag_name, tagged_book[tag_name]) for tag_name in corpus_tag_counter])
            current_books_chosen_tag_counts_sum = sum([count for tag, count in current_books_chosen_tag_counts.items()])
            current_books_chosen_tag_counts = dict([(tag, count/current_books_chosen_tag_counts_sum) for tag, count in current_books_chosen_tag_counts.items()])
            current_books_chosen_tag_counts["book_name"] = book_name
            data.append(current_books_chosen_tag_counts)
        
        df = pd.DataFrame(data)
        return df

    def get_tag_distribution(self, k):
        result_df = None
        for tag_type in ['pos']: # ['pos', 'tag', 'dep']:
            for gram_type in ['unigram', 'bigram', 'trigram']:
                current_df = self.__tag_books(self.doc_paths, tag_type, gram_type, k)
                if result_df is None:
                    result_df = current_df
                else:
                    result_df = result_df.merge(current_df, on='book_name')
        return result_df
    
    def get_spelling_error_distribution(self):
        def __get_spelling_error_count_in_sentence(self, sentence):
            misspelled = self.spell_checker.unknown(sentence.split())
            return len(misspelled)
        
        def __get_spelling_error_rate_in_book(book):
            error_counter = sum([__get_spelling_error_count_in_sentence(sentence) for sentence in book])
            error_rate = error_counter / len(book)
            return error_rate
        
        data = []
        for doc_path in self.doc_paths:
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", "/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            error_rate = __get_spelling_error_rate_in_book(sentences)
            data.append({"book_name": book_name,
                         "error_rate": error_rate})
        df = pd.DataFrame(data)
        return df

    def __get_book_production_counts(self, book, pre):
        book_production_counter = Counter()
        for sentence in book:
            sentence_production_counter = pre.get_sentence_production_counter(sentence)
            book_production_counter.update(sentence_production_counter)
        return book_production_counter

    def get_production_distribution(self, k):
        """
        Returns an empty dataframe if the language is German. Reason is explained in
        docstring of ProductionRuleExtractor.
        """
        if self.lang == "ger":
            return pd.DataFrame()
        elif self.lang == "eng":
            pass
        else:
            raise Exception("Not a valid language")

        pre = ProductionRuleExtractor()
        book_production_counters = {}
        corpus_production_counter = Counter()

        for doc_path in self.doc_paths:
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", "/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            book_production_counter = self.__get_book_production_counts(sentences, pre)
            book_production_counters[book_name] = book_production_counter
            corpus_production_counter.update(book_production_counter)
        
        # get first k tags of corpus_tag_counter
        corpus_production_counter = sorted([(tag, count) for tag, count in corpus_production_counter.items()], key=lambda x: -x[1])[:k]
        corpus_production_counter = list(corpus_production_counter.keys())
        
        data = []
        
        # get first k tags of each book_tag_counter
        for book_name, book_prodution_counter in book_production_counters.items():
            current_books_chosen_production_counts = dict([(production_type, book_prodution_counter[production_type]) for production_type in corpus_production_counter])
            current_books_chosen_production_counts_sum = sum([count for tag, count in current_books_chosen_production_counts.items()])
            current_books_chosen_production_counts = dict([(tag, count/current_books_chosen_production_counts_sum) for tag, count in current_books_chosen_production_counts.items()])
            current_books_chosen_production_counts["book_name"] = book_name
            data.append(current_books_chosen_production_counts)
        
        df = pd.DataFrame(data)
        return df
    
    def __get_word_statistics(self):
        # get total counts over all documents
        all_word_unigram_counts = Counter()
        book_name_abs_word_unigram_mapping = {}
        book_name_rel_word_unigram_mapping = {}

        for doc_path in tqdm(self.doc_paths):
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            processed_sentences = self.__preprocess_sentences(sentences)
            unigram_path = doc_path.replace("/raw_docs", f"/unigram_counts")
            if os.path.isfile(unigram_path):
                with open(unigram_path, 'rb') as f:
                    unigram_counts = pickle.load(f)
            else:
                unigram_counts = self.__find_word_unigram_counts(processed_sentences)
                with open(unigram_path, 'wb') as f:
                    pickle.dump(unigram_counts, f)
            all_word_unigram_counts.update(unigram_counts)
            # get (relative) counts per document
            total_unigram_count = sum(unigram_counts.values())
            #absolute frequency
            book_name_abs_word_unigram_mapping[book_name] = unigram_counts
            #relative frequencies
            book_name_rel_word_unigram_mapping[book_name] = dict((unigram, count / total_unigram_count) for unigram, count in unigram_counts.items()) #all words
        
        all_word_unigram_counts = dict(sorted(list(all_word_unigram_counts.items()), key=lambda x: -x[1])) #all words
        word_statistics = {
            "all_word_unigram_counts": all_word_unigram_counts,
            "book_name_abs_word_unigram_mapping": book_name_abs_word_unigram_mapping,
            "book_name_rel_word_unigram_mapping": book_name_rel_word_unigram_mapping,
        }
        return word_statistics

    def __add_bigrams_trigrams_word_statistics(self):
        # get total counts over all documents
        all_word_bigram_counts = Counter()
        all_word_trigram_counts = Counter()
        
        for doc_path in tqdm(self.doc_paths):
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            processed_sentences = self.__preprocess_sentences(sentences)
            bigram_counts = self.__find_word_bigram_counts(processed_sentences)
            trigram_counts = self.__find_word_trigram_counts(processed_sentences)
            all_word_bigram_counts.update(bigram_counts)
            all_word_trigram_counts.update(trigram_counts)
        
        all_word_bigram_counts = dict(sorted(list(all_word_bigram_counts.items()), key=lambda x: -x[1])[:2000])
        all_word_trigram_counts = dict(sorted(list(all_word_trigram_counts.items()), key=lambda x: -x[1])[:2000])
        
        # get (relative) counts per document
        book_name_word_bigram_mapping = {}
        book_name_word_trigram_mapping = {}
        for doc_path in tqdm(self.doc_paths):
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            processed_sentences = self.__preprocess_sentences(sentences)
            bigram_counts = self.__find_word_bigram_counts(processed_sentences)
            trigram_counts = self.__find_word_trigram_counts(processed_sentences)
            total_bigram_count = sum(bigram_counts.values())
            total_trigram_count = sum(trigram_counts.values())
            #relative frequencies
            book_name_word_bigram_mapping[book_name] = dict((bigram, count / total_bigram_count) for bigram, count in bigram_counts.items() if bigram in all_word_bigram_counts.keys())
            book_name_word_trigram_mapping[book_name] = dict((trigram, count / total_trigram_count) for trigram, count in trigram_counts.items() if trigram in all_word_trigram_counts.keys())
        
        word_stat_dir = self.word_statistics
        word_stat_dir["all_word_bigram_counts"] = all_word_bigram_counts
        word_stat_dir["all_word_trigram_counts"] = all_word_trigram_counts
        word_stat_dir["book_name_word_bigram_mapping"] = book_name_word_bigram_mapping
        word_stat_dir["book_name_word_trigram_mapping"] = book_name_word_trigram_mapping
        return word_stat_dir

    def get_k_most_common_word_ngram_counts(self, k, n, include_stopwords):
        if n == 1:
            dct1 = self.word_statistics["all_word_unigram_counts"]
            dct2 = self.word_statistics["book_name_rel_word_unigram_mapping"]
        elif n == 2 or n == 3:
            self.word_statistics = self.__add_bigrams_trigrams_word_statistics()
            if n == 2:
                dct1 = self.word_statistics["all_word_bigram_counts"]
                dct2 = self.word_statistics["book_name_word_bigram_mapping"]
            else:
                dct1 = self.word_statistics["all_word_trigram_counts"]
                dct2 = self.word_statistics["book_name_word_trigram_mapping"]
        else:
            raise Exception(f"Not a valid n: {n}")
        if include_stopwords:
            # words that are the most common in the whole corpus
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
            # get freq of k most common n-grams in current document
            dct = dict((f"{k}_most_common_{n}gram_stopword_{include_stopwords}_{common_ngram}", dct2[book_name].get(common_ngram, 0)) for common_ngram in most_common_k_ngrams)
            dct["book_name"] = book_name
            result.append(dct)
        result = pd.DataFrame(result)
        return result
    
    def get_k_most_common_word_unigram_counts_including_stopwords(self, k):
        return self.get_k_most_common_word_ngram_counts(k, 1, True)
        
    def get_k_most_common_word_bigram_counts_including_stopwords(self, k):
        return self.get_k_most_common_word_ngram_counts(k, 2, True)
        
    def get_k_most_common_word_trigram_counts_including_stopwords(self, k):
        return self.get_k_most_common_word_ngram_counts(k, 3, True)
    
    def get_k_most_common_word_unigram_counts_excluding_stopwords(self, k):
        return self.get_k_most_common_word_ngram_counts(k, 1, False)
        
    def get_k_most_common_word_bigram_counts_excluding_stopwords(self, k):
        return self.get_k_most_common_word_ngram_counts(k, 2, False)
        
    def get_k_most_common_word_trigram_counts_excluding_stopwords(self, k):
        return self.get_k_most_common_word_ngram_counts(k, 3, False)
    
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
            stop_words = spacy.lang.en.stop_words.STOP_WORDS
        elif self.lang == "ger":
            stop_words = spacy.lang.de.stop_words.STOP_WORDS
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

    def get_tfidf(self, k=50):
        document_term_matrix = pd.DataFrame.from_dict(self.word_statistics['book_name_abs_word_unigram_mapping']).fillna(0).T
        # Tfidf
        t = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True)
        tfidf = pd.DataFrame.sparse.from_spmatrix(t.fit_transform(document_term_matrix), columns=document_term_matrix.columns, index=document_term_matrix.index)
        # Keep only those words which occur in more than 10% of documents
        document_frequency = document_term_matrix.astype(bool).sum(axis=0)
        min_nr_documents = round(0.1 * tfidf.shape[0])
        reduced_columns = [document_term_matrix.columns[x] for x in range(0,len(document_term_matrix.columns)) if document_frequency[x]>min_nr_documents]
        tfidf_reduced = tfidf[reduced_columns]
        # From remaining words, keep only those that are in the top k for at least one book
        all_top_k_words = []
        for index, row in tfidf_reduced.iterrows():
            top_k_words = row.nlargest(n=k, keep='all')
            all_top_k_words.extend(top_k_words.index.to_list())
        all_top_k_words = list(set(all_top_k_words))
        tfidf_top_k = tfidf_reduced[all_top_k_words]

        tfidf_top_k.columns = [f"tfidf_{column}" for column in tfidf_top_k.columns]
        tfidf_top_k = tfidf_top_k.reset_index().rename(columns={'level_0':'book_name', 'index':'book_name'}) #automatically created column name can be 'index' or 'level_0'
        return tfidf_top_k
        '''
        # remove preceding string from column names, only keep word
        words_to_return = [column[len("50_most_common_1gram_stopword_False_"):] for column in self.get_k_most_common_word_unigram_counts_excluding_stopwords(k).columns if column != "book_name"]
        print(len(words_to_return), words_to_return)
        processed_sents_paths = [path.replace("/raw_docs", f"/processed_sentences") for path in self.doc_paths]
        book_names = [path.split("/")[-1][:-4] for path in processed_sents_paths]
        vect = TfidfVectorizer(input='filename')
        X = vect.fit_transform(processed_sents_paths)
        X = pd.DataFrame(X.toarray(), columns = vect.get_feature_names())
        X = X.loc[:, set(words_to_return).intersection(set(X.columns))] #error here, excludes random words
        print(X.shape)
        X.columns = [f"tfidf_{column}" for column in X.columns]
        X['book_name'] = book_names

        return X
        '''

    def get_keyness(self, k):
        # evaluate keyness in current book compared to whole corpus
        # "%diff" is best keyness metric (according to Gabrielatos & Marchi, 2011)
        book_name_word_keyness_mapping = {}
        corpus_unigram_counts = self.word_statistics["all_word_unigram_counts"]
        for book_name, book_unigram_counts in self.word_statistics["book_name_abs_word_unigram_mapping"].items(): 
            #reference corpus is the word frequencies in all documents except the current one
            reference_corpus = {key: corpus_unigram_counts[key] - book_unigram_counts.get(key, 0) for key in corpus_unigram_counts.keys()}
            book_keyness = ct.keyness(book_unigram_counts, reference_corpus, effect='%diff')
            book_name_word_keyness_mapping[book_name] = book_keyness
        keyness_df = pd.DataFrame(book_name_word_keyness_mapping).T

        #Keep only those words that are in at least 10% of documents and which are in the top k for at least one book
        document_frequency = keyness_df.astype(bool).sum(axis=0)
        min_nr_documents = round(0.1 * keyness_df.shape[0])
        reduced_columns = [keyness_df.columns[x] for x in range(0,len(keyness_df.columns)) if document_frequency[x] > min_nr_documents]
        keyness_df_reduced = keyness_df[reduced_columns]
        # From remaining words, keep only those that are in the top k for at least one book
        all_top_k_words = []
        for index, row in keyness_df_reduced.iterrows():
            top_k_words = row.nlargest(n=k, keep='all')
            all_top_k_words.extend(top_k_words.index.to_list())
        all_top_k_words = list(set(all_top_k_words))
        keyness_df_top_k = keyness_df_reduced[all_top_k_words]
        #return book_name_word_keyness_mapping
        keyness_df_top_k.columns = [f"keyness_{column}" for column in keyness_df_top_k.columns]
        keyness_df_top_k = keyness_df_top_k.reset_index().rename(columns={'level_0':'book_name', 'index':'book_name'}) #automatically created column name can be 'index' or 'level_0'
        return keyness_df_top_k

    def get_all_features(self, k=100):
        ''' Get corpus-based features

        Args:
            k (int): number of features to return

        Returns: 
            pd.DataFrame of corpus-based features
        '''
        result = None
        for feature_function in [self.get_k_most_common_word_unigram_counts_including_stopwords(k=k),
                                 self.get_k_most_common_word_bigram_counts_including_stopwords(k=k),
                                 self.get_k_most_common_word_trigram_counts_including_stopwords(k=k),
                                 self.get_overlap_score_doc2vec(),
                                 self.get_overlap_score_sbert(),
                                 self.get_outlier_score_doc2vec(),
                                 self.get_outlier_score_sbert(),
                                 #self.get_lda_topic_distribution,
                                 self.get_tfidf(k=30),
                                 self.get_tag_distribution(k=30),
                                 self.get_spelling_error_distribution(),
                                 self.get_production_distribution(k=30), # this returns an empty dataframe if language is German
                                ]:
                                #self.get_keyness(k)]:
            if result is None:
                result = feature_function
            else:
                result = result.merge(feature_function, on="book_name")
        return result
