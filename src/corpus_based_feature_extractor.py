from code import compile_command
import os
from xml.sax.handler import feature_namespace_prefixes
import spacy
import re
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, _document_frequency
import scipy
from gensim.models import LdaMulticore
from gensim.matutils import Sparse2Corpus
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import pickle
from collections import Counter
from utils import load_list_of_lines, save_list_of_lines, unidecode_custom, df_from_dict
from production_rule_extractor import ProductionRuleExtractor
from doc_based_feature_extractor import DocBasedFeatureExtractor
from corpus_toolkit import corpus_tools as ct
logging.basicConfig(level=logging.DEBUG)


class CorpusBasedFeatureExtractor():
    def __init__(self, lang, doc_paths, all_average_sbert_sentence_embeddings, all_doc2vec_chunk_embeddings, sentences_per_chunk=200, nr_features=100):
        self.lang = lang
        self.doc_paths = doc_paths
        self.all_average_sbert_sentence_embeddings = all_average_sbert_sentence_embeddings
        self.all_doc2vec_chunk_embeddings = all_doc2vec_chunk_embeddings
        self.sentences_per_chunk = sentences_per_chunk
        self.nr_features = nr_features

        self.word_statistics = self.__get_word_statistics(include_ngrams=True)

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

        self.chunks, self.new_sbert, self.new_doc2vec = self.__get_chunks()

    def __get_chunks(self):
        chunks = []
        all_average_sbert_sentence_embeddings = []
        all_doc2vec_chunk_embeddings = []

        for doc_path in self.doc_paths:
            extractor = DocBasedFeatureExtractor(self.lang, doc_path, self.sentences_per_chunk)
            chunks.extend(extractor.chunks)
            all_average_sbert_sentence_embeddings.append([np.array(chunk.sbert_sentence_embeddings).mean(axis=0) for chunk in extractor.chunks])
            all_doc2vec_chunk_embeddings.append([chunk.doc2vec_chunk_embedding for chunk in extractor.chunks])
            #sbert_embeddings = [np.array(chunk.sbert_sentence_embeddings).mean(axis=0) for chunk in self.chunks] # Average across sentences belonging to a chunk
            #doc2vec_embeddings = [chunk.doc2vec_chunk_embedding for chunk in self.chunks]
        return chunks, all_average_sbert_sentence_embeddings, all_doc2vec_chunk_embeddings

    def __get_word_statistics(self, include_ngrams=True):
        # get total counts over all documents
        total_unigram_counts = Counter()
        book_unigram_count = {}
        book_unigram_mapping_abs = {}
        book_unigram_mapping_rel = {}

        for doc_path in self.doc_paths:
            book_name = doc_path.split("/")[-1][:-4]
            tokenized_sentences_path = doc_path.replace("/raw_docs", f"/tokenized_sentences")
            tokenized_sentences = load_list_of_lines(tokenized_sentences_path, "str")
            processed_sentences = self.__preprocess_sentences(tokenized_sentences)
            unigram_counts = self.__find_unigram_counts(processed_sentences)
            total_unigram_counts.update(unigram_counts)            
            sum_unigram_counts = sum(unigram_counts.values())# get counts per document    
            book_unigram_count[book_name] = total_unigram_counts
            book_unigram_mapping_abs[book_name] = unigram_counts#absolute frequency            
            book_unigram_mapping_rel[book_name] = dict((unigram, count / sum_unigram_counts) for unigram, count in unigram_counts.items() 
                if unigram in total_unigram_counts.keys()) #relative frequencies

        total_unigram_counts = dict(sorted(list(total_unigram_counts.items()), key=lambda x: -x[1])) #all words
        word_statistics = {
            # {unigram: count}
            "total_unigram_counts": total_unigram_counts,
            # {book_name: {unigram: count}
            "book_unigram_mapping_abs": book_unigram_mapping_abs,
            #{book_name: {unigram: rel freq}}
            "book_unigram_mapping_rel": book_unigram_mapping_rel,
        }

        if include_ngrams == True:
            total_bigram_counts = Counter()
            total_trigram_counts = Counter()
            book_bigram_count = {}
            book_trigram_count = {}
            book_bigram_mapping_abs = {}
            book_trigram_mapping_abs = {}
            book_bigram_mapping_rel = {}
            book_trigram_mapping_rel = {}

            for doc_path in self.doc_paths:
                book_name = doc_path.split("/")[-1][:-4]
                tokenized_sentences_path = doc_path.replace("/raw_docs", f"/tokenized_sentences")
                tokenized_sentences = load_list_of_lines(tokenized_sentences_path, "str")
                processed_sentences = self.__preprocess_sentences(tokenized_sentences)
                
                bigram_counts = self.__find_bigram_counts(processed_sentences)
                trigram_counts = self.__find_trigram_counts(processed_sentences)
                total_bigram_counts.update(bigram_counts)
                total_trigram_counts.update(trigram_counts)
                sum_bigram_counts = sum(bigram_counts.values())
                sum_trigram_counts = sum(trigram_counts.values())
                book_bigram_count[book_name] = total_bigram_counts
                book_trigram_count[book_name] = total_trigram_counts
                book_bigram_mapping_abs[book_name] = bigram_counts
                book_trigram_mapping_abs[book_name] = trigram_counts
                book_bigram_mapping_rel[book_name] = dict((bigram, count / sum_bigram_counts) for bigram, count in bigram_counts.items() if bigram in total_bigram_counts.keys())
                book_trigram_mapping_rel[book_name] = dict((trigram, count / sum_trigram_counts) for trigram, count in trigram_counts.items() if trigram in total_trigram_counts.keys())

            total_bigram_counts = dict(sorted(list(total_bigram_counts.items()), key=lambda x: -x[1])[:2000]) ######################################
            total_trigram_counts = dict(sorted(list(total_trigram_counts.items()), key=lambda x: -x[1])[:2000])
            
            #filter dicts so that they only contain most frequent n-grams
            for mapping in [book_bigram_mapping_abs, book_bigram_mapping_rel]:
                for book, book_dict in mapping.items():
                    for ngram in list(book_dict.keys()):
                        if ngram not in list(total_bigram_counts.keys()):
                            del book_dict[ngram]
            for mapping in [book_trigram_mapping_abs, book_trigram_mapping_rel]:
                for book, book_dict in mapping.items():
                    for ngram in list(book_dict.keys()):
                        if ngram not in list(total_trigram_counts.keys()):
                            del book_dict[ngram]

            # {bigram: count}
            word_statistics["total_bigram_counts"] = total_bigram_counts
            word_statistics["total_trigram_counts"] = total_trigram_counts
            # {book_name: {bigram: count}}
            word_statistics["book_bigram_mapping_abs"] = book_bigram_mapping_abs
            word_statistics["book_trigram_mapping_abs"] = book_trigram_mapping_abs
            # {book_name: {bigram: relative freq}}
            word_statistics["book_bigram_mapping_rel"] = book_bigram_mapping_rel
            word_statistics["book_trigram_mapping_rel"] = book_trigram_mapping_rel
        return word_statistics

    def __find_unigram_counts(self, processed_sentences):
        unigram_counts = {}
        for processed_sentence in processed_sentences:
            for unigram in processed_sentence.split():
                if unigram in unigram_counts.keys():
                    unigram_counts[unigram] += 1
                else:
                    unigram_counts[unigram] = 1
        return unigram_counts

    def __find_bigram_counts(self, processed_sentences):
        processed_text = "<BOS> " + " <EOS> <BOS> ".join(processed_sentences) + " <EOS>"
        processed_text_split = processed_text.split()
        bigram_counts = {}
        for i in range(len(processed_text_split) - 1):
            current_bigram = processed_text_split[i] + " " + processed_text_split[i+1]
            if current_bigram in bigram_counts:
                bigram_counts[current_bigram] += 1
            else:
                bigram_counts[current_bigram] = 1
        return bigram_counts

    def __find_trigram_counts(self, processed_sentences):
        processed_text = "<BOS> <BOS> " + " <EOS> <EOS> <BOS> <BOS> ".join(processed_sentences) + " <EOS> <EOS>"
        processed_text_split = processed_text.split()
        trigram_counts = {}
        for i in range(len(processed_text_split) - 2):
            current_trigram = processed_text_split[i] + " " + processed_text_split[i+1] + " " + processed_text_split[i+2]
            if current_trigram in trigram_counts.keys():
                trigram_counts[current_trigram] += 1
            else:
                trigram_counts[current_trigram] = 1
        return trigram_counts

    def __preprocess_sentences(self, tokenized_sentences):
        def __preprocess_sentences_helper(text):
            text = text.lower()
            text = unidecode_custom(text)
            text = re.sub("[^a-zA-Z]+", " ", text).strip()
            text = text.split()
            text = " ".join(text)
            return text
        return [__preprocess_sentences_helper(sentence) for sentence in tokenized_sentences]

    def __tag_chunks(self, tag_type, gram_type):
        def __tag_sentence(sentence_tags, gram_type):
            if gram_type == "unigram":
                return sentence_tags
            elif gram_type == "bigram":
                tokens_bigram_temp = ["BOS"] + sentence_tags + ["EOS"]
                tokens_bigram = ["_".join([tokens_bigram_temp[i], tokens_bigram_temp[i+1]]) for i in range(len(tokens_bigram_temp)-1)]
                return tokens_bigram
            elif gram_type == "trigram":
                tokens_trigram_temp = ["BOS", "BOS"] + sentence_tags + ["EOS", "EOS"]
                tokens_trigram = ["_".join([tokens_trigram_temp[i], tokens_trigram_temp[i+1], tokens_trigram_temp[i+2]]) for i in range(len(tokens_trigram_temp)-2)]
                return tokens_trigram
            else:
                raise Exception("Not a valid gram_type")

        def __tag_chunk(chunk, tag_type, gram_type):
            tags_path = chunk.doc_path.replace("/raw_docs", f"/{tag_type}_tags_spc_{self.sentences_per_chunk}").replace(".txt", f"_chunkid_{chunk.chunk_id}.txt")
            if os.path.exists(tags_path):
                all_sentence_tags = [line for line in load_list_of_lines(tags_path, "str")]
            else:
                all_sentence_tags = []
                # Represent sentences as strings of tags
                for sentence in chunk.tokenized_sentences:
                    doc = self.nlp(sentence)
                    if tag_type == "pos":
                        sentence_tags = [token.pos_.replace(" ", "") for token in doc]
                    elif tag_type == "tag":
                        sentence_tags = [token.tag_.replace(" ", "") for token in doc]
                    elif tag_type == "dep":
                        sentence_tags = [token.dep_.replace(" ", "") for token in doc]
                    else:
                        raise Exception("Not a valid tag_type")
                    all_sentence_tags.append(" ".join(sentence_tags))
                save_list_of_lines(all_sentence_tags, tags_path, "str")
            
            # Count number of occurrences of tags
            chunk_tag_counter = Counter()
            for sentence_tags in all_sentence_tags:
                sentence_tags = __tag_sentence(sentence_tags.split(), gram_type)
                chunk_tag_counter.update(sentence_tags)
            return chunk_tag_counter

        tagged_chunks = {}
        corpus_tag_counter = Counter()
        for chunk in self.chunks:
            chunk_tag_counter = __tag_chunk(chunk, tag_type, gram_type)
            tagged_chunks[chunk.book_name + "_" + str(chunk.chunk_id)] = chunk_tag_counter
            corpus_tag_counter.update(chunk_tag_counter)

        # get first k tags of corpus_tag_counter
        corpus_tag_counter = sorted([(tag, count) for tag, count in corpus_tag_counter.items()], key=lambda x: -x[1])[:self.nr_features]
        corpus_tag_counter = [tag for tag, count in corpus_tag_counter]

        data = []

        # get first k tags of each chunk_tag_counter
        for chunk_name, tagged_chunk in tagged_chunks.items():
            # create label
            current_chunk_chosen_tag_counts = dict([(tag_type + "_" + gram_type + "_" + tag_name, tagged_chunk[tag_name]) for tag_name in corpus_tag_counter])
            # relative counts
            current_chunk_chosen_tag_counts_sum = sum([count for tag, count in current_chunk_chosen_tag_counts.items()])
            current_chunk_chosen_tag_counts = dict([(tag, count/current_chunk_chosen_tag_counts_sum) for tag, count in current_chunk_chosen_tag_counts.items()])
            current_chunk_chosen_tag_counts["book_name"] = chunk_name
            data.append(current_chunk_chosen_tag_counts)

        df = pd.DataFrame(data)
        return df

    def get_tag_distribution(self):
        result_df = None
        for tag_type in ['pos']:  # ['pos', 'tag', 'dep']:
            for gram_type in ['unigram', 'bigram', 'trigram']:
                current_df = self.__tag_chunks(tag_type, gram_type)
                if result_df is None:
                    result_df = current_df
                else:
                    result_df = result_df.merge(current_df, on='book_name')
        return result_df

    def __get_book_production_counts(self, book, pre):
        chunk_production_counter = Counter()
        for sentence in book:
            sentence_production_counter = pre.get_sentence_production_counter(sentence)
            chunk_production_counter.update(sentence_production_counter)
        return chunk_production_counter

    def get_production_distribution(self):
        """
        Returns an empty dataframe if the language is German. Reason is explained in
        docstring of ProductionRuleExtractor.
        """
        if self.lang == "ger":
            return pd.DataFrame(data=[doc_path.split("/")[-1][:-4] for doc_path in self.doc_paths], columns=["book_name"])
        elif self.lang == "eng":
            pass
        else:
            raise Exception("Not a valid language")

        pre = ProductionRuleExtractor()
        chunk_production_counters = {}
        corpus_production_counter = Counter()

        for chunk in self.chunks:
            chunk_production_counter = self.__get_book_production_counts(chunk.tokenized_sentences, pre)
            chunk_production_counters[chunk.book_name + "_" + str(chunk.chunk_id)] = chunk_production_counter
            corpus_production_counter.update(chunk_production_counter)

        # get first k tags of corpus_tag_counter
        corpus_production_counter = sorted([(tag, count) for tag, count in corpus_production_counter.items()], key=lambda x: -x[1])[:self.nr_features]
        corpus_production_counter = [tag for tag, count in corpus_production_counter]

        data = []

        # get first k tags of each chunk_tag_counter
        for chunk_name, chunk_prodution_counter in chunk_production_counters.items():
            current_chunks_chosen_production_counts = dict([(production_type, chunk_prodution_counter[production_type]) for production_type in corpus_production_counter])
            current_chunks_chosen_production_counts_sum = sum([count for tag, count in current_chunks_chosen_production_counts.items()])
            current_chunks_chosen_production_counts = dict([(tag, count/current_chunks_chosen_production_counts_sum) for tag, count in current_chunks_chosen_production_counts.items()])
            current_chunks_chosen_production_counts["book_name"] = chunk_name
            data.append(current_chunks_chosen_production_counts)

        df = pd.DataFrame(data)
        return df

    def get_overlap_score(self, embedding_type):
        if embedding_type == "doc2vec":
            all_embeddings = self.all_doc2vec_chunk_embeddings
        elif embedding_type == "sbert":
            all_embeddings = self.all_average_sbert_sentence_embeddings
        else:
            raise Exception(f"Not a valid embedding_type {embedding_type}.")

        cluster_means = []
        for index, current_list_of_embeddings in enumerate(all_embeddings):
            cluster_means.append(np.array(current_list_of_embeddings).mean(axis=0)) #average of all sentenfces

        labels = []
        predictions = []
        for label_index, current_list_of_embeddings in list(enumerate(all_embeddings)):
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
            # fraction of nearest chunks that are part of other books
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

    def filter_document_term_matrix(self, dtm, min_nr_documents=None, min_percent_documents=None, max_nr_documents=None, max_percent_documents=None):
        if min_nr_documents is None and min_percent_documents is None and max_nr_documents is None and max_percent_documents is None:
            raise Exception("Specify at least one filtering criterion.")
        min_columns = []
        max_columns = []
        document_frequency = dtm.astype(bool).sum(axis=0)

        #Filter minimum
        if min_nr_documents is not None and min_percent_documents is not None:
            raise Exception("Specify either the the minimum number or the minimum percentage of documents in which a term must occur.")
        elif min_percent_documents is not None:
            min_nr_documents = round(min_percent_documents/100 * dtm.shape[0])
        if min_nr_documents is not None:
            min_columns = [dtm.columns[x] for x in range(0,len(dtm.columns)) if document_frequency[x]>=min_nr_documents]

        #Filter maximum
        if max_nr_documents is not None and max_percent_documents is not None:
            raise Exception("Specify either the the maximum number or the maximum percentage of documents in which a term can occur.")
        elif max_percent_documents is not None:
            max_nr_documents = round(max_percent_documents/100 * dtm.shape[0])
        if max_nr_documents is not None:
            max_columns = [dtm.columns[x] for x in range(0,len(dtm.columns)) if document_frequency[x]<=max_nr_documents]

        if min_columns and max_columns:
            dtm_reduced = dtm[list(set(min_columns).intersection(max_columns))]
        elif min_columns:
            dtm_reduced = dtm[min_columns]
        else:
            dtm_reduced = dtm[max_columns]
        return dtm_reduced

    def get_distance_from_corpus(self, ngram_type, min_nr_documents=None, min_percent_documents=None, max_nr_documents=None, max_percent_documents=None):
        if ngram_type == 'unigram':
            corpus_freq = df_from_dict(d=self.word_statistics['total_unigram_counts'], keys_as_index=True, keys_column_name='ngram', values_column_value='corpus_freq')
            dtm = pd.DataFrame(self.word_statistics['book_unigram_mapping_abs'])
        elif ngram_type == 'bigram':
            if not 'total_bigram_counts' in self.word_statistics.keys():
                self.word_statistics = self.__get_word_statistics(include_ngrams=True)
            corpus_freq = df_from_dict(d=self.word_statistics['total_bigram_counts'], keys_as_index=True, keys_column_name='ngram', values_column_value='corpus_freq')
            dtm = pd.DataFrame(self.word_statistics['book_bigram_mapping_abs'])
        elif ngram_type == 'trigram':
            if not 'total_trigram_counts' in self.word_statistics.keys():
                self.word_statistics = self.__get_word_statistics(include_ngrams=True)
            corpus_freq = df_from_dict(d=self.word_statistics['total_trigram_counts'], keys_as_index=True, keys_column_name='ngram', values_column_value='corpus_freq')
            dtm = pd.DataFrame(self.word_statistics['book_trigram_mapping_abs'])
        dtm = dtm.fillna(0).T
        dtm_filtered = self.filter_document_term_matrix(dtm, min_nr_documents, min_percent_documents, max_nr_documents, max_percent_documents).T

        distances = {}
        # for unigrams
        for chunk in self.chunks:
            if ngram_type == 'unigram':
                chunk_ngram_counts = chunk.unigram_counts
            elif ngram_type == 'bigram':
                chunk_ngram_counts = chunk.bigram_counts
            elif ngram_type == 'trigram':
                chunk_ngram_counts = chunk.trigram_counts
            chunk_freq = df_from_dict(d=chunk_ngram_counts, keys_as_index=True, keys_column_name='ngram', values_column_value='chunk_freq')
            df = corpus_freq.merge(chunk_freq, how='outer', left_index=True, right_index=True, validate='one_to_one').fillna(0)
            if df['corpus_freq'].isnull().values.any():
                raise Exception(f"Error in word statistics: Not all {ngram_type}s in total counts.")
            # Only keep words that are in filtered matrix            
            df = df[df.index.isin(dtm_filtered.index)]    
            # Frequency in corpus without the chunk
            df['corpus_freq'] = df['corpus_freq'] - df['chunk_freq']
            # Relative frequencies in corpus and chunk
            df = df.div(df.sum(axis=0), axis=1)
            cosine_distance = scipy.spatial.distance.cosine(df["corpus_freq"], df["chunk_freq"])
            distances[chunk.book_name + "_" + str(chunk.chunk_id)] = cosine_distance
        distances = df_from_dict(d=distances, keys_as_index=False, keys_column_name='book_name', values_column_value=f"{ngram_type}_distance")
        return distances 

    def get_unigram_distance(self):
        distances = self.get_distance_from_corpus(ngram_type='unigram', min_nr_documents=2)
        return distances

    def get_unigram_distance_limited(self):
        #Filter for mid-frequency words
        distances = self.get_distance_from_corpus(ngram_type='unigram', min_percent_documents=5, max_percent_documents=50)
        distances = distances.rename(columns={"unigram_distance": "unigram_distance_limited"})
        return distances

    def get_bigram_distance(self):
        distances = self.get_distance_from_corpus(ngram_type='bigram', min_nr_documents=2)
        return distances

    def get_trigram_distance(self):
        distances = self.get_distance_from_corpus(ngram_type='trigram', min_nr_documents=2)
        return distances

    def get_all_features(self):
        corpus_chunk_feature_mapping = [self.get_unigram_distance,
                                self.get_unigram_distance_limited,
                                self.get_bigram_distance,
                                self.get_trigram_distance,
                                self.get_tag_distribution,
                                self.get_production_distribution,  # this returns an empty dataframe if language is German
                                #self.get_spelling_error_distribution,
                                ]
        corpus_book_feature_mapping = [#self.get_lda_topic_distribution,
        #                         self.get_k_most_common_unigram_counts_including_stopwords,
        #                         self.get_k_most_common_bigram_counts_including_stopwords,
        #                         self.get_k_most_common_trigram_counts_including_stopwords,
                                self.get_overlap_score_doc2vec,
                                self.get_overlap_score_sbert,
                                self.get_outlier_score_doc2vec,
                                self.get_outlier_score_sbert,
                                ]

        corpus_chunk_features = None
        for feature_function in corpus_chunk_feature_mapping:
            if  corpus_chunk_features is None:
                corpus_chunk_features = feature_function()
            else:
                corpus_chunk_features = corpus_chunk_features.merge(feature_function(), on="book_name")
        if self.sentences_per_chunk == None:
            corpus_chunk_features['book_name'] = corpus_chunk_features['book_name'].str.split('_').str[:4].str.join('_')

        corpus_book_features = None
        if self.sentences_per_chunk is not None:
            for feature_function in corpus_book_feature_mapping:
                if corpus_book_features is None:
                    corpus_book_features = feature_function()
                else:
                    corpus_book_features = corpus_book_features.merge(feature_function(), on="book_name")

        return corpus_chunk_features, corpus_book_features



    # def get_spelling_error_distribution(self, chunk, doc_path):
    #     def __get_spelling_error_count_in_sentence(sentence):
    #         misspelled = self.spell_checker.unknown(sentence.split())
    #         return len(misspelled)

    #     def __get_spelling_error_rate_in_book(book):
    #         error_counter = sum([__get_spelling_error_count_in_sentence(sentence) for sentence in book])
    #         error_rate = error_counter / len(book)
    #         return error_rate

    #     data = []
    #     for doc_path in self.doc_paths:
    #         book_name = doc_path.split("/")[-1][:-4]
    #         tokenized_sentences_path = doc_path.replace("/raw_docs", "/tokenized _sentences")
    #         sentences = load_list_of_lines(tokenized_sentences_path, "str")
    #         error_rate = __get_spelling_error_rate_in_book(sentences)
    #         data.append({"book_name": book_name,
    #                      "error_rate": error_rate})
    #     df = pd.DataFrame(data)
    #     return df


    # def get_lda_topic_distribution(self):
    #     num_topics = 10

    #     documents = []
    #     for doc_path in self.doc_paths:
    #         with open(doc_path, "r") as reader:
    #             documents.append(reader.read().strip())

    #     if self.lang == "eng":
    #         stop_words = spacy.lang.en.stop_words.STOP_WORDS
    #     elif self.lang == "ger":
    #         stop_words = spacy.lang.de.stop_words.STOP_WORDS
    #     else:
    #         raise Exception(f"Not a valid language {self.lang}")

    #     vect = CountVectorizer(min_df=20, max_df=0.2, stop_words=stop_words,
    #                            token_pattern='(?u)\\b\\w\\w\\w+\\b')
    #     X = vect.fit_transform(documents)
    #     corpus = Sparse2Corpus(X, documents_columns=False)
    #     id_map = dict((v, k) for k, v in vect.vocabulary_.items())
    #     lda_model = LdaMulticore(corpus=corpus, id2word=id_map, passes=2, random_state=42, num_topics=num_topics, workers=3)

    #     topic_distributions = []
    #     book_names = []
    #     for doc_path, document in zip(self.doc_paths, documents):
    #         book_name = doc_path.split("/")[-1][:-4]
    #         string_input = [document]
    #         X = vect.transform(string_input)
    #         corpus = Sparse2Corpus(X, documents_columns=False)
    #         output = list(lda_model[corpus])[0]
    #         full_output = [0] * num_topics
    #         for topic_id, ratio in output:
    #             full_output[topic_id] = ratio
    #         topic_distributions.append(full_output)
    #         book_names.append(book_name)
    #     topic_distributions = pd.DataFrame(topic_distributions, columns=[f"lda_topic_{i+1}" for i in range(num_topics)])
    #     topic_distributions["book_name"] = book_names
    #     return topic_distribution

    # def get_tfidf(self):
    #     document_term_matrix = pd.DataFrame.from_dict(self.word_statistics['book_unigram_mapping_abs']).fillna(0).T
    #     # Tfidf
    #     t = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True)
    #     tfidf = pd.DataFrame.sparse.from_spmatrix(t.fit_transform(document_term_matrix), columns=document_term_matrix.columns, index=document_term_matrix.index)
    #     tfidf_reduced = self.filter_document_term_matrix(tfidf, min_percent_documents=10)
    #     # From remaining words, keep only those that are in the top k for at least one book
    #     all_top_k_words = []
    #     for index, row in tfidf_reduced.iterrows():
    #         top_k_words = row.nlargest(n=self.nr_features, keep='all')
    #         all_top_k_words.extend(top_k_words.index.to_list())
    #     all_top_k_words = list(set(all_top_k_words))
    #     tfidf_top_k = tfidf_reduced[all_top_k_words]
    #     tfidf_top_k.columns = [f"tfidf_{column}" for column in tfidf_top_k.columns]
    #     tfidf_top_k = tfidf_top_k.reset_index().rename(columns={'level_0':'book_name', 'index':'book_name'}) #automatically created column name can be 'index' or 'level_0'
    #     return tfidf_top_k

    # def get_k_most_common_ngram_counts(self, k, n, include_stopwords):
    #     if n == 1:
    #         dct1 = self.word_statistics["total_unigram_counts"]
    #         dct2 = self.word_statistics["book_unigram_mapping_rel"]
    #     elif n == 2 or n == 3:
    #         if not "total_bigram_counts" in self.word_statistics:
    #             self.word_statistics = self.__get_word_statistics(include_ngrams=True)
    #         if n == 2:
    #             dct1 = self.word_statistics["total_bigram_counts"]
    #             dct2 = self.word_statistics["book_bigram_mapping_rel"]
    #         else:
    #             dct1 = self.word_statistics["total_trigram_counts"]
    #             dct2 = self.word_statistics["book_trigram_mapping_rel"]
    #     else:
    #         raise Exception(f"Not a valid n: {n}")
    #     if include_stopwords:
    #         # words that are the most common in the whole corpus
    #         most_common_k_ngrams = [ngram for ngram, count in sorted(list(dct1.items()), key=lambda x: -x[1])[:k]]
    #     else:
    #         filtered_ngrams = []
    #         for ngram, count in dct1.items():
    #             split_ngram = ngram.split()
    #             exclude = False
    #             for word in split_ngram:
    #                 if word in self.stopwords:
    #                     exclude = True
    #             if exclude:
    #                 continue
    #             else:
    #                 filtered_ngrams.append((ngram, count))
    #         most_common_k_ngrams = [ngram for ngram, count in sorted(filtered_ngrams, key=lambda x: -x[1])[:k]]
    #     result = []
    #     for book_name, ngram_counts in dct2.items():
    #         # get freq of k most common n-grams in current document
    #         dct = dict((f"{k}_most_common_{n}gram_stopword_{include_stopwords}_{common_ngram}", dct2[book_name].get(common_ngram, 0)) for common_ngram in most_common_k_ngrams)
    #         dct["book_name"] = book_name
    #         result.append(dct)
    #     result = pd.DataFrame(result)
    #     return result

    # def get_k_most_common_unigram_counts_including_stopwords(self):
    #     return self.get_k_most_common_ngram_counts(self.nr_features, 1, True)

    # def get_k_most_common_bigram_counts_including_stopwords(self):
    #     return self.get_k_most_common_ngram_counts(self.nr_features, 2, True)

    # def get_k_most_common_trigram_counts_including_stopwords(self):
    #     return self.get_k_most_common_ngram_counts(self.nr_features, 3, True)

    # def get_k_most_common_unigram_counts_excluding_stopwords(self):
    #     return self.get_k_most_common_ngram_counts(self.nr_features, 1, False)

    # def get_k_most_common_bigram_counts_excluding_stopwords(self):
    #     return self.get_k_most_common_ngram_counts(self.nr_features, 2, False)

    # def get_k_most_common_trigram_counts_excluding_stopwords(self):
    #     return self.get_k_most_common_ngram_counts(self.nr_features, 3, False)