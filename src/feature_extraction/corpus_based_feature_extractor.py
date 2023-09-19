import os
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)
from tqdm import tqdm
from collections import Counter
from functools import wraps, reduce
from multiprocessing import current_process, Queue, Process, cpu_count
import scipy
import pickle
import time
from sklearn.neighbors import BallTree
import spacy
import sys
sys.path.append("..")
from utils import load_list_of_lines, save_list_of_lines, get_filename_from_path
from .production_rule_extractor import ProductionRuleExtractor
from .doc_based_feature_extractor import DocBasedFeatureExtractor
from .ngrams import NgramCounter


class CorpusBasedFeatureExtractor():
    '''Get features for which the whole corpus needs to be considered.'''
    def __init__(self, language, doc_paths, as_chunk, tokens_per_chunk, nr_features=100):
        self.language = language
        self.doc_paths = doc_paths
        self.as_chunk = as_chunk
        self.tokens_per_chunk = tokens_per_chunk
        self.nr_features = nr_features
        self.nlp = self.load_spacy_model()
        self.nc = NgramCounter(self.language)
        self.ngrams = self.nc.load_all_ngrams(as_chunk=self.as_chunk)
        self.generate_chunks = self.get_chunks()
        print(len(self.generate_chunks))


        self.all_average_sbert_embeddings = []
        self.all_d2v_embeddings = []
        for doc_chunks in self.generate_chunks:
            print(len(doc_chunks))
            curr_sbert = []
            curr_doc2vec = []
            for chunk in doc_chunks:
                print(type(chunk))
                curr_sbert.append(np.array(chunk.sbert_embeddings).mean(axis=0))
                curr_doc2vec.append(chunk.d2v_embeddings)
                print(np.array(chunk.sbert_embeddings).mean(axis=0).shape, chunk.d2v_embeddings.shape) #########################
            self.all_average_sbert_embeddings.append(curr_sbert)
            self.all_d2v_embeddings.append(curr_doc2vec)
        print(f'all_average_sbert_embeddings', len(self.all_average_sbert_embeddings), len(self.all_average_sbert_embeddings[0]))
        print(f'all_d2v_embeddings', len(self.all_d2v_embeddings), len(self.all_d2v_embeddings[0]))


    def load_spacy_model(self):
        if self.language == 'eng':
            model_name = 'en_core_web_sm'
        elif self.language == 'ger':
            model_name = 'de_core_news_sm'
        else:
            raise Exception(f'Not a valid language {self.language}')

        try:
            nlp = spacy.load(model_name)
            return nlp
        except OSError:
            print(f'The model {model_name} for Spacy is missing.')



    def get_chunks(self,         
        get_ngrams=False,
        get_char_counts=False):
        
        s = time.time()
        chunks = []
        for doc_path in tqdm(self.doc_paths):
            doc_chunks = DocBasedFeatureExtractor(
                language=self.language,
                doc_path=doc_path,
                as_chunk=self.as_chunk,
                tokens_per_chunk=self.tokens_per_chunk, 
                ngrams=self.ngrams,
                get_ngrams=False,
                get_char_counts=False).chunks
            chunks.append(doc_chunks)
        object_size = sys.getsizeof(chunks) #############################a
        print(f"Size of the chunks of 1 doc: {object_size} bytes")
        print(f'{time.time()-s}s to get all chunksf')
        return chunks

    # def generate_chunks(self,         
    #     get_ngrams=False,
    #     get_char_counts=False):
        

    #     for doc_path in tqdm(self.doc_paths):
    #         doc_chunks = DocBasedFeatureExtractor(
    #             language=self.language,
    #             doc_path=doc_path,
    #             as_chunk=self.as_chunk,
    #             tokens_per_chunk=self.tokens_per_chunk, 
    #             ngrams=self.ngrams,
    #             get_ngrams=False,
    #             get_char_counts=False).chunks
    #         yield doc_chunks
    

    def tag_chunks(self, tag_type, gram_type):
        def __tag_sentence(sentence_tags, gram_type):
            if gram_type == 'unigram':
                return sentence_tags
            elif gram_type == 'bigram':
                tokens_bigram_temp = ['BOS'] + sentence_tags + ['EOS']
                tokens_bigram = ['_'.join([tokens_bigram_temp[i], tokens_bigram_temp[i+1]]) for i in range(len(tokens_bigram_temp)-1)]
                return tokens_bigram
            elif gram_type == 'trigram':
                tokens_trigram_temp = ['BOS', 'BOS'] + sentence_tags + ['EOS', 'EOS']
                tokens_trigram = ['_'.join([tokens_trigram_temp[i], tokens_trigram_temp[i+1], tokens_trigram_temp[i+2]]) for i in range(len(tokens_trigram_temp)-2)]
                return tokens_trigram
            else:
                raise Exception('Not a valid gram_type')

        def tag_chunk(chunk, tag_type, gram_type):
            tags_path = chunk.doc_path.replace('/text_raw', f'/{tag_type}_tags_tpc_{self.tokens_per_chunk}_usechunks_{self.as_chunk}').replace('.txt', f'_chunkid_{chunk.chunk_idx}.txt')
            if os.path.exists(tags_path):
                all_sentence_tags = [line for line in load_list_of_lines(tags_path, 'str')]
            else:
                all_sentence_tags = []
                # Represent sentences as strings of tags
                for sentence in chunk.sentences:
                    doc = self.nlp(sentence)
                    if tag_type == 'pos':
                        sentence_tags = [token.pos_.replace(' ', '') for token in doc]
                    elif tag_type == 'tag':
                        sentence_tags = [token.tag_.replace(' ', '') for token in doc]
                    elif tag_type == 'dep':
                        sentence_tags = [token.dep_.replace(' ', '') for token in doc]
                    else:
                        raise Exception('Not a valid tag_type')
                    all_sentence_tags.append(' '.join(sentence_tags))
                save_list_of_lines(all_sentence_tags, tags_path, 'str')
            
            # Count number of occurrences of tags
            chunk_tag_counter = Counter()
            for sentence_tags in all_sentence_tags:
                sentence_tags = __tag_sentence(sentence_tags.split(), gram_type)
                chunk_tag_counter.update(sentence_tags)
            return chunk_tag_counter

        tagged_chunks = {}
        corpus_tag_counter = Counter()
        for doc_chunks in self.generate_chunks:
            for chunk in doc_chunks:
                chunk_tag_counter = tag_chunk(chunk, tag_type, gram_type)
                tagged_chunks[chunk.chunkname] = chunk_tag_counter
                corpus_tag_counter.update(chunk_tag_counter)

        # get first k tags of corpus_tag_counter
        corpus_tag_counter = sorted([(tag, count) for tag, count in corpus_tag_counter.items()], key=lambda x: -x[1])[:self.nr_features]
        corpus_tag_counter = [tag for tag, count in corpus_tag_counter]

        data = []

        # get first k tags of each chunk_tag_counter
        for chunk_name, tagged_chunk in tagged_chunks.items():
            # create label
            current_chunk_chosen_tag_counts = dict([(tag_type + '_' + gram_type + '_' + tag_name, tagged_chunk[tag_name]) for tag_name in corpus_tag_counter])
            # relative counts
            current_chunk_chosen_tag_counts_sum = sum([count for tag, count in current_chunk_chosen_tag_counts.items()])
            current_chunk_chosen_tag_counts = dict([(tag, count/current_chunk_chosen_tag_counts_sum) for tag, count in current_chunk_chosen_tag_counts.items()])
            current_chunk_chosen_tag_counts['file_name'] = chunk_name
            data.append(current_chunk_chosen_tag_counts)

        df = pd.DataFrame(data)
        return df

    def get_tag_distribution(self):
        result_df = None
        for tag_type in ['pos']:  # ['pos', 'tag', 'dep']:
            for gram_type in ['unigram', 'bigram', 'trigram']:
                current_df = self.tag_chunks(tag_type, gram_type)
                if result_df is None:
                    result_df = current_df
                else:
                    result_df = result_df.merge(current_df, on='file_name')
        return result_df

    def get_book_production_counts(self, book, pre):
        chunk_production_counter = Counter()
        for sentence in book:
            sentence_production_counter = pre.get_sentence_production_counter(sentence)
            chunk_production_counter.update(sentence_production_counter)
        return chunk_production_counter

    def get_production_distribution(self):
        '''
        Returns an empty dataframe if the language is German. Reason is explained in
        docstring of ProductionRuleExtractor.
        '''
        if self.language == 'ger':
            raise Exception('Not implemented for German.')
        elif self.language == 'eng':
            pass
        else:
            raise Exception('Not a valid language')

        pre = ProductionRuleExtractor()
        chunk_production_counters = {}
        corpus_production_counter = Counter()

        for doc_chunks in self.generate_chunks:
            for chunk in doc_chunks:
                chunk_production_counter = self.get_book_production_counts(chunk.text, pre)
                chunk_production_counters[chunk.chunkname] = chunk_production_counter
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
            current_chunks_chosen_production_counts['file_name'] = chunk_name
            data.append(current_chunks_chosen_production_counts)

        df = pd.DataFrame(data)
        return df


    def get_overlap_score(self, embedding_type):
        if embedding_type == 'doc2vec':
            all_embeddings = self.all_d2v_embeddings
        elif embedding_type == 'sbert':
            all_embeddings = self.all_average_sbert_embeddings
        else:
            raise Exception(f'Not a valid embedding_type {embedding_type}.')

        # Centroid of chunks making up a text
        centroids = []
        n_chunks_per_doc = []
        for embeddings_per_doc in all_embeddings:
            centroids.append(np.array(embeddings_per_doc).mean(axis=0)) #average of all sentences
            n_chunks_per_doc.append(len(embeddings_per_doc))

        all_labels = []
        for i in range(0, len(n_chunks_per_doc)):
            all_labels.append(list(range(sum(n_chunks_per_doc[:i]), sum(n_chunks_per_doc[:i+1]))))

        # Get list of chunk arrays
        labels = []
        chunk_embeddings_list = []
        for embeddings_per_doc in all_embeddings:
            for chunk_embedding in embeddings_per_doc:
                chunk_embeddings_list.append(chunk_embedding)

        # Find centroid that has the smallest distance to current chunk embedding (nearest neighbour)
        # BallTree algorithm (Cranenburgh2019)
        # Find k nearest neighbours to each centroid, with k being the number of chunks in a text
        all_predictions = []
        tree = BallTree(chunk_embeddings_list, metric='euclidean')
        for centroid, curr_n_chunks_per_doc in zip(centroids, n_chunks_per_doc):
            # indices of k nearest neighbors of centroid
            indices = tree.query(X=centroid.reshape(1,-1), k=curr_n_chunks_per_doc, return_distance=False).tolist()
            indices = [int(index) for inner_list in indices for index in inner_list]
            all_predictions.append(list(indices))

        file_names = []
        overlap_scores = []
        for doc_path, labels, predictions in zip(self.doc_paths, all_labels, all_predictions):
            if not len(labels) == len(predictions):
                raise Exception(f'Number true and predicted values are not the same.')
            file_name = get_filename_from_path(doc_path)
            correct = 0
            incorrect = 0
            for prediction in predictions:
                if prediction in labels:
                    correct += 1
                else:
                    incorrect += 1
            # Fraction of nearest chunks that are part of other books
            overlap_score = incorrect / (incorrect + correct)
            file_names.append(file_name)
            overlap_scores.append(overlap_score)
        return pd.DataFrame.from_dict({'file_name': file_names, f'overlap_score_{embedding_type}': overlap_scores})


    def get_overlap_score_doc2vec(self):
        return self.get_overlap_score('doc2vec')
    

    def get_overlap_score_sbert(self):
        return self.get_overlap_score('sbert')


    def get_outlier_score(self, embedding_type):
        # Get embeddings
        if embedding_type == 'doc2vec':
            all_embeddings = self.all_d2v_embeddings
        elif embedding_type == 'sbert':
            all_embeddings = self.all_average_sbert_embeddings
        else:
            raise Exception(f'Not a valid embedding_type {embedding_type}.')

        # Calculate centroids
        centroids = []
        for index, embeddings_per_doc in enumerate(all_embeddings):
            centroids.append(np.array(embeddings_per_doc).mean(axis=0))

        # Find distance to nearest centroid
        outlier_scores = []
        file_names = []
        for current_index, current_centroid in enumerate(centroids):
            doc_path = self.doc_paths[current_index]
            file_name = get_filename_from_path(doc_path)
            nearest_distance = np.inf
            
            for other_index, other_centroid in enumerate(centroids):
                if current_index == other_index:
                    continue
                current_distance = np.linalg.norm(current_centroid - other_centroid)
                if current_distance < nearest_distance:
                    nearest_distance = current_distance
                    
            outlier_scores.append(nearest_distance)
            file_names.append(file_name)
        return pd.DataFrame.from_dict({'file_name': file_names, f'outlier_score_{embedding_type}': outlier_scores})
        

    def get_outlier_score_doc2vec(self):
        return self.get_outlier_score('doc2vec')

    def get_outlier_score_sbert(self):
        return self.get_outlier_score('sbert')


    def get_corpus_distance(self, ngram_type, min_docs=None, min_percent=None, max_docs=None, max_percent=None):
        data_dict = self.ngrams[ngram_type]
        dtm, _ = self.nc.filter_dtm(data_dict, min_docs, min_percent, max_docs, max_percent)

        dtm_sum = np.array(dtm.sum(axis=0))[0]

        distances = {}
        for idx, file_name in tqdm(enumerate(data_dict['file_names'])):
            # Access the term frequency values for the specific document
            file_counts = dtm[idx].toarray()
            corpus_counts = dtm_sum - file_counts
            cosine_distance = scipy.spatial.distance.cosine(corpus_counts, file_counts) # use sim or dist??? ################3
            distances[file_name] = cosine_distance

        # Turn both keys and values of a dict into columns of a df.
        distances = pd.DataFrame(distances.items(), columns=['file_name', f'{ngram_type}_distance'])
        return distances 


    def get_unigram_distance(self):
        distances = self.get_corpus_distance(ngram_type='unigram', min_docs=2)
        return distances


    def get_unigram_distance_limited(self):
        #Filter for mid-frequency words
        distances = self.get_corpus_distance(ngram_type='unigram', min_percent=5, max_percent=50)
        distances = distances.rename(columns={'unigram_distance': 'unigram_distance_limited'})
        return distances


    def get_bigram_distance(self):
        # min_docs is 2 by default, filtering not necessary
        distances = self.get_corpus_distance(ngram_type='bigram')
        return distances


    def get_trigram_distance(self):
        # min_docs is 2 by default, filtering not necessary
        distances = self.get_corpus_distance(ngram_type='trigram')
        return distances


    # def get_all_features(self):

    #     def multiprocessing_decorator(queue):
    #         def inner_decorator(func):
    #             @wraps(func)
    #             def wrapper(*args, **kwargs):
    #                 print(f'Starting process: {current_process().name}.')
    #                 features = func()
    #                 queue.put(features)
    #             return wrapper
    #         return inner_decorator

    #     chunk_queue = Queue()
    #     book_queue = Queue()

    #     chunk_functions = [self.get_unigram_distance,
    #                         self.get_unigram_distance_limited,
    #                         self.get_bigram_distance,
    #                         self.get_trigram_distance,
    #                         self.get_tag_distribution]
    #     if self.language == 'eng':
    #         chunk_functions.append(self.get_production_distribution)
            
    #     book_functions = [self.get_overlap_score_doc2vec,
    #                         self.get_outlier_score_doc2vec]
    #                         # self.get_overlap_score_sbert,
    #                         # self.get_outlier_score_sbert]
    #     print('chunk functions', chunk_functions)
        
    #     # Decorate functions to make them useable for multiprocessing
    #     # Reverse chunk_functions to start get_tag_distribution() and get_production_distribution() first
    #     chunk_functions = reversed([multiprocessing_decorator(chunk_queue)(func) for func in chunk_functions])
    #     book_functions = [multiprocessing_decorator(book_queue)(func) for func in book_functions]
        
    #     # Create new process for every function
    #     chunk_processes = [Process(target=func, name=func.__name__) for func in chunk_functions]
    #     if self.as_chunk == True:
    #         book_processes = [Process(target=func, name=func.__name__) for func in book_functions]
    #     else:
    #         book_processes = []
    #     processes = chunk_processes + book_processes

    #     nr_processes = max(cpu_count() - 4, 1) ####################     
    #     def _start_process(p):
    #         # Limit the number of cores used to avoid oversubscription
    #         alive = sum([p.is_alive() for p in processes])
    #         print('nr. processes alive', alive)
    #         if alive <= (nr_processes):
    #             p.start()
    #         else:
    #             time.sleep(15)
    #             _start_process(p)
    #     for p in processes:
    #         print('nr. processes alive', sum([p.is_alive() for p in processes]))
    #         _start_process(p)

    #     chunk_features = []
    #     book_features = []

    #     # Take elements from queues before joining processes
    #     while True:
    #         alive = any([p.is_alive() for p in processes])
    #         if not chunk_queue.empty():
    #             chunk_features.append(chunk_queue.get())
    #         if not book_queue.empty():
    #             book_features.append(book_queue.get())
    #         if not alive:
    #             break
            
    #     for p in processes:
    #         p.join()

    #     chunk_features = reduce(lambda df1, df2: df1.merge(df2, how='inner', on='file_name', validate='one_to_one'), chunk_features)
    #     if self.as_chunk == False:
    #         chunk_features['file_name'] = chunk_features['file_name'].str.split('_').str[:4].str.join('_')
    #     else:
    #         book_features = reduce(lambda df1, df2: df1.merge(df2, how='inner', on='file_name', validate='one_to_one'), book_features)


    #     return chunk_features, book_features

    def get_all_features(self):
        chunk_queue = []
        book_queue = []

        chunk_functions = [self.get_unigram_distance,
                        self.get_unigram_distance_limited,
                        self.get_bigram_distance,
                        self.get_trigram_distance,
                        self.get_tag_distribution]
        if self.language == 'eng':
            chunk_functions.append(self.get_production_distribution)

        book_functions = [self.get_overlap_score_doc2vec,
                        self.get_outlier_score_doc2vec,
                        self.get_overlap_score_sbert,
                        self.get_outlier_score_sbert]

        chunk_features = []
        book_features = []

        for func in chunk_functions:
            s = time.time()
            features = func()
            chunk_features.append(features)
            print(f'{time.time()-s} to calculate {func}')

        for func in book_functions:
            s = time.time()
            features = func()
            book_features.append(features)
            print(f'{time.time()-s} to calculate {func}')

        chunk_features = reduce(lambda df1, df2: df1.merge(df2, how='inner', on='file_name', validate='one_to_one'), chunk_features)
        if self.as_chunk==False:
            chunk_features['file_name'] = chunk_features['file_name'].str.split('_').str[:4].str.join('_')
        else:
            book_features = reduce(lambda df1, df2: df1.merge(df2, how='inner', on='file_name', validate='one_to_one'), book_features)

        return chunk_features, book_features
