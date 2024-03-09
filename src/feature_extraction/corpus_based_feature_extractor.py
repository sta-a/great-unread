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
import multiprocessing
import pickle
import time
from sklearn.neighbors import BallTree
import spacy
import sys
sys.path.append("..")
from utils import load_list_of_lines, save_list_of_lines, get_filename_from_path
from .doc_based_feature_extractor import DocBasedFeatureExtractor
from .ngrams import NgramCounter


class CorpusBasedFeatureExtractor():
    '''Get features for which the whole corpus needs to be considered.'''
    def __init__(self, language, doc_paths, as_chunk, pickle_dir, tokens_per_chunk, nr_features=100):
        self.language = language
        self.doc_paths = doc_paths
        self.as_chunk = as_chunk
        print(f'as chunk: {self.as_chunk}')
        self.pickle_dir = pickle_dir
        self.tokens_per_chunk = tokens_per_chunk
        self.nr_features = nr_features
        self.nlp = self.load_spacy_model()
        self.nc = NgramCounter(self.language)
        self.ngrams = self.nc.load_all_ngrams(as_chunk=self.as_chunk)

        if self.as_chunk:
            self.all_average_sbert_embeddings, self.all_d2v_embeddings = self.load_embeddings()

    def load_embeddings(self):
        path = os.path.join(self.pickle_dir, f'cbfe_embeddings_aschunk_{self.as_chunk}.pkl')
        if os.path.exists(path):
            all_average_sbert_embeddings,  all_d2v_embeddings = self.load_pickle(path)
        else:
            all_average_sbert_embeddings = []
            all_d2v_embeddings = []
            for doc_chunks in self.generate_chunks():
                curr_sbert = []
                curr_doc2vec = []
                for chunk in doc_chunks:
                    curr_sbert.append(np.array(chunk.sbert_embeddings).mean(axis=0))
                    curr_doc2vec.append(chunk.d2v_embeddings)
                all_average_sbert_embeddings.append(curr_sbert)
                all_d2v_embeddings.append(curr_doc2vec)
            self.save_pickle(path, (all_average_sbert_embeddings, all_d2v_embeddings))
        return all_average_sbert_embeddings, all_d2v_embeddings


    def load_spacy_model(self):
        if self.language == 'eng':
            model_name = 'en_core_web_sm'
        elif self.language == 'ger':
            model_name = 'de_core_news_sm'
        else:
            raise Exception(f'Not a valid language {self.language}')

        nlp = spacy.load(model_name, disable=["lemmatizer"])
        return nlp



    def generate_chunks(self,         
        get_ngrams=False,
        get_char_counts=False):
        

        for doc_path in tqdm(self.doc_paths): 
            doc_chunks = DocBasedFeatureExtractor(
                language=self.language,
                doc_path=doc_path,
                as_chunk=self.as_chunk,
                tokens_per_chunk=self.tokens_per_chunk, 
                ngrams=self.ngrams,
                get_ngrams=False,
                get_char_counts=False).chunks
            yield doc_chunks
    

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
        for doc_chunks in self.generate_chunks():
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
        
        # Remove chunk index from file name
        # result_df['file_name'] = result_df['file_name'].str.split('_').str[:4].str.join('_') #######################################
        return result_df


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

        # Get list of all embeddings
        labels = []
 
        # Create an array of all embedding arrays
        all_embeddings = np.concatenate([*map(np.vstack, all_embeddings)])

        # Find centroid that has the smallest distance to current chunk embedding (nearest neighbour)
        # BallTree algorithm (Cranenburgh2019)
        # Find k nearest neighbours to each centroid, with k being the number of chunks in a text
        all_predictions = []
        tree = BallTree(all_embeddings, metric='euclidean')
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

        # Get total count of each word
        dtm_sum = dtm.sum(axis=0)

        # Not possible to work with sparse matrices because dtm_sum is not sparse (does not contain any 0) and because broadcasting for subtraction of csr matrices is not implemented
        distances = {}
        for idx, file_name in tqdm(enumerate(data_dict['file_names'])):
            # if idx < 5: 
            # Access the term frequency values for the specific document
            file_counts = dtm[idx].toarray()
            corpus_counts = dtm_sum - file_counts
            cosine_distance = scipy.spatial.distance.cosine(corpus_counts, file_counts)
            distances[file_name] = cosine_distance
        # Turn both keys and values of a dict into columns of a df.
        distances = pd.DataFrame(distances.items(), columns=['file_name', f'{ngram_type}_distance'])
        return distances 



    # def get_corpus_distance(self, ngram_type, min_docs=None, min_percent=None, max_docs=None, max_percent=None):
    #     data_dict = self.ngrams[ngram_type]
    #     dtm, _ = self.nc.filter_dtm(data_dict, min_docs, min_percent, max_docs, max_percent)

    #     # Get total count of each word
    #     dtm_sum = dtm.sum(axis=0)
    #     dtm_sum_norm = np.linalg.norm(dtm_sum)
    #     dtm_sum_t = dtm_sum.T

    #     num_docs = len(data_dict['file_names'])
    #     batch_size = 100
    #     batch_size = batch_size or num_docs  # Use batch size equal to the number of documents if not specified.

    #     # Initialize an empty DataFrame to store distances
    #     distances = pd.DataFrame(columns=['file_name', f'{ngram_type}_distance'])

    #     for start_idx in tqdm(range(0, num_docs, batch_size)):
    #         start = time.time()
    #         end_idx = min(start_idx + batch_size, num_docs)

    #         # Calculate cosine distances for the current batch
    #         batch_dtm = dtm[start_idx:end_idx]
    #         batch_dtm_diff = dtm_sum - batch_dtm
    #         print('dtm_sum_norm', dtm_sum_norm, 'np.linalg.norm(batch_dtm_diff, axis=1)', np.linalg.norm(batch_dtm_diff, axis=1).shape)
    #         print('(batch_dtm_diff @ dtm_sum_t)', (batch_dtm_diff @ dtm_sum_t).shape, '(np.linalg.norm(batch_dtm_diff, axis=1) * dtm_sum_norm)', (np.linalg.norm(batch_dtm_diff, axis=1) * dtm_sum_norm).shape)
            
    #         numerator = (batch_dtm_diff @ dtm_sum_t)#.flatten()
    #         print('numerator', type(numerator), numerator.shape, numerator)
    #         # denominator = (np.linalg.norm(batch_dtm_diff, axis=1) * dtm_sum_norm)
    #         # denominator = denominator[:, np.newaxis] # Make it 2D
    #         # print('denominator', type(denominator), denominator.shape, denominator)
    #         # batch_dists = numerator / denominator
            
    #         # print('batch_dists', type(batch_dists), batch_dists.shape, batch_dists)

    #         # batch_dists = 1 - batch_dists

    #         # print('batch_dists', type(batch_dists), batch_dists.shape)

    #         # # Add distances for the current batch to the DataFrame
    #         # batch_file_names = data_dict['file_names'][start_idx:end_idx]
    #         # print(len(batch_file_names), batch_dists.flatten().shape)
    #         batch_dists = pd.DataFrame({'file_name': batch_file_names, f'{ngram_type}_distance': batch_dists.flatten()})
    #         distances = pd.concat([distances, batch_dists], ignore_index=True)
    #         print(f'{time.time()-start}s to process one batch.')
    #     return distances



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
    
    def get_dependency_labels_distribution(self, docs):
        doc_dist = Counter()
        chunk_dists = []
        for chunk in docs:
            # Extract dependency labels
            dependency_labels = [token.dep_ for token in chunk]

            # Calculate the distribution
            chunk_dist = Counter(dependency_labels)
            doc_dist.update(chunk_dist)
            chunk_dists.append(chunk_dist)
        return doc_dist, chunk_dists
            

    def get_sentence_dependencies(self):
        all_chunk_dists = []
        all_doc_dist = []

        # Use chunk-based processing for fulltext features because some texts are too long to be handled in one piece
        for doc_chunks in self.generate_chunks():
            chunks = {}
            first_chunk = True
            for chunk in doc_chunks:
                
                if first_chunk:
                    bookname = get_filename_from_path(chunk.doc_path) # Save book name for later use
                    first_chunk = False

                chunks[chunk.chunkname] = chunk.text
            docs = list(self.nlp.pipe(chunks.values(), batch_size=6, disable=['tagger', 'ner'], n_process=multiprocessing.cpu_count()-2))

            doc_dist, chunk_dists = self.get_dependency_labels_distribution(docs)

            # Convert the list of Counters to a list of dictionaries
            chunk_dists = [dict(counter) for counter in chunk_dists]
            df = pd.DataFrame(chunk_dists).fillna(0)
            df['file_name'] = list(chunks.keys())
            all_chunk_dists.append(df)

            doc_dist = dict(doc_dist)
            df = pd.DataFrame(doc_dist, index=[0]).fillna(0)
            df['file_name'] = bookname

            # Add 'sd_' prefix to every column label except 'file_name'
            df.columns = ['sentd_' + col if col != 'file_name' else col for col in df.columns]
            print(df.columns)
            
            all_doc_dist.append(df)


        result = []
        for dfs in [all_chunk_dists, all_doc_dist]:

            df = pd.concat(dfs, ignore_index=True, sort=False, axis=0)
            df = df.fillna(0)

            # Get relative frequencies
            df = df.set_index('file_name', inplace=False)
            df = df.div(df.sum(axis=1), axis=0)
            df = df.reset_index(inplace=False)

            result.append(df)

        return result


    def load_pickle(self, path):
        with open(path, 'rb') as f:
            features = pickle.load(f)
            return features
    
    def save_pickle(self, path, data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def call_function(self, func):
        print(func.__name__)
        path = os.path.join(self.pickle_dir, f'{func.__name__}_aschunk_{self.as_chunk}.pkl')
        if os.path.exists(path):
            features = self.load_pickle(path)
        else:
            features = func()
            self.save_pickle(path, features)
        return features
    
    def get_all_features(self):
        chunk_functions = [self.get_unigram_distance,
                        self.get_unigram_distance_limited,
                        self.get_bigram_distance,
                        self.get_trigram_distance,
                        self.get_tag_distribution]

        book_functions = [self.get_overlap_score_doc2vec,
                        self.get_outlier_score_doc2vec,
                        self.get_overlap_score_sbert,
                        self.get_outlier_score_sbert,
                        self.get_sentence_dependencies]

        chunk_features = []
        book_features = []

        
        for func in chunk_functions:
            features = self.call_function(func)
            chunk_features.append(features)
            
        if self.as_chunk:
            for func in book_functions:
                features = self.call_function(func)
                if func.__name__ == 'get_sentence_dependencies':
                    chunk_df, doc_df = features
                    chunk_features.append(chunk_df)
                    book_features.append(doc_df)
                else:
                    book_features.append(features)

        chunk_features = reduce(lambda df1, df2: df1.merge(df2, how='inner', on='file_name', validate='one_to_one'), chunk_features)
        if self.as_chunk:
            book_features = reduce(lambda df1, df2: df1.merge(df2, how='inner', on='file_name', validate='one_to_one'), book_features)

        return chunk_features, book_features
