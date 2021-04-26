import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from multiprocessing import cpu_count
from gensim.models import Doc2Vec
from transformers import BertModel
from gensim.models.doc2vec import TaggedDocument
from sklearn.utils import shuffle
from tqdm import tqdm
import pickle
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)


class Doc2VecVectorizer(object):
    def __init__(self,
                 dm=1,
                 dm_mean=1
                 seed=42,
                 n_cores=-1):
        self.dm = dm
        self.dm_mean = dm_mean
        self.seed = seed
        if n_cores == -1 or n_cores is None:
            self.n_cores = cpu_count()
        else:
            self.n_cores = n_cores
        self.d2v_model = None
        self.doc_paths = None
    
    def fit(self, doc_paths):
        logging.info("Fitting Doc2VecVectorizer...")
        def corpus_reader(doc_paths):
            doc_paths_and_ids = [(doc_id, doc_path) for doc_id, doc_path in enumerate(doc_paths)]
            for doc_id, doc_path in shuffle(doc_paths_and_ids):
                with open(doc_path, "r") as reader:
                    words = reader.read().strip().split()
                    yield TaggedDocument(words=words, tags=[f'doc_{doc_id}'])
        # self.d2v_model = Doc2Vec(dm=self.dm,
        #                          dm_mean=self.dm_mean,
        #                          vector_size=self.vector_size,
        #                          window=self.window,
        #                          negative=self.negative,
        #                          alpha=self.alpha,
        #                          min_alpha=self.min_alpha,
        #                          min_count=self.min_count,
        #                          workers=self.n_cores,
        #                          seed=self.seed)
        self.d2v_model = Doc2Vec([x for x in corpus_reader(doc_paths)],
                                 dm=self.dm,
                                 dm_mean=self.dm_mean,
                                 workers=self.n_cores,
                                 seed=self.seed)
        # self.d2v_model.build_vocab(corpus_reader(doc_paths))
        # for epoch in tqdm(list(range(self.epochs))):
        #     self.d2v_model.train(corpus_reader(doc_paths), total_examples=len(doc_paths), epochs=1)
        #     self.d2v_model.alpha -= 0.002
        #     self.d2v_model.min_alpha = self.d2v_model.alpha
        self.doc_paths = doc_paths
        logging.info("Fitted Doc2VecVectorizer.")
        return self
    
    def get_doc_vectors(self):
        vectors = []
        for doc_id, doc_path in enumerate(self.doc_paths):
            vectors.append(self.d2v_model.dv[f'doc_{doc_id}'])
        df = pd.DataFrame(vectors)
        df["doc_path"] = self.doc_paths
        return df


class BertVectorizer(object):
    def __init__(self, lang, sentence_to_doc_agg):
        self.lang = lang
        if self.lang == "eng":
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        elif self.lang == "ger":
            self.bert_model = BertModel.from_pretrained('bert-base-german-cased')
        else:
            raise Exception(f"Not a valid language {self.lang}")
        self.sentence_to_doc_agg = sentence_to_doc_agg
    
    def __get_sentence_vector(self, input_tokens):
        out = self.bert_model(**input_tokens)
        embeddings_of_last_layer = out[0]
        cls_embeddings = embeddings_of_last_layer[0]
        sentence_vector = cls_embeddings.detach().numpy().mean(axis=0)
        return sentence_vector

    def fit(self, pickle_paths):
        logging.info("Running BertVectorizer...")
        doc_vectors = []
        for pickle_path in tqdm(pickle_paths):
            input_token_list = pickle.load(open(pickle_path, "rb"))
            doc_vector = None
            for index, input_tokens in enumerate(input_token_list):
                if self.sentence_to_doc_agg == "first":
                    if index == 0:
                        doc_vector = self.__get_sentence_vector(input_tokens)
                    else:
                        break
                elif self.sentence_to_doc_agg == "mean":
                    if index == 0:
                        doc_vector = self.__get_sentence_vector(input_tokens)
                    else:
                        doc_vector = (doc_vector * index + self.__get_sentence_vector(input_tokens)) / (index + 1)
                elif self.sentence_to_doc_agg == "last":
                    if index < len(input_token_list) - 1:
                        continue
                    else:
                        doc_vector = self.__get_sentence_vector(input_tokens)
                else:
                    raise Exception(f"Not a valid sentence_to_doc_agg: {self.sentence_to_doc_agg}")
            doc_vectors.append(doc_vector)
        self.doc_vectors = pd.DataFrame(doc_vectors)
        self.doc_vectors["pickle_path"] = pickle_paths
        logging.info("Finished BertVectorizer.")
        return self
    
    def get_doc_vectors(self):
        return self.doc_vectors
