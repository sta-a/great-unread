from multiprocessing import cpu_count
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.utils import shuffle
from tqdm import tqdm
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)


class Doc2VecVectorizer(object):
    def __init__(self,
                 dm=1,
                 dm_mean=1,
                 vector_size=100,
                 min_count=2,
                 window=4,
                 negative=5,
                 alpha=0.065,
                 min_alpha=0.065,
                 epochs=30,
                 seed=42,
                 n_cores=-1):
        self.dm = dm
        self.dm_mean = dm_mean
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.epochs = epochs
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
        self.d2v_model = Doc2Vec(dm=self.dm,
                                 dm_mean=self.dm_mean,
                                 vector_size=self.vector_size,
                                 window=self.window,
                                 negative=self.negative,
                                 alpha=self.alpha,
                                 min_alpha=self.min_alpha,
                                 min_count=self.min_count,
                                 workers=self.n_cores,
                                 seed=self.seed)
        # self.d2v_model = Doc2Vec([x for x in corpus_reader(doc_paths)],
        #                          dm=self.dm,
        #                          dm_mean=self.dm_mean,
        #                          workers=self.n_cores,
        #                          seed=self.seed)
        self.d2v_model.build_vocab(corpus_reader(doc_paths))
        for epoch in tqdm(list(range(self.epochs))):
            self.d2v_model.train(corpus_reader(doc_paths), total_examples=len(doc_paths), epochs=1)
            self.d2v_model.alpha -= 0.002
            self.d2v_model.min_alpha = self.d2v_model.alpha
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
