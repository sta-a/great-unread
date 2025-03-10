# %%
%load_ext autoreload
%autoreload 2

import os
import time
import sys
sys.path.append("..")
from utils import get_doc_paths, check_equal_files, DataHandler, get_filename_from_path, get_doc_paths_sorted
from feature_extraction.embeddings import D2vProcessor, SbertProcessor, RewriteSbertData
from feature_extraction.ngrams import NgramCounter, MfwExtractor
from feature_extraction.process_rawtext import Tokenizer, ChunkHandler, SentenceTokenizer


class FeaturePreparer(DataHandler):
    '''
    Extract basic features that take a while to process before extracting the more complex features.
    This is not necessary, since all data preparation steps are also called from the feature extraction script, but it is convenient.
    '''
    def __init__(self, language):
        super().__init__(language=language, output_dir='features')
        self.doc_paths = get_doc_paths_sorted(self.text_raw_dir)


    def sentence_tokenizer(self):
        t = SentenceTokenizer(self.language)
        t.create_all_data()

    def tokenizer(self, remove_files=False):
        def remove_files(dirpath):
            print(dirpath)
            if os.path.exists(dirpath):
                for item in os.listdir(dirpath):
                    item_path = os.path.join(dirpath, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)

        t = Tokenizer(self.language)
        t.create_all_data()

    def chunker(self):
        c = ChunkHandler(self.language, self.tokens_per_chunk)
        # c.create_all_data()
        c.check_data()


    def ngramcounter(self):
        nc = NgramCounter(self.language)
        nc.create_all_data()
        nc.check_data()
        
    def ngram_chunkloader(self):
        nc = NgramCounter(self.language)
        s = time.time()
        data_dict = nc.load_data(file_name='trigram_chunk')
        print(f'Time to load tri full: {time.time()-s}')

        s = time.time()
        if self.language == 'eng':
            counts = nc.load_values_for_chunk(file_name='Kipling_Rudyard_How-the-Rhinoceros-Got-His-Skin_1902_0', data_dict=data_dict, values_only=True)
            countdict = nc.load_values_for_chunk(file_name='Kipling_Rudyard_How-the-Rhinoceros-Got-His-Skin_1902_0', data_dict=data_dict, values_only=False)
            print(f'Time to load 1 chunk: {time.time()-s}')

    def ngramshapes(self):
        nc = NgramCounter(self.language)
        ngrams = nc.load_all_ngrams(as_chunk=False)
        uc = ngrams['unigram']['dtm']
        print(uc.shape)
        uc = ngrams['bigram']['dtm']
        print(uc.shape)
        uc = ngrams['trigram']['dtm']
        print(uc.shape)

        object_size = sys.getsizeof(ngrams) #############################a
        print(f"Size of the chunks of 1 doc: {object_size} bytes")


    def mfwextractor(self):
        m = MfwExtractor(self.language)
        m.create_all_data()

    def d2v(self):
        d = D2vProcessor(language=self.language, tokens_per_chunk=self.tokens_per_chunk)
        d.create_all_data()

    def sbert(self):
        s = SbertProcessor(language=self.language, tokens_per_chunk=self.tokens_per_chunk)
        s.create_all_data()
        s.check_data() 

    def rewrite_sbert(self):
        sb = RewriteSbertData(language=self.language, tokens_per_chunk=self.tokens_per_chunk)
        sb.process_and_save_data()
        
    def run(self):
        # self.sentence_tokenizer()
        # self.tokenizer()
        self.chunker()
        # self.ngramcounter()
        # self.ngram_chunkloader()
        # self.ngramshapes()
        # self.mfwextractor()
        # self.d2v()
        # self.sbert()
        # self.rewrite_sbert()


if __name__ == '__main__':
    for language in ['eng', 'ger']:
        fp = FeaturePreparer(language=language)
        fp.run()


# %%
