# %%
%load_ext autoreload
%autoreload 2

import os
import time
import sys
sys.path.append("..")
from utils import get_doc_paths, check_equal_files, DataHandler, get_filename_from_path, get_doc_paths_sorted
from feature_extraction.embeddings import D2vProcessor, SbertProcessor
from feature_extraction.ngrams import NgramCounter, MfwExtractor
from feature_extraction.process_rawtext import Tokenizer, ChunkHandler, SentenceTokenizer
import logging


class FeaturePreparer(DataHandler):
    '''
    Extract basic features that take a while to process before using more detailed processing.
    This is not necessary, since all data preparation steps are also called from the detailed processing, but is practical.
    '''
    def __init__(self, language):
        super().__init__(language, 'features')
        # self.doc_paths = self.doc_paths[:10]  ##################################
        self.doc_paths = get_doc_paths_sorted(self.text_raw_dir)[:5]
        # self.doc_paths = list(reversed(self.doc_paths))
        # print('doc paths sorted', self.doc_paths)


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
        if remove:
            chars_path = f'/home/annina/scripts/great_unread_nlp/src/special_chars_{language}.txt'
            if os.path.exists(chars_path):
                os.remove(chars_path)
            annotation_path = f'/home/annina/scripts/great_unread_nlp/src/annotation_words_{language}.txt'
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
            remove_files(f'/home/annina/scripts/great_unread_nlp/data/text_tokenized/{language}')
            remove_files('/home/annina/scripts/great_unread_nlp/data/preprocess/regex_checks')

        t = Tokenizer(self.language)
        t.create_all_data()

    def chunker(self):
        c = ChunkHandler(self.language, self.tokens_per_chunk)
        # c.create_all_data()
        c.check_data()


    def ngramcounter(self):
        nc = NgramCounter(self.language)
        # nc.create_all_data()
        # nc.check_data()
        
        s = time.time()
        print('loading from file')
        data_dict = nc.load_data(file_name='unigram_chunk')

        print(f'Time to load uni chunks: {time.time()-s}')

        s = time.time()
        file_counts = nc.load_values_for_chunk(file_name='Kipling_Rudyard_How-the-Rhinoceros-Got-His-Skin_1902_0', data_dict=data_dict)
        print('file_counts', file_counts)
        print(f'Time to load 1 chunk: {time.time()-s}')

    def mfwextractor(self):
        m = MfwExtractor(self.language)
        m.create_all_data()

    def d2v(self):
        d = D2vProcessor(self.language)
        d.create_all_data()

    def sbert(self):
        s = SbertProcessor(self.language)
        s.create_all_data()
        # s.check_data()

    def run(self):
        # self.load_text_tokenized()
        # self.sentence_tokenizer()
        # self.tokenizer()
        # self.chunker()
        # self.ngramcounter()
        # self.mfwextractor()
        # self.d2v()
        self.sbert()


remove = False
for language in ['eng', 'ger']:
    fp = FeaturePreparer(language)
    fp.run()




# %%
