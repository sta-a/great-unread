# %%
%load_ext autoreload
%autoreload 2

import os
import time
import sys
sys.path.append("..")
from utils import get_doc_paths, check_equal_files, DataHandler, get_bookname
from feature_extraction.embeddings import SbertProcessor
from feature_extraction.ngrams import NgramCounter, MfwExtractor
from feature_extraction.process_rawtext import Tokenizer

class FeaturePreparer(DataHandler):
    '''
    Extract basic features that take a while to process before using more detailed processing.
    This is not necessary, since all data preparation steps are also called from the detailed processing, but is practical.
    '''
    def __init__(self, language):
        super().__init__(language, 'features')
        self.text_raw_dir = os.path.join(self.data_dir, 'text_raw', self.language)
        self.doc_paths = get_doc_paths(self.text_raw_dir)#[:None]  ##################################

    def load_text_tokenized(self):
        chunks = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk).create_data(self.doc_paths[0], remove_punct=False, lower=False, as_chunk=True)
        no_chunks = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk).create_data(self.doc_paths[0], remove_punct=False, lower=False, as_chunk=False)

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

        t = Tokenizer(self.language, self.doc_paths, self.tokens_per_chunk)
        t.create_all_data()
        t.check_data()

    def ngramcounter(self):
        c = NgramCounter(self.language)
        c.create_all_data()
        c.get_total_unigram_freq()
        c.load_data(file_name='unigram_chunk.pkl')
        print(c.data_dict['words'][1500:1600])
        # df = c.load_data('trigram_chunks.pkl')
        file_ngram_counts = c.load_values_for_chunk(file_name='Ainsworth_William-Harrison_Rookwood_1834_0')
        for k, v in file_ngram_counts.items():
            print(k, v)

    def mfwextractor(self):
        m = MfwExtractor(self.language)
        m.create_all_data()

    def sbert(self):
        s = SbertProcessor(self.language)
        s.create_all_data()

    def run(self):
        start = time.time()
        # self.load_text_tokenized()
        self.tokenizer()
        # self.ngramcounter()
        # self.mfwextractor()
        # self.sbert()
        print('Time: ', time.time()-start)





remove = False
for language in ['eng']:
    fp = FeaturePreparer(language)
    fp.run()
# %%

# %%
