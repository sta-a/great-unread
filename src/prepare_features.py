# %%
%load_ext autoreload
%autoreload 2

import os
import time
import sys
sys.path.append("..")
from utils import get_doc_paths, check_equal_files, DataHandler, get_bookname, get_doc_paths_sorted
from feature_extraction.embeddings import SbertProcessor
from feature_extraction.ngrams import NgramCounter, MfwExtractor
from feature_extraction.process_rawtext import Tokenizer, TextLoader, ChunkHandler
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt

class FeaturePreparer(DataHandler):
    '''
    Extract basic features that take a while to process before using more detailed processing.
    This is not necessary, since all data preparation steps are also called from the detailed processing, but is practical.
    '''
    def __init__(self, language):
        super().__init__(language, 'features')
        self.doc_paths = self.doc_paths[:10]  ##################################
        self.doc_paths = get_doc_paths_sorted(self.text_raw_dir)[:5]
        self.doc_paths = list(reversed(self.doc_paths))
        print('doc paths sorted', self.doc_paths)

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

        t = Tokenizer(self.language)
        t.create_all_data()

    def chunker(self):
        c = ChunkHandler(self.language, self.tokens_per_chunk)
        c.create_all_data()

    def textloader(self):
        for doc_path in self.doc_paths:
            t = TextLoader(self.language, self.tokens_per_chunk)
            text = t.load_data(doc_path=doc_path, as_chunk=False, lower=False, remove_punct=False)
            print(text)

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
        # self.tokenizer()
        # self.chunker()
        self.textloader()
        # self.ngramcounter()
        # self.mfwextractor()
        # self.sbert()
        print('Time: ', time.time()-start)


shortest_texts = {
    'eng': ['Kipling_Rudyard_How-the-Rhinoceros-Got-His-Skin_1902', 'Potter_Beatrix_Peter-Rabbit_1901', 'Kipling_Rudyard_How-the-Whale-Got-His-Throat_1902', 'Kipling_Rudyard_How-the-Camel-Got-His-Hump_1902', 'Kipling_Rudyard_The-Sing-Song-of-the-Old-Man-Kangaroo_1902', 'Kipling_Rudyard_The-Story-of-Muhammad-Din_1888', 'Kipling_Rudyard_The-Other-Man_1888', 'Kipling_Rudyard_Three-and-An-Extra_1888', 'Kipling_Rudyard_Venus-Annodomini_1888', 'Kipling_Rudyard_In-Error_1888'],
    'ger': ['Altenberg_Peter_Wie-wunderbar_1914', 'Hebel_Johann-Peter_Kannitverstan_1808', 'Wildermuth_Ottilie_Streit-in-der-Liebe-und-Liebe-im-Streit_1910', 'Kleist_Heinrich_Das-Bettelweib-von-Locarno_1810', 'Kleist_Heinrich_Unwahrscheinlich-Wahrhaftigkeiten_1811', 'Wackenroder_Wilhelm_Morgenlaendisches-Maerchen_1799', 'Rilke_Rainer-Maria_Die-Turnstunde_1899', 'Sacher-Masoch_Leopold_Lola_1907', 'Rilke_Rainer-Maria_Die-Weise-von-Liebe-und-Tod_1904', 'Moerike_Eduard_Die-Hand-der-Jezerte_1853']
}
remove = False
for language in ['eng']:
    fp = FeaturePreparer(language)
    fp.run()
# %%

# %%
