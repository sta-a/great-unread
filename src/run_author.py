# %%
'''
This script prepares data for running the whole pipeline based on texts where all works of an author are combined into one.
The default data_dir in the class DataHandler needs to be set to "data".
'''
%load_ext autoreload
%autoreload 2
from utils import TextsByAuthor, DataHandler
import os
from statistics import mean
import pandas as pd
import numpy as np
import shutil
from feature_extraction.embeddings import SbertProcessor
from prepare_features import FeaturePreparer
from feature_extraction.process_rawtext import DataChecker


class AuthorCombiner(DataHandler):
    def __init__(self, language, output_dir, data_type):
        super().__init__(language, output_dir=output_dir, data_type=data_type)

        self.output_dir = self.output_dir.replace('data', 'data_author')
        if not os.path.exists(self.output_dir):
            self.create_dir(self.output_dir)
        self.tba = TextsByAuthor(self.language)

    def get_author_path(self, cdir, author, list_of_works):
        title = self.get_title(author, list_of_works)
        file_name = f'{title}.{self.data_type}' # Keep title 'all' for compatibility
        return os.path.join(cdir, file_name)
    
    def get_title(self, author, list_of_works):        
        year = self.get_average_year(list_of_works)
        title = f'{author}_all_{year}'
        return title
    
    def get_average_year(self, list_of_works):
        year = []
        for work in list_of_works:
            year.append(int(work[-4:]))
        return round(mean(year))

    def get_anon_path(self, cdir, title):
        file_name = f'{title}.{self.data_type}'
        return os.path.join(cdir, file_name) 


class AuthorTexts(AuthorCombiner):
    '''
    Combine texts by the same author into one file.
    '''
    def __init__(self, language, output_dir='text_raw'):
        super().__init__(language, output_dir=output_dir, data_type='txt')
        self.indir = self.text_raw_dir


    def combine_texts(self):
        # Combine texts by the same author into one text
        for author, list_of_works in self.tba.author_filename_mapping.items():

            if not 'Anonymous' in author:
                outpath = self.get_author_path(self.output_dir, author, list_of_works)
                with open(outpath, 'w') as f:
                    for title in list_of_works:
                        inpath = os.path.join(self.indir, f'{title}.txt')
                        with open(inpath, 'r') as infile:
                            content = infile.read()
                            f.write(content)

            else:
                # Treat anonymous authors and different authors, don't combine texts
                # Copy files, don't change title
                for title in list_of_works:
                    inpath = self.get_anon_path(self.indir, title)
                    outpath = self.get_anon_path(self.output_dir, title)
                    shutil.copy(inpath, outpath)


class AuthorTitleMapping(AuthorCombiner):
    '''
    Map new titles to original titles.
    '''
    def __init__(self, language, output_dir='title_mapping'):
        super().__init__(language, output_dir=output_dir, data_type='txt')


    def create_titles(self):
        title_mapping = []
        # Combine texts by the same author into one text
        for author, list_of_works in self.tba.author_filename_mapping.items():

            if not 'Anonymous' in author:
                newtitle = self.get_title(author, list_of_works)
                for title in list_of_works:
                    title_mapping.append((author, newtitle, title))

            else:
                for title in list_of_works:
                    title_mapping.append((author, title, title))

        title_mapping = pd.DataFrame(title_mapping, columns=['author', 'new_file_name', 'file_name'])
        print(title_mapping)
        self.save_data(data=title_mapping, data_type='csv', file_name=f'title_mapping')


class AuthorChunks(AuthorTexts):
    '''
    Combine chunks of texts by the same author into one file.
    Copy chunks instead of recalculating them from scratch because preprocessing depends on file paths to individual texts (not author-based).
    '''
    def __init__(self, language):
        super().__init__(language)
        self.output_dir = self.output_dir.replace('text_raw', f'text_chunks_tpc_{self.tokens_per_chunk}') # Can't pass output_dir because self.tokens_per_chunk must already be initialized
        if not os.path.exists(self.output_dir):
            self.create_dir(self.output_dir)
        self.indir = self.indir.replace('text_raw', f'text_chunks_tpc_{self.tokens_per_chunk}') 


class AuthorSbert(AuthorCombiner):
    '''
    Combine the sbert embeddings for texts by the same author into one.
    '''
    def __init__(self, language):
        super().__init__(language, output_dir='sbert_embeddings', data_type='npz')
        self.sp = SbertProcessor(self.language, self.tokens_per_chunk)


    def combine_sbert_embeddings(self):
        for author, list_of_works in self.tba.author_filename_mapping.items():

            if not 'Anonymous' in author:
                all_embeddings = []
                outpath = self.get_author_path(self.output_dir, author, list_of_works)
                for title in list_of_works:
                    curr_embeddings = self.sp.load_data(file_name=title)
                    all_embeddings.append(curr_embeddings)
                all_embeddings = np.concatenate(all_embeddings, axis=0)
                self.save_data(file_path=outpath, data=all_embeddings)

            else:
                for title in list_of_works:
                    inpath = self.get_anon_path(self.sp.output_dir, title)
                    outpath = self.get_anon_path(self.output_dir, title)
                    shutil.copy(inpath, outpath)

    def check_data(self, ac):
        # Check if there is one embedding for every sentence
        dc = DataChecker(self.language, ac.output_dir)
        _, sentences_per_doc = dc.count_sentences_per_chunk()

        file_names = []
        for filename in os.listdir(ac.output_dir):
            # Get the base name without extension
            fn = os.path.splitext(os.path.basename(filename))[0]
            file_names.append(fn)

        for fn in file_names:
            nr_sents = sentences_per_doc[fn]
            sbert = self.load_data(file_name=f'{fn}.npz')
            assert len(sbert) == nr_sents
            self.logger.debug(f'One embedding per sentence for {fn}.')


# data dir as data
for language in ['eng', 'ger']:
    at = AuthorTexts(language)
    at.combine_texts()
    atm = AuthorTitleMapping(language)
    atm.create_titles()
    ac = AuthorChunks(language)
    ac.combine_texts()
    asb = AuthorSbert(language)
    asb.combine_sbert_embeddings()
    asb.check_data(ac)



# %%
    
%load_ext autoreload
%autoreload 2
from utils import TextsByAuthor, DataHandler
import os
from statistics import mean
import numpy as np
import shutil
from feature_extraction.embeddings import SbertProcessor
from prepare_features import FeaturePreparer
from feature_extraction.process_rawtext import DataChecker


class FeaturePreparerAuthor(FeaturePreparer):
    '''
    Prepare only features that have not already been prepared above with the other author-specific classes.
    '''
    def __init__(self, language):
        super().__init__(language=language)

    def sbert(self):
        # Data has already been created with class AuthorSbert, just check
        s = SbertProcessor(language=self.language, tokens_per_chunk=self.tokens_per_chunk)
        s.check_data() 

    def run(self):
        self.ngramcounter()
        self.ngramshapes()
        self.mfwextractor()
        self.d2v()


# data_dir as data_author
for language in ['eng', 'ger']:
    fp = FeaturePreparerAuthor(language)
    fp.run()
# %%
