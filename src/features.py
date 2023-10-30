# %%
%load_ext autoreload
%autoreload 2

import os
import pickle
import numpy as np
import pandas as pd
import shutil
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
from feature_extraction.doc_based_feature_extractor import DocBasedFeatureExtractor
from feature_extraction.corpus_based_feature_extractor import CorpusBasedFeatureExtractor
from feature_extraction.ngrams import NgramCounter
from feature_extraction.process_rawtext import ChunkHandler

import sys
sys.path.append("..")
from utils import DataHandler, get_filename_from_path, get_doc_paths_sorted


class FeatureProcessor(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'features', test=False)
        self.pickle_dir = self.text_raw_dir.replace('/text_raw', '/pickle')
        os.makedirs(self.pickle_dir, exist_ok=True)

        # self.doc_paths = get_doc_paths_sorted(self.text_raw_dir)[:5]

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            features = pickle.load(f)
            return features
    
    def save_pickle(self, path, datatuple):
        with open(path, 'wb') as f:
            pickle.dump(datatuple, f)
    
    def get_doc_features_helper(self, doc_path, as_chunk, ngrams):
        fe = DocBasedFeatureExtractor(language=self.language, doc_path=doc_path, as_chunk=as_chunk, tokens_per_chunk=self.tokens_per_chunk, ngrams=ngrams)
        chunk_features, book_features = fe.get_all_features()
        return chunk_features, book_features
    

    def get_doc_features(self, as_chunk):
        all_chunk_features = []
        all_book_features = []

        # Load ngrams only once for efficiency
        nc = NgramCounter(self.language)
        ngrams = nc.load_all_ngrams(as_chunk=as_chunk)

        for doc_path in tqdm(self.doc_paths):
            chunk_features, book_features = self.get_doc_features_helper(doc_path, as_chunk, ngrams)
            all_chunk_features.extend(chunk_features)
            all_book_features.append(book_features)

        # Save book features only once (not when running with fulltext chunks)
        return all_chunk_features, all_book_features

    
    def get_corpus_features(self, as_chunk):
        cbfe = CorpusBasedFeatureExtractor(self.language, self.doc_paths, as_chunk, self.pickle_dir, self.tokens_per_chunk)
        chunk_features, book_features = cbfe.get_all_features()
        # Save book features only once (not when running with fulltext chunks)
        return chunk_features, book_features
    
    def merge_features(self, doc_chunk_features, doc_book_features, doc_chunk_features_fulltext, corpus_chunk_features, corpus_book_features, corpus_chunk_features_fulltext):
        # Book features
        doc_book_features = pd.DataFrame(doc_book_features)
        doc_chunk_features = pd.DataFrame(doc_chunk_features)
        doc_chunk_features_fulltext = pd.DataFrame(doc_chunk_features_fulltext)

        for df in [doc_chunk_features, doc_book_features, doc_chunk_features_fulltext, corpus_chunk_features, corpus_book_features, corpus_chunk_features_fulltext]:
            print(df.shape)


        book_df = doc_book_features\
                    .merge(right=doc_chunk_features_fulltext, on='file_name', how='outer', validate='one_to_one')\
                    .merge(right=corpus_book_features, on='file_name', validate='one_to_one')\
                    .merge(right=corpus_chunk_features_fulltext, on='file_name', validate='one_to_one')
        book_df.columns = [col + '_full' if col != 'file_name' else col for col in book_df.columns]

        # Chunk features
        chunk_df = doc_chunk_features.merge(right=corpus_chunk_features, on='file_name', how='outer', validate='one_to_one')
        # Remove chunk id from file_name
        chunk_df['file_name'] = chunk_df['file_name'].str.split('_').str[:4].str.join('_')
        chunk_df.columns = [col + '_chunk' if col != 'file_name' else col for col in chunk_df.columns]

        # Combine book features and averages of chunksaveraged chunk features
        # baac: book and averaged chunk
        baac_df = book_df.merge(chunk_df.groupby('file_name').mean().reset_index(drop=False), on='file_name', validate='one_to_many')
        # cacb: chunk and copied book
        cacb_df = chunk_df.merge(right=book_df, on='file_name', how='outer', validate='many_to_one')
        print(book_df.shape, chunk_df.shape, baac_df.shape, cacb_df.shape)

        dfs = {'book': book_df, 'baac': baac_df, 'chunk': chunk_df, 'cacb': cacb_df}
        for file_name, df in dfs.items():
            self.save_data(data=df, file_name=file_name)

    
    def run(self):
        start = time.time()
        path = os.path.join(self.pickle_dir, 'doc_chunk.pkl')
        if os.path.exists(path):
            doc_chunk_features, doc_book_features = self.load_pickle(path)
        else:
            doc_chunk_features, doc_book_features = self.get_doc_features(as_chunk=True)
            self.save_pickle(path, (doc_chunk_features, doc_book_features))
        print(f'Doc features: {time.time()-start}s.')

        start = time.time()
        path = os.path.join(self.pickle_dir, 'doc_full.pkl')
        if os.path.exists(path):
            doc_chunk_features_fulltext = self.load_pickle(path)
        else:
            # Recalculate the chunk features for the whole book, which is treated as one chunk
            doc_chunk_features_fulltext, _ = self.get_doc_features(as_chunk=False)
            self.save_pickle(path, (doc_chunk_features_fulltext))
        print(f'Doc full features: {time.time()-start}s.')


        start = time.time()
        path = os.path.join(self.pickle_dir, 'corpus_chunk.pkl')
        if os.path.exists(path):
            corpus_chunk_features, corpus_book_features = self.load_pickle(path)
        else:
            corpus_chunk_features, corpus_book_features = self.get_corpus_features(as_chunk=True)
            self.save_pickle(path, (corpus_chunk_features, corpus_book_features))
        print(f'Corpus features: {time.time()-start}s.')
    

        start = time.time()
        path = os.path.join(self.pickle_dir, 'corpus_full.pkl')
        if os.path.exists(path):
            #corpus_chunk_features_fulltext = self.load_pickle(path)[0]
            corpus_chunk_features_fulltext = self.load_pickle(path)
        else:
            # Recalculate the chunk features for the whole book, which is considered as one chunk
            corpus_chunk_features_fulltext, _ = self.get_corpus_features(as_chunk=False)
            self.save_pickle(path, (corpus_chunk_features_fulltext))   
        print(f'Corpus full features: {time.time()-start}s.')   

        
        self.merge_features(
            doc_chunk_features,
            doc_book_features,
            doc_chunk_features_fulltext,
            corpus_chunk_features,
            corpus_book_features,
            corpus_chunk_features_fulltext
        )
        

for language in ['ger']:
    fe = FeatureProcessor(language).run()
















 # %%
import os
import pickle
import numpy as np
import pandas as pd
import shutil
import time
from tqdm import tqdm
from feature_extraction.process_rawtext import ChunkHandler

import sys
sys.path.append("..")
from utils import DataHandler, get_filename_from_path, get_doc_paths_sorted


class FeatureChecker(DataHandler):
    def __init__(self, language):
        super().__init__(language, 'features')
        self.dfs = self.load_dfs()
        for n, d in self.dfs.items():
            print(n,d.shape)
        self.outfile = open(f'featurechecker_{self.language}.txt', 'w')


    def load_dfs(self):
        dataframes = {}
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.output_dir, filename)
                df_name = get_filename_from_path(file_path)
                dataframes[df_name] = pd.read_csv(file_path)# .iloc[:50, :50] #########################
        return dataframes
    
    
    def check_columns(self, df):
        df = df.drop('file_name', inplace=False, axis=1)
        # Check for columns with duplicated values
        # duplicated_cols = df.T.duplicated()
        # assert not duplicated_cols.any()

        # Finding columns with identical values
        duplicate_cols = []
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j and (df[col1].reset_index(drop=True).equals(df[col2].reset_index(drop=True))):
                    duplicate_cols.append((col1, col2))
        # Displaying the result
        if duplicate_cols:
            self.outfile.write(f"\nColumns with identical values: \n")
            for tup in duplicate_cols:
                line = f"{tup[0]}, {tup[1]}\n"
                self.outfile.write('\t' + line)
        else:
            self.outfile.write(f"\nNo columns with identical values.")


        # Check if cols contain values that are not int, float or np.nan
        non_float_values = {}
        for col in df.columns:
            vals = df[col][~df[col].apply(lambda x: isinstance(x, (int, float, np.nan)))]
            if not vals.empty:
                non_float_values[col] = vals
        
        if non_float_values:
            for col, values in non_float_values.items():
                self.outfile.write(f"\nColumn '{col}' has non-float values:")
                for value in values:
                    self.outfile.write(f"\n   - '\tValue: {value}, Type: {type(value)}")
        else:
            self.outfile.write(f"\nNo non-float values:")


        # Find correlated columns
        cols_to_keep = [col for col in df.columns if not 'embedding' in col]
        dfck = df[cols_to_keep]
        print('cols_to_keep', dfck.columns)
        correlation_matrix = dfck.corr(numeric_only=False)
        correlated_columns = []
        for col1 in correlation_matrix.columns:
            for col2 in correlation_matrix.columns:
                if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.95:
                    correlated_columns.append((col1, col2))
        if correlated_columns:
            self.outfile.write(f"\nCorrelated columns:\n")
            for tup in correlated_columns:
                line = f"{tup[0]}, {tup[1]}\n"
                self.outfile.write('\t' + line)
        else:
            self.outfile.write(f"\nNo correlated columns.")
    
    def check_all_dfs(self):
        for name, df in self.dfs.items():
            self.outfile.write(f'\n------------------------------\n {name}')
            # Check for missing values
            cols_with_missing_values = df.columns[df.isnull().any()].tolist()
            if cols_with_missing_values:
                self.outfile.write(f"\nColumns with missing values:\n")
                self.outfile.write('\n\t'.join(cols_with_missing_values))
            else:
                self.outfile.write(f"\nNo missing values in any column.")

            # Check if there are columns where all values are 0
            assert not (df == 0).all().any()

            # Calculate the proportion of non-zero values in each column
            non_zero_proportions = (df != 0).sum() / len(df)
            # Find columns where the proportion of non-zero values is below the threshold
            mostly_zero_columns = non_zero_proportions[non_zero_proportions < 0.5].index.tolist()
            if mostly_zero_columns:
                self.outfile.write(f"\nColumns with mostly zero values:\n")
                self.outfile.write('\n\t'.join(mostly_zero_columns))
            else:
                self.outfile.write(f"\nNo columns with mostly zero vlaues.") 
           
            # Check for rows with duplicated values
            duplicated_rows = df.duplicated()
            # Print the boolean array of duplicated rows
            assert not duplicated_rows.any()

            # Check for rows with duplicated values while ignoring the file name
            nofn = df.drop('file_name', inplace=False, axis=1)
            duplicated_rows = nofn.duplicated()
            # Print the boolean array of duplicated rows
            assert not duplicated_rows.any()
            self.check_columns(df)
   
    
    def check_book(self):
        for key in ['book', 'baac']:
            df = self.dfs[key]
            assert df.shape[0] == self.nr_texts

    def check_chunk(self):
        ch = ChunkHandler(self.language, self.tokens_per_chunk)
        nr_chunks_per_doc, total_nr_chunks = ch.DataChecker(self.language, ch.output_dir).count_chunks_per_doc()
        for key in ['chunk', 'cacb']:
            df = self.dfs[key]
            assert df.shape[0] == total_nr_chunks

    # def merge_dfs(self):
    #     # Unnecessary function
    #     bookdf = self.dfs['book'].reset_index(drop=True).merge(self.dfs['baac'].reset_index(drop=True), how='inner', on='file_name', validate='1:1')

    #     assert all(self.dfs['chunk']['file_name'] == self.dfs['cacb']['file_name'])
    #     chunkdf = pd.concat([self.dfs['chunk'].reset_index(drop=True), self.dfs['cacb'].reset_index(drop=True)], axis=1)

    #     # Because of concatenation, df can contain duplicated column names
    #     # Check for duplicated column names
    #     duplicated_columns = chunkdf.columns[chunkdf.columns.duplicated()]
    #     # Check if cols with duplicated column names contain the same values
    #     if not duplicated_columns.empty:
    #         dup = []
    #         for col in duplicated_columns:
    #             dupdf = chunkdf[col]
    #             # Check if all columns have the same values per row
    #             same_values_per_row = (dupdf.apply(lambda x: x.nunique(), axis=1) == 1).all()
    #             dup.append(same_values_per_row)
    #     # Drop duplicated columns
    #     if all(dup):
    #         chunkdf = chunkdf.loc[:, ~chunkdf.columns.duplicated()]

    #     for name, df in {'bookdf': bookdf, 'chunkdf': chunkdf}.items():
    #         self.outfile.write(f"\nChecking combined df {name}\n-------------------------- \n")
    #         df = df.drop('file_name', inplace=False, axis=1)
    #         # Check for columns with duplicated values

    #         # Finding columns with identical values
    #         duplicate_cols = []
    #         for i, col1 in enumerate(df.columns):
    #             for j, col2 in enumerate(df.columns):
    #                 if i < j and (df[col1].reset_index(drop=True).equals(df[col2].reset_index(drop=True))):
    #                     duplicate_cols.append((col1, col2))
    #         # Displaying the result
    #         if duplicate_cols:
    #             self.outfile.write(f"\nColumns with identical values in combined {name}: \n")
    #             for tup in duplicate_cols:
    #                 line = f"{tup[0]}, {tup[1]}\n"
    #                 self.outfile.write('\t' + line)
    #         else:
    #             self.outfile.write(f"\nNo columns with identical values in combined {name}.")


for language in ['ger']:
    fc = FeatureChecker(language)
    fc.check_book()
    fc.check_chunk()
    fc.check_all_dfs()




# %%
