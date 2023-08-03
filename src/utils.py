# %%
# %load_ext autoreload
# %autoreload 2
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import logging
import Levenshtein

logging.basicConfig(level=logging.DEBUG)


class DataHandler():
    '''
    Base class for creating, saving, and loading data.
    '''
    def __init__(self, language=None, output_dir=None, data_type='csv', modes=None, tokens_per_chunk=500, data_dir='/home/annina/scripts/great_unread_nlp/data'):
        self.language = language
        self.data_dir = data_dir
        self.output_dir = self.create_output_dir(output_dir)
        self.data_type = data_type
        self.modes = modes
        self.tokens_per_chunk = tokens_per_chunk
        self.logger = logging.getLogger(__name__)

    def create_output_dir(self, output_dir):
        if self.data_dir is None or output_dir is None or self.language is None:
            output_dir = None
        else:
            output_dir = os.path.join(self.data_dir, output_dir, self.language)
        return output_dir

    def create_data(self,**kwargs):
        # Implement the distance calculation logic
        raise NotImplementedError
    
    def create_filename(self,**kwargs):
        return self.create_filename_base(**kwargs)

    # def create_filename_base(self, **kwargs):
    #     data_type = self.get_custom_datatype(**kwargs)
    #     if 'file_string' in kwargs:
    #         file_string = kwargs['file_string'] + '-'
    #     else:
    #         file_string = ''
    #     return f"{file_string}{kwargs['mode']}.{data_type}"

    def create_filename_base(self, **kwargs):
        data_type = self.get_custom_datatype(**kwargs)
        if 'file_string' in kwargs:
            file_string = kwargs['file_string'] + '-'
        else:
            file_string = ''
        kwargs_str = '_'.join(f"{str(value)}" for key, value in kwargs.items() if key != 'file_string')
        return f"{file_string}{kwargs_str}.{data_type}"


    def save_data(self, data, file_name=None, **kwargs):
        file_path  = self.get_file_path(file_name, **kwargs)
        dir_path = Path(file_path).parent
        if not os.path.exists(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=False)
        self.save_data_type(data, file_path,**kwargs)
    
    def get_custom_datatype(self, **kwargs):
        if 'data_type' in kwargs:
            data_type = kwargs['data_type']
        else:
            data_type = self.data_type
        return data_type

    def save_data_type(self, data, file_path, **kwargs):
        data_type = self.get_custom_datatype(**kwargs)

        if data_type == 'csv':
            data = data.sort_index(axis=0).sort_index(axis=1)
            print(data)
            data.to_csv(file_path, header=True, index=True)
        elif data_type == 'pkl':
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, -1)
        elif data_type == 'png':
            data.savefig(file_path, format="png")

    def file_exists_or_create(self, file_path=None, **kwargs):
        print('check if file exists:', file_path)
        if not self.file_exists(file_path, **kwargs):
            self.logger.info(f'Creating data for {self.create_filename(**kwargs)}.')
            self.create_data(**kwargs)

    def load_data(self, file_name=None, **kwargs):
        endings = ('.npz', '.csv', '.np', '.pkl')
        if file_name is not None and not file_name.endswith(endings):
            raise ValueError(f"The file extensions must be provided. Supported extensions are {' '.join(endings)}.")

        file_path = self.get_file_path(file_name, **kwargs)

        self.file_exists_or_create(file_path, **kwargs)

        data = self.load_data_type(file_path, **kwargs)
        self.logger.info(f'Loaded {file_path} from file.')
        return data
    
    def load_data_type(self, file_path,**kwargs):
        if self.data_type == 'csv':
            data = pd.read_csv(file_path, header=0, index_col=0)
        elif self.data_type == 'np':
            data = load_list_of_lines(file_path, 'np')
        elif self.data_type == 'pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        return data
    
    def load_all_data(self):
        all_data = {}
        for mode in self.modes:
            data  = self.load_data(mode=mode)
            all_data[mode] = data
        return all_data

    def get_file_path(self, file_name=None, **kwargs):
        if file_name is None and not kwargs:
            raise ValueError("Either 'file_name' or 'kwargs' must be provided.")
        if file_name is not None and kwargs:
            self.logger.info('Both file_name and kwargs were passed to get_file_path(). file_name is used, kwargs are ignored.')

        if file_name is None:
            file_name = self.create_filename(**kwargs)
        file_path = os.path.join(self.output_dir, file_name)
        return file_path
    
    def file_exists(self, file_name=None, **kwargs):
        return os.path.exists(self.get_file_path(file_name, **kwargs))


def compare_line_counts(dir1, dir2, extension):
    '''
    Extension: the extension of the files in dir2
    '''
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    for file_name in files1:
        base_name, ext = os.path.splitext(file_name)
        matching_file = f"{base_name}.{extension}"
        if matching_file in files2:
            file_path1 = os.path.join(dir1, file_name)
            file_path2 = os.path.join(dir2, matching_file)

            line_count1 = sum(1 for _ in open(file_path1))
            line_count2 = sum(1 for _ in open(file_path2))

            print(f"{base_name}:\n{line_count1} lines in {dir1}, \n{line_count2} lines in {dir2}")     


def compare_directories(dir1, dir2):
    files1 = set(filename for filename in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, filename)))
    files2 = set(filename for filename in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, filename)))

    common_files = files1.intersection(files2)
    unique_files_dir1 = files1 - common_files
    unique_files_dir2 = files2 - common_files

    if not unique_files_dir1 and not unique_files_dir2:
        print(f'The directories "{dir1}" and "{dir2}" contain the same files.')
    else:
        print('The directories do not contain the same files.')
        print('Files unique to dir1:', unique_files_dir1)
        print('Files unique to dir2:', unique_files_dir2)


def count_chunks(doc_paths):
    # Count chunks
    nr_chunks_per_doc = {}
    for doc_path in doc_paths:
        tokenized_words_path = doc_path.replace('/raw_docs', '/tokenized_words') 
        with open(tokenized_words_path, 'r') as f:
            nr_chunks = sum(1 for _ in f)
            nr_chunks_per_doc[doc_path] = nr_chunks
    #print(sorted(nr_chunks_per_doc.items(), key=lambda x: x[1]))
    total_nr_chunks = Counter(nr_chunks_per_doc.values())
    total_nr_chunks = sorted(total_nr_chunks.items(), key=lambda pair: pair[0], reverse=False)
    return nr_chunks_per_doc, total_nr_chunks

def get_bookname(doc_path):
    return Path(doc_path).stem


def get_doc_paths(docs_dir):
    doc_paths = sorted([os.path.join(docs_dir, doc_name) for doc_name in os.listdir(docs_dir) if Path(doc_name).suffix == '.txt'])
    return doc_paths

def get_filename_from_path(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]
    

def load_list_of_lines(path, line_type):
    if line_type == 'str':
        with open(path, 'r') as reader:
            lines = [line.strip() for line in reader]
    elif line_type == 'np':
        lines = np.load(path)['arr_0'].tolist()
    else:
        raise Exception(f'Not a valid line_type {line_type}')
    return lines


def save_list_of_lines(lst, path, line_type):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if line_type == 'str':
        with open(path, 'w') as f:
            for item in lst:
                f.write(str(item) + '\n')
    elif line_type == 'np':
        np.savez_compressed(path, np.array(lst))
    elif line_type == 'list':
        # list of lists of strings
        with open(path, 'w') as f:
            for clst in lst:
                f.write(','.join(clst) + '\n')
    else:
        raise Exception(f'Not a valid line_type {line_type}')
    

class TextsByAuthor(DataHandler):
    '''
    Map the filenames to the authors and count the number of texts per author.
    '''
    def __init__(self, language, filenames=None):
        super().__init__(language, output_dir='raw_docs')
        self.filenames = filenames
        if self.filenames is None:
            self.filenames = self.get_filenames()
        self.author_filename_mapping, self.nr_texts_per_author = self.create_data()

    def get_filenames(self):
        filenames = os.listdir(self.output_dir)
        print(filenames)
        filenames = [os.path.splitext(filename)[0] for filename in filenames]
        print(self.output_dir, filenames)
        return filenames
    

    def create_data(self):
        authors = []
        author_filename_mapping = {}
        #Get texts per authors
        for file_name in self.filenames:
            author = '_'.join(file_name.split('_')[:2])
            authors.append(author)
            if author in author_filename_mapping:
                author_filename_mapping[author].append(file_name)
            else:
                author_filename_mapping[author] = []
                author_filename_mapping[author].append(file_name)
                
        # Aggregate if author has collaborations with others
            agg_dict = {'Hoffmansthal_Hugo': ['Hoffmansthal_Hugo-von'], 
                        'Schlaf_Johannes': ['Holz-Schlaf_Arno-Johannes'],
                        'Arnim_Bettina': ['Arnim-Arnim_Bettina-Gisela'],
                        'Stevenson_Robert-Louis': ['Stevenson-Grift_Robert-Louis-Fanny-van-de', 
                                                    'Stevenson-Osbourne_Robert-Louis-Lloyde']}
            
        for author, aliases in agg_dict.items():
            if author in authors:
                for alias in aliases:
                    if alias in authors:
                        author_filename_mapping[author].extend(author_filename_mapping[alias]) 
                        del author_filename_mapping[alias]
                        authors = [author for author in authors if author != alias]
        
        nr_texts_per_author = Counter(authors)
        # author_filename_mapping: dict{author name: [list with all works by author]}
        # nr_texts_per_author: dict{author name: nr texts by author}
        return author_filename_mapping, nr_texts_per_author



# General function for finding a string in all code files
def search_string_in_files(directory, search_string):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        for line_number, line in enumerate(f, start=1):
                            if search_string in line:
                                print(f"Found '{search_string}' in '{file_path}' (Line {line_number}): {line.rstrip()}")

# # Provide the directory path and the string to search for
# directory_path = '/home/annina/scripts/great_unread_nlp/src'
# search_string = 'NgramCounter'
# search_string_in_files(directory_path, search_string)



class DataChecks(DataHandler):
    '''
    Compare the metadata with each other to check consistency.
    Compare file names of raw docs with metadata.
    '''

    def __init__(self, language):
        super().__init__(language) 
        self.language = language
        self.dfs = self.load_dfs()
        # self.print_cols()
        # self.compare_df_values()
        # self.compare_rawdocs_dfs()

    def load_dfs(self):
        # 'sentiscores', [], []),
        # dirs_with_files = [
        #     ('metadata', ['authors.csv'], ['author_viaf', 'name', 'first_name', 'gender']),
        #     ('metadata', [f'{self.language.upper()}_texts_meta.csv'], ['author_viaf', 'name', 'first_name', 'gender']),
        #     ('metadata', [f'{self.language.upper()}_texts_circulating-libs.csv'], ['author_viaf', 'name', 'first_name', 'gender']),
        #     ('canonscores', [], []),
        # ]
        dirs = ['metadata', 'canonscores', 'sentiscores']

        all_dfs = {}

        for dir_name in dirs:
            if dir_name == 'canonscores':
                dir_path = os.path.join(self.data_dir, dir_name)
            else:
                dir_path = os.path.join(self.data_dir, dir_name, self.language)
            filenames = os.listdir(dir_path)
            # or filename.endswith('.csv#') for opened libre office files
            assert all(filename.endswith('.csv') or filename.endswith('.csv#') for filename in filenames), f"Not all files have the '.csv' extension in {dir_path}."
            filenames = [file for file in filenames if file.endswith('.csv')]

            for filename in filenames:
                file_path = os.path.join(dir_path, filename)

                df = pd.read_csv(file_path, header=0, sep=None, engine='python')
                
                # Remove whitespace from author_viaf column
                if 'author_viaf' in df.columns:
                    df['author_viaf'] = df['author_viaf'].astype(str).str.replace(r'\s', '', regex=True)
                    # print(f'{filename}: Removed whitespace from author_viaf column.')

                all_dfs[filename] = df

                # print(f"File Path: {file_path}")
                # print(f"Columns: {', '.join(df.columns.tolist())}")
                # print('\n')
        return all_dfs

                
    def print_cols(self):
        # for key, df in self.dfs.items():
        #     print(key, df.columns)
        columns = []
        for key, df in self.dfs.items():
            columns.extend(df.columns.tolist())
        for col in set(columns):
            print(col)


    def find_duplicate_merge_keys(self, df, merge_keys):
        duplicate_rows = df[df.duplicated(subset=merge_keys, keep=False)]
        return duplicate_rows
    
    
    def is_sentifile(self, string_list):
        for string in string_list:
            if '_senti_' in string:
                return True
        return False


    def compare_df_values(self):
        # Compare every pair of dfs
        for key1, df1 in self.dfs.items():
            for key2, df2 in self.dfs.items():
                if key1 != key2:
                    dfs_dict = {key1: df1, key2: df2}
                    self.compare_dfs_pairwise(dfs_dict)

    def compare_dfs_pairwise(self, dfs_dict):

        for key, df in dfs_dict.items():
            if 'text_id' in df.columns and 'id' in df.columns:
                raise ValueError(f"{key} contains duplicate column names: {', '.join(df.columns)}")
            if 'text_id' in df.columns:
                df.rename(columns={'text_id': 'id'}, inplace=True)

        for key, df in dfs_dict.items():
            assert not df.columns.duplicated().any()

        keys = list(dfs_dict.keys())

        unique_cols = ['id', 'file_name'] # Cols which must have a unique value in every row
        main_cols = unique_cols + ['author_viaf'] # Cols to check
        # Cols from main cols that are in both dfs
        main_common_columns = list(set(dfs_dict[keys[0]].columns) & set(dfs_dict[keys[1]].columns) & set(main_cols))
        # Cols from unique cols that are in both dfs
        unique_common_cols = list(set(unique_cols) & set(main_common_columns))
        
        # Check if values that should be unique are duplicated
        # (Sentiscores files can have duplicated values because they have multiple entries for the same text if a review appeared in different magazines)
        if unique_common_cols:
            if not self.is_sentifile(keys):
                for key, df in dfs_dict.items():
                    for col in unique_common_cols:
                        assert df[col].nunique(dropna=True) == df[col].size, f'Column "{col}" not unique in {key}.'

                # Some dfs don't have entries for all texts
                # User 'inner' merge to ignore them
                try:
                    df = pd.merge(dfs_dict[keys[0]][unique_common_cols], dfs_dict[keys[1]][unique_common_cols], on=unique_common_cols, how='inner', validate='one_to_one')
                    assert not df.isna().any().any()
                except pd.errors.MergeError as e:
                    print(e)
            else:
                try:
                    df = pd.merge(dfs_dict[keys[0]][unique_common_cols], dfs_dict[keys[1]][unique_common_cols], on=unique_common_cols, how='outer')
                    assert not df.isna().any().any()
                except pd.errors.MergeError as e:
                    print(e)

        # Merge dfs to check if rows contain the same unique cols in both dfs
        if main_common_columns:
            try:
                df = pd.merge(dfs_dict[keys[0]][main_common_columns], dfs_dict[keys[1]][main_common_columns], on=main_common_columns, how='outer')
                assert not df.isna().any().any()
            except pd.errors.MergeError as e:
                print(e)


    def compare_rawdocs_dfs(self):
        raw_docs_dir = os.path.join(self.data_dir, 'raw_docs', self.language)
        rdfn = os.listdir(raw_docs_dir)
        assert all(filename.endswith('.txt') for filename in rdfn), "Not all files have the '.txt' extension."
        rdfn = set([x.rstrip('.txt') for x in rdfn])

        results = []
        for key, df in self.dfs.items():
            if 'file_name' in df.columns:
                dffn = set(df['file_name'])
                # if len(rdfn) == len(df['file_name']):
                #     print(f'All rd in {key}')

                common_files = rdfn.intersection(dffn)
                df_unique = sorted(dffn - common_files)
                # if not df_unique:
                #     print(f'No unique fn in {key}.')

                if df_unique:
                    for str1 in df_unique:
                        min_distance = 10000
                        mind_dist_string = ''
                        
                        for str2 in rdfn:
                            dist = Levenshtein.distance(str1, str2)
                            if dist < min_distance:
                                min_distance = dist
                                mind_dist_string = str2

                        # if min_distance > 0.3*len(str1):
                        results.append([str1, mind_dist_string, min_distance, key])

                # Create a pandas DataFrame
                fn_mapping = pd.DataFrame(results, columns=['metadata-fn', 'rawdocs-fn', 'edit-dist', 'file']).sort_values(by='edit-dist', ascending=True)
        fn_mapping.to_csv(f'compare_filenames_{self.language}.csv', index=False)

        # Manually check fn_mapping for distance where where the filenames are almost equal
        if self.language == 'eng':
            dist_cutoff = 3 
        else:
            dist_cutoff = 4

        fn_mapping = fn_mapping.loc[fn_mapping['edit-dist'] <= dist_cutoff].loc[:, ['metadata-fn', 'rawdocs-fn']]
        fn_mapping = fn_mapping.drop_duplicates()
        # Check if all columns contain only unique values
        assert (fn_mapping.nunique() == fn_mapping.shape[0]).all()
        return fn_mapping
    

    def prepare_metadata(self, data):
        fn_mapping = self.compare_rawdocs_dfs()
        fn_mapping = dict(zip(fn_mapping['metadata-fn'], fn_mapping['rawdocs-fn']))

        if data == 'canon':
            file_name = '210907_regression_predict_02_setp3_FINAL.csv'
            df = self.dfs[file_name]
            df = df.loc[:, ['file_name', 'm3']]
            df.loc[:, 'file_name'] = df['file_name'].replace(to_replace=fn_mapping)
        
        elif data == 'meta' or data == 'gender':
            df = self.dfs[f'{self.language.upper()}_texts_meta.csv']
            df.loc[:, 'file_name'] = df['file_name'].replace(to_replace=fn_mapping)

            if data == 'gender':
                authors = self.dfs['authors.csv']
                authors = authors.loc[:, ['author_viaf', 'gender']]
    
                df = df.merge(authors, how='left', on='author_viaf', validate='many_to_one')
                # Check if whitespace has been removed
                assert not df['author_viaf'].astype(str).str.contains(r'\s', na=False).any(), f'The author_viaf column contains whitespace.'
                df = df.loc[:, ['file_name', 'gender']]
                df.loc[:, 'gender'] = df['gender'].replace(to_replace={'w':'f'})

                # Set anonymous values to 'a'
                mask = df['file_name'].str.contains('anonymous_anonymous', case=False)
                # Set the value 'Female' in the 'Gender' column for the matched rows
                df.loc[mask, 'gender'] = 'a' # a for anonymous

                if self.language == 'ger':
                    df.loc[df['file_name'] == 'Hebel_Johann-Peter_Kannitverstan_1808', 'gender'] = 'm'
                    df.loc[df['file_name'] == 'May_Karl_Ardistan-und-Dschinnistan_1909', 'gender'] = 'm'
                    df.loc[df['file_name'] == 'May_Karl_Das-Waldroeschen_1883', 'gender'] = 'm'

                if self.language == 'eng':
                    df.loc[df['file_name'] == 'Somerville-Ross_Edith-Martin_Experience-of-an-Irish-RM_1899', 'gender'] = 'f'
                    df.loc[df['file_name'] == 'Somerville-Ross_Edith-Martin_The-Real-Charlotte_1894', 'gender'] = 'f'
                    df.loc[df['file_name'] == 'Stevenson-Grift_Robert-Louis-Fanny-van-de_The-Dynamiter_1885', 'gender'] = 'b' # Both male and female writer
                print(df['gender'].unique())
                # assert df['gender'].isin(['f', 'm', 'a', 'b']).all()
                # assert not df.isna().any().any()
                print(df['gender'].value_counts())
        return df

    def get_collaborations(self):
        df = self.prepare_metadata(data='meta')
        assert not df['author_viaf'].isna().any()
        # Check if author_viaf col contains anything besides numbers and the '|' character
        assert df['author_viaf'].astype(str).str.contains(r'[^0-9|]+').any() == False

        # Get author_viaf of authors that are involved in collaborations
        collabs = df[df['author_viaf'].astype(str).str.contains(r'\|', na=False, regex=True)]
        collabs.to_csv(f'collaborations-{self.language}.csv')
        no_collabs = df[~df['author_viaf'].astype(str).str.contains(r'\|', na=False, regex=True)]
        coll_av = df.loc[df['author_viaf'].astype(str).str.contains(r'\|', na=False, regex=True), 'author_viaf'].tolist()
        coll_av = list(set(val for item in coll_av for val in item.split('|')))

        # Find works by collaborating authors that are not collaborations
        # Eng: Stevenson_Robert-Louis is the only collaborating author that has other texts
        # Ger: Arnim_Bettina, Schlaf_Johannes
        nocolls = no_collabs[no_collabs['author_viaf'].isin(coll_av)]
        print(collabs, '\n\n', nocolls, '\n', coll_av)
                






# c = DataChecks('eng')
# # c.compare_df_values()
# # c.compare_rawdocs_dfs()
# c.get_collaborations()
# %%
