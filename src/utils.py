# %%
# %load_ext autoreload
# %autoreload 2
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import difflib
import hashlib
from collections import Counter
import joblib
import logging
import Levenshtein

logging.basicConfig(level=logging.DEBUG)


def get_bookname(doc_path):
    return Path(doc_path).stem

def get_filename_from_path(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]
    

def get_files_in_dir(files_dir):
    doc_paths = sorted([os.path.join(files_dir, doc_name) for doc_name in os.listdir(files_dir) if Path(doc_name).suffix == '.txt'])
    return doc_paths

def get_doc_paths(files_dir):
    return get_files_in_dir(files_dir)

def get_doc_paths_sorted(files_dir):
    paths = get_files_in_dir(files_dir)
    text_lengths= {}
    for path in paths:
        with open(path, 'r') as f:
            text = f.read().split()
            text_lengths[path] = len(text)
    text_lengths= dict(sorted(text_lengths.items(), key=lambda item: item[1]))
    return list(text_lengths.keys())


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
    

def find_overlapping_passages(path):
    def hash_lines(lines):
        return [hashlib.md5(line.encode()).hexdigest() for line in lines]

    def find_repeated_passages(text):
        lines = text.splitlines()
        line_hashes = hash_lines(lines)
        matches = []

        for i, line_hash in enumerate(line_hashes):
            for j, other_line_hash in enumerate(line_hashes[i+1:], start=i+1):
                if line_hash == other_line_hash:
                    match_ratio = difflib.SequenceMatcher(None, lines[i], lines[j]).ratio()
                    if match_ratio > 0.5:  # Adjust this threshold as needed
                        matches.append((i, j, match_ratio))

        return matches

    with open(path, 'r') as f:
        text = f.read()

    repeated_passages = find_repeated_passages(text)

    for match in repeated_passages:
        i, j, match_ratio = match
        print(f"Repeated passages found between lines {i+1} and {j+1} (Match ratio: {match_ratio:.2f})")
        print(text.splitlines()[i])
        print(text.splitlines()[j])
        print("=" * 40)



def check_equal_line_count(dir1, dir2, extension):
    '''
    Extension: the extension of the files in dir2
    '''
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    for file_name in files1:
        base_name, ext = os.path.splitext(file_name)
        matching_file = f"{base_name}.{extension}"
        if matching_file in files2:
            tok_path = os.path.join(dir1, file_name)
            chunk_path = os.path.join(dir2, matching_file)

            line_count1 = sum(1 for _ in open(tok_path))
            line_count2 = sum(1 for _ in open(chunk_path))
            if line_count1==line_count2:
                print(f'{base_name}: Equal nr lines in both dirs.')
                return True
            else:
                print(f'{base_name}: Unequal nr lines in both dirs.')
                print(f'Nr lines {dir1}: {line_count1}')
                print(f'Nr lines {dir2}: {line_count2}')
                return False


def check_equal_files(dir1, dir2):
    files1 = set(filename for filename in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, filename)))
    files2 = set(filename for filename in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, filename)))

    common_files = files1.intersection(files2)
    unique_files_dir1 = files1 - common_files
    unique_files_dir2 = files2 - common_files

    if not unique_files_dir1 and not unique_files_dir2:
        print(f'The directories "{dir1}" and "{dir2}" contain the same files.')
        return True
    else:
        print('The directories do not contain the same files.')
        print('Files unique to dir1:', unique_files_dir1)
        print('Files unique to dir2:', unique_files_dir2)
        return False
    

# # General function for finding a string in all code files
# def search_string_in_files(directory, search_string, extension):
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith(extension):
#                 file_path = os.path.join(root, file)
#                 if os.path.isfile(file_path):
#                     with open(file_path, 'r') as f:
#                         for line_number, line in enumerate(f, start=1):
#                             if search_string.lower() in line.lower():
#                                 print(f"Found '{search_string}' in '{os.path.basename(file_path)}' (Line {line_number}): \n{line.rstrip()}\n")

# General function for finding a string in all code files
def search_string_in_files(directory, search_string, extensions):
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        for line_number, line in enumerate(f, start=1):
                            if search_string.lower() in line.lower():
                                print(f"Found '{search_string}' in '{os.path.basename(file_path)}' (Line {line_number}): \n{line.rstrip()}\n")


class DataHandler():
    '''
    Base class for creating, saving, and loading data.
    '''
    def __init__(self, language=None, output_dir=None, create_outdir=False, data_type='csv', modes=None, tokens_per_chunk=500, data_dir='/home/annina/scripts/great_unread_nlp/data'):
        '''
        create_outdir: If True, create output dir when class is initialized.
        '''
        self.language = language
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        self.output_dir = self.create_output_dir(output_dir, create_outdir)
        self.text_raw_dir = os.path.join(self.data_dir, 'text_raw', language)
        self.doc_paths = get_doc_paths(self.text_raw_dir)
        self.data_type = data_type
        self.modes = modes
        self.tokens_per_chunk = tokens_per_chunk
        self.data_types = ('.npz', '.csv', '.np', '.pkl', '.txt', '.svg')
        self.separator = 'ƒ'
        self.subdir = None
        self.print_logs = False
        if self.language == 'eng':
            self.nr_texts = 605
        else:
            self.nr_texts = 547

    def create_dir(self, dir_path):
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path) # Create dir and parent dirs if necessary
                print(f"Directory '{dir_path}' created successfully.")
            except OSError as e:
                print(f"Error creating directory '{dir_path}\n': {e}")
            self.logger.info(f'{self.__class__.__name__}: Created dir {dir_path}')

    
    def add_subdir(self, subdir):
        self.subdir = os.path.join(self.output_dir, subdir)
        if not os.path.exists(self.subdir):
            self.create_dir(self.subdir)


    def create_output_dir(self, output_dir, create_outdir):
        if self.data_dir is None or output_dir is None or self.language is None:
            output_dir = None
        else:
            output_dir = os.path.join(self.data_dir, output_dir, self.language)
            if not os.path.exists(output_dir) and create_outdir:
                self.create_dir(output_dir)
        return output_dir

    def create_data(self,**kwargs):
        raise NotImplementedError
    
    
    def save_data(self, data, file_name=None, **kwargs):
        file_path  = self.get_file_path(file_name, **kwargs)
        dir_path = Path(file_path).parent
        if not os.path.exists(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=False)
        self.save_data_type(data, file_path,**kwargs)
    

    def save_data_type(self, data, file_path, **kwargs):
        data_type = self.get_custom_datatype(**kwargs)

        if data_type == 'csv':
            # data = data.sort_index(axis=0).sort_index(axis=1)
            print('Data not sorted before saving.')
            sep = ','
            if 'pandas_sep' in kwargs:
                sep = kwargs['pandas_sep']
            index = True
            if 'pandas_index' in kwargs:
                index = kwargs['pandas_index']
            data.to_csv(file_path, header=True, sep=sep, index=index)
        elif data_type == 'pkl':
            joblib.dump(data, file_path)
        elif data_type == 'svg':
            data.savefig(file_path, format="svg")
        elif data_type =='dict':
            with open(file_path, 'w') as f:
                for key, value in data.items():
                    f.write(f"{key}, {value}\n")
        elif data_type == 'txt':
            assert isinstance(data, list)
            list_of_strings = all(isinstance(element, str) for element in data)
            list_of_lists = all(isinstance(element, list) for element in data)
            assert list_of_strings or list_of_lists, "Input data should contain either a list of strings or a list of lists of strings"

            if list_of_strings:
                with open(file_path, 'w') as f:
                    for s in data:
                        f.write(str(s) + '\n')
            elif list_of_lists:
                # list of lists of strings
                with open(file_path, 'w') as f:
                    for l in data:
                        f.write(f'{self.separator.join(l)}\n')
                self.logger.info(f'Writing list of lists to file using {self.separator} as the separator.')


    def file_exists_or_create(self, file_path=None, file_name=None, **kwargs):
        if (file_path is None and file_name is None) or (file_path is not None and file_name is not None):
            raise ValueError("Either 'file_name' or 'file_path' must be provided.")
        if file_path is None:
            file_path = self.get_file_path(file_name, **kwargs)

        if not self.file_exists(file_path=file_path):
            if self.print_logs:
                self.logger.info(f'{self.__class__.__name__}: Creating data for {file_path}.')
            self.create_data(**kwargs)
        # else:
        #     if self.print_logs:
        #         self.logger.info(f'{self.__class__.__name__}: already exists: {file_path}')

    def load_data(self, load=True, file_name=None, **kwargs):
        file_path = self.get_file_path(file_name=file_name, **kwargs)
        self.file_exists_or_create(file_path=file_path, **kwargs)

        data = None
        if load:
            if self.print_logs:
                self.logger.info(f'{self.__class__.__name__}: Loading {file_path} from file.')
            data = self.load_data_type(file_path, **kwargs)
        return data
    
    def load_data_type(self, file_path,**kwargs):
        if self.data_type == 'csv':
            sep = ','
            if 'pandas_sep' in kwargs:
                sep = kwargs['pandas_sep']
            data = pd.read_csv(file_path, header=0, index_col=0, sep=sep)
        elif self.data_type == 'np':
            data = data = np.load(file_path)['arr_0'].tolist()
        elif self.data_type == 'pkl':
            data = joblib.load(file_path)
        elif self.data_type == 'txt':
            with open(file_path, 'r') as f:
                # Return a list of strings
                data = f.readlines()
                data = [line.rstrip('\n') for line in data]            
        return data

    def create_all_data(self):
        # Check if file exists and create it if necessary
        for mode in self.modes:
            print(mode)
            _ = self.load_data(load=False, mode=mode)

    
    def load_all_data(self):
        # Check if file exists, create it if necessary, return all data
        all_data = {}
        for mode in self.modes:
            data  = self.load_data(load=True, mode=mode)
            all_data[mode] = data
        return all_data
    
    def get_custom_datatype(self, **kwargs):
        if 'data_type' in kwargs:
            data_type = kwargs['data_type']
        else:
            data_type = self.data_type
        return data_type

    def create_filename(self, **kwargs):
        data_type = self.get_custom_datatype(**kwargs)
        if 'file_string' in kwargs:
            file_string = kwargs['file_string'] + '-'
        else:
            file_string = ''

        # If kwargs are tuples, turn them into strings
        for key, value in kwargs.items():
            if isinstance(value, tuple):
                kwargs[key] = '_'.join(value)

        kwargs_str = '_'.join(f"{str(value)}" for key, value in kwargs.items() if key != 'file_string')
        file_name = f"{file_string}{kwargs_str}.{data_type}"
        return file_name
    
    def validate_filename(self, file_name, **kwargs):
        ending_count = file_name.count('.')
        if ending_count == 0:
            data_type = self.get_custom_datatype(**kwargs)
            file_name = f'{file_name}.{data_type}'
            if self.print_logs:
                self.logger.info(f'{self.__class__.__name__}: Added extension to file name: {file_name}')
        elif ending_count == 1:
            if not file_name.endswith(self.data_types):
                raise ValueError(f'Invalid file extension: {file_name}')
        else:
            raise ValueError(f'Multiple file extension: {file_name}')
        return file_name

    
    def get_file_path(self, file_name=None, **kwargs):
        if file_name is None and not kwargs:
            raise ValueError("Either 'file_name' or kwargs must be provided.")
        elif file_name is not None and kwargs and self.print_logs:
            self.logger.info(f'{self.__class__.__name__}: Both file_name and kwargs were passed to get_file_path().') # file_name is used, kwargs are ignored. \nfile_name: {file_name}. \nkwargs: {kwargs}')
        if file_name is None:
            file_name = self.create_filename(**kwargs)
        file_name = self.validate_filename(file_name=file_name, **kwargs)

        pathdir = self.output_dir
        if 'subdir' in kwargs:
            if isinstance(kwargs['subdir'], bool):
                assert self.subdir is not None, f'Subdir must be initialized or passed to get_file_path().'
            elif isinstance(kwargs['subdir'], str):
                assert self.subdir is None, f'Subdir is already initialized.'
                self.add_subdir(kwargs['subdir'])
            pathdir = self.subdir
        file_path = os.path.join(pathdir, file_name)
        return file_path
    
    def file_exists(self, file_path=None, file_name=None, **kwargs):
        if (file_path is None and file_name is None) or (file_path is not None and file_name is not None):
            raise ValueError("Either 'file_name' or 'file_path' must be provided.")
        if file_path is None:
            file_path = self.get_file_path(file_name, **kwargs)
        return os.path.exists(file_path)
    

class TextsByAuthor(DataHandler):
    '''
    Map the filenames to the authors and count the number of texts per author.
    '''
    def __init__(self, language, filenames=None):
        super().__init__(language, output_dir='text_raw')
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


class DataChecks(DataHandler):
    '''
    Compare the metadata with each other to check consistency.
    Compare file names of raw docs with metadata.
    '''

    def __init__(self, language):
        super().__init__(language, data_type='csv') 
        self.language = language
        self.dfs = self.load_dfs()
        self.output_dir = os.path.join(self.data_dir, 'corpus_corrections')
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
            assert all(filename.endswith('.csv') or filename.endswith('.csv#') for filename in filenames), f"Not all files in {dir_path} have the '.csv' extension."
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
        text_raw_dir = os.path.join(self.data_dir, 'text_raw', self.language)
        rdfn = os.listdir(text_raw_dir)
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

        # Manually check fn_mapping for distance where where the filenames are almost equal
        if self.language == 'eng':
            dist_cutoff = 3 
        else:
            dist_cutoff = 4

        fn_mapping = fn_mapping.loc[fn_mapping['edit-dist'] <= dist_cutoff].loc[:, ['metadata-fn', 'rawdocs-fn']]
        fn_mapping = fn_mapping.drop_duplicates()
        self.save_data(data=fn_mapping, file_name=f'compare_filenames_{self.language}.csv', pandas_index=False)
        # fn_mapping.to_csv(f'compare_filenames_{self.language}.csv', index=False)
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
        self.save_data(data=collabs, file_name=f'collaborations-{self.language}.csv', pandas_index=False)
        # collabs.to_csv(f'collaborations-{self.language}.csv')
        no_collabs = df[~df['author_viaf'].astype(str).str.contains(r'\|', na=False, regex=True)]
        coll_av = df.loc[df['author_viaf'].astype(str).str.contains(r'\|', na=False, regex=True), 'author_viaf'].tolist()
        coll_av = list(set(val for item in coll_av for val in item.split('|')))

        # Find works by collaborating authors that are not collaborations
        # Eng: Stevenson_Robert-Louis is the only collaborating author that has other texts
        # Ger: Arnim_Bettina, Schlaf_Johannes
        nocolls = no_collabs[no_collabs['author_viaf'].isin(coll_av)]
        print(collabs, '\n\n', nocolls, '\n', coll_av)


# for lang in ['eng', 'ger']:
#     c = DataChecks(lang)
#     c.compare_df_values()
#     c.compare_rawdocs_dfs()
#     c.get_collaborations()


# # # Provide the directory path and the string to search for
# directory_path = '/home/annina/scripts/great_unread_nlp/data/text_raw'
# directory_path = '/home/annina/scripts/great_unread_nlp/src/'
# search_string = 'self.text_raw_dir'
# extension = ['.txt', '.py']
# search_string_in_files(directory_path, search_string, extension)


