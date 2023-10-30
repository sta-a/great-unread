# %%
import os
import re
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
from itertools import combinations

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
    # Sorts the file paths in ascending order based on the text length
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
    elif line_type == 'np': # npz?
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
    elif line_type == 'np': # npz?
        np.savez_compressed(path, np.array(lst))
    elif line_type == 'list':
        # list of lists of strings
        with open(path, 'w') as f:
            for clst in lst:
                f.write(','.join(clst) + '\n')
    else:
        raise Exception(f'Not a valid line_type {line_type}')
    

# Only works for passages that are exactly equal, because of hashing
def find_duplicated_lines(path):
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
                    if match_ratio > 0.5:  # Doesn't work, only exact matches are found
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



def check_equal_line_count(dir_eng, dir_ger, extension):
    '''
    Extension: the extension of the files in dir_ger
    '''
    files1 = os.listdir(dir_eng)
    files2 = os.listdir(dir_ger)

    for file_name in files1:
        base_name, ext = os.path.splitext(file_name)
        matching_file = f"{base_name}.{extension}"
        if matching_file in files2:
            tok_path = os.path.join(dir_eng, file_name)
            chunk_path = os.path.join(dir_ger, matching_file)

            line_count1 = sum(1 for _ in open(tok_path))
            line_count2 = sum(1 for _ in open(chunk_path))
            if line_count1==line_count2:
                print(f'{base_name}: Equal nr lines in both dirs.')
                return True
            else:
                print(f'{base_name}: Unequal nr lines in both dirs.')
                print(f'Nr lines {dir_eng}: {line_count1}')
                print(f'Nr lines {dir_ger}: {line_count2}')
                return False


def check_equal_files(dir_eng, dir_ger):
    files1 = set(filename for filename in os.listdir(dir_eng) if os.path.isfile(os.path.join(dir_eng, filename)))
    files2 = set(filename for filename in os.listdir(dir_ger) if os.path.isfile(os.path.join(dir_ger, filename)))

    common_files = files1.intersection(files2)
    unique_files_dir_eng = files1 - common_files
    unique_files_dir_ger = files2 - common_files

    if not unique_files_dir_eng and not unique_files_dir_ger:
        print(f'The directories "{dir_eng}" and "{dir_ger}" contain the same files.')
        return True
    else:
        print('The directories do not contain the same files.')
        print('Files unique to dir_eng:', unique_files_dir_eng)
        print('Files unique to dir_ger:', unique_files_dir_ger)
        return False
    

# General function for finding a string in all code files
def search_string_in_files(directory, search_string, extensions, full_word=False):
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        for line_number, line in enumerate(f, start=1):
                            # if search_string.lower() in line.lower():

                            # Use regular expressions to search for the word
                            if full_word:
                                pattern = r'\b{}\b'.format(re.escape(search_string))
                            else:
                                pattern = re.escape(search_string)
                            
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match:

                                print(f"Found '{search_string}' in '{os.path.basename(file_path)}' (Line {line_number}): \n{line.rstrip()}\n")


class DataHandler():
    '''
    Base class for creating, saving, and loading data.
    '''
    def __init__(self, language=None, output_dir=None, data_type='csv', modes=None, tokens_per_chunk=1000, data_dir='/home/annina/scripts/great_unread_nlp/data', test=False):

        self.test = test
        self.language = language
        self.data_dir = data_dir

        self.output_dir = self.create_output_dir(output_dir)
        self.text_raw_dir = os.path.join(self.data_dir, 'text_raw', language)
        self.doc_paths = get_doc_paths(self.text_raw_dir)
        self.data_type = data_type
        self.modes = modes
        self.tokens_per_chunk = tokens_per_chunk
        self.data_types = ('.npz', '.csv', '.pkl', '.txt', '.svg', '.png')
        self.separator = 'ƒ'
        self.subdir = None
        if self.language == 'eng':
            self.nr_texts = 605
        else:
            self.nr_texts = 547

        # Set the logger's name to the class name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)  # Set the logger's threshold level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Set the handler's threshold level
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if self.test:
            self.doc_paths = get_doc_paths_sorted(self.text_raw_dir)[:3]

    def create_dir(self, dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True) # Create dir and parent dirs if necessary
            self.logger.info(f"Created dir '{dir_path}'.")
        except OSError as e:
            self.logger.info(f"Error creating directory '{dir_path}\n': {e}")

    
    def add_subdir(self, subdir=None):
        if subdir is None:
            subdir = self.__class__.__name__.lower()
            self.logger.info(f'Adding subdir as class name.')
        self.subdir = os.path.join(self.output_dir, subdir)
        if not os.path.exists(self.subdir):
            self.create_dir(self.subdir)


    def create_output_dir(self, output_dir):
        if self.data_dir is None or output_dir is None or self.language is None:
            output_dir = None
        else:
            output_dir = os.path.join(self.data_dir, output_dir, self.language)
            if not os.path.exists(output_dir):
                self.create_dir(output_dir)
        return output_dir


    def create_data(self, **kwargs):
        raise NotImplementedError
    

    def save_data(self, data, file_name=None, **kwargs):
        file_path  = self.get_file_path(file_name, **kwargs)
        dir_path = Path(file_path).parent
        if not os.path.exists(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=False)
        self.save_data_type(data, file_path,**kwargs)
    

    def save_data_type(self, data, file_path, **kwargs):
        data_type = self.get_custom_datatype(file_name_or_path=file_path, **kwargs)

        if data_type == 'csv':
            kwargs_dict = {'index': False, 'na_rep': np.nan, 'sep': ',', 'header': True}
            if 'pandas_kwargs' in kwargs:
                pandas_kwargs = kwargs['pandas_kwargs']
                kwargs_dict.update(pandas_kwargs)

            data.to_csv(file_path, **kwargs_dict)
        elif data_type == 'pkl':
            joblib.dump(data, file_path)
        elif data_type == 'svg':
            data.savefig(file_path, format='svg')
        elif data_type == 'png':
            data.savefig(file_path, format='png')
        elif data_type == 'npz':
            np.savez(file_path, data)
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
                        sep = kwargs.get('txt_sep', self.separator)
                        f.write(f'{sep.join(l)}\n')
                self.logger.info(f'Writing list of lists to file using {self.separator} as the separator.')
        self.logger.debug(f'Saved {data_type} data to {file_path}')


    def file_exists_or_create(self, file_path=None, file_name=None, **kwargs):
        if (file_path is None and file_name is None) or (file_path is not None and file_name is not None):
            raise ValueError("Either 'file_name' or 'file_path' must be provided.")
        if file_path is None:
            file_path = self.get_file_path(file_name, **kwargs)

        if not self.file_exists(file_path=file_path):
            self.logger.debug(f'Creating data for {file_path}.')
            self.create_data(**kwargs)
        # else:
        #     self.logger.debug(f'already exists: {file_path}')

    def load_data(self, load=True, file_name=None, **kwargs):
        self.logger.debug(f'Loading data. If create_data loads data from file, doc_path must be passed with kwargs.')
        file_path = self.get_file_path(file_name=file_name, **kwargs)
        self.file_exists_or_create(file_path=file_path, **kwargs)

        data = None
        if load:
            self.logger.debug(f'Loading {file_path} from file.')
            data = self.load_data_type(file_path, **kwargs)
        return data
    
    def load_data_type(self, file_path, **kwargs):
        if self.data_type == 'csv':
            data = pd.read_csv(file_path, header=0, index_col=0, sep=',')

        elif self.data_type == 'npz':
            data = np.load(file_path)['arr_0'].tolist()

        elif self.data_type == 'pkl':
            data = joblib.load(file_path)

        elif self.data_type == 'txt':
            with open(file_path, 'r') as f:
                # Return a list of strings
                data = f.readlines()
                data = [line.rstrip('\n') for line in data]
                          
        return data

    def create_all_data(self, **kwargs):
        # Check if file exists and create it if necessary
        for mode in self.modes:
            _ = self.load_data(load=False, mode=mode, **kwargs)


    # def create_all_data(self):
    #     startc = time.time()
    #     for i, doc_path in enumerate(self.doc_paths):
    #         _ = self.load_data(load=False, file_name=get_filename_from_path(doc_path), doc_path=doc_path)
    #     print(f'{time.time()-startc}s to tokenize all texts')

    
    def load_all_data(self, **kwargs):
        # Check if file exists, create it if necessary, return all data
        all_data = {}
        for mode in self.modes:
            data  = self.load_data(load=True, mode=mode, **kwargs)
            all_data[mode] = data
        return all_data
    
    def get_custom_datatype(self, file_name_or_path=None, **kwargs):
        if 'data_type' in kwargs:
            data_type = kwargs['data_type']
            if file_name_or_path is not None:
                fdata_type = os.path.splitext(file_name_or_path)[1]
                fdata_type = fdata_type[1:] # Remove dot
                if fdata_type != data_type:
                    raise ValueError(f'File extension ({file_name_or_path}) and data_type passed as kwarg ({data_type}) are not equal.')
        else:
            if file_name_or_path is not None:
                data_type = os.path.splitext(file_name_or_path)[1]
                data_type = data_type[1:] # Remove dot
                if data_type != self.data_type:
                    self.logger.info(f'File extension ({data_type}) and class data type ({self.data_type}) are different. Using file name extension.')
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

        use_kwargs_for_fn = kwargs.get('use_kwargs_for_fn', True)
        if use_kwargs_for_fn == True:
            kwargs_str = '_'.join(f"{str(value)}" for key, value in kwargs.items() if key != 'file_string')
        elif use_kwargs_for_fn == False:
            kwargs_str = ''
        elif use_kwargs_for_fn == 'mode':
            assert 'mode' in kwargs, f'If use_kwargs_for_fn=mode, mode must be passed.'
            kwargs_str = kwargs.get('mode')

        file_name = f"{file_string}{kwargs_str}.{data_type}"
        return file_name
    
    def validate_filename(self, file_name, **kwargs):
        ending_count = file_name.count('.')
        if ending_count == 0:
            data_type = self.get_custom_datatype(**kwargs) # Get data type, don't pass file_name because it has no extension
            file_name = f'{file_name}.{data_type}'
            self.logger.debug(f'Added extension to file name: {file_name}')
        elif ending_count == 1:
            _ = self.get_custom_datatype(file_name_or_path=file_name, **kwargs) # Pass file_name to check it, return value is ignored because file_name already has extension
            if not file_name.endswith(self.data_types):
                raise ValueError(f'Invalid file extension: {file_name}')
        else:
            raise ValueError(f'Multiple file extension: {file_name}')
        return file_name

    
    def get_file_path(self, file_name=None, **kwargs):
        if file_name is None and not kwargs:
            raise ValueError("Either 'file_name' or kwargs must be provided.")
        elif file_name is not None and kwargs:
            self.logger.debug(f'Both file_name and kwargs were passed to get_file_path().')
             # file_name is used, kwargs are ignored. \nfile_name: {file_name}. \nkwargs: {kwargs}')
        if file_name is None:
            file_name = self.create_filename(**kwargs)
        file_name = self.validate_filename(file_name=file_name, **kwargs)

        pathdir = self.output_dir
        if 'subdir' in kwargs:
            if isinstance(kwargs['subdir'], bool):
                if self.subdir is None:
                    if 'mode' in kwargs:
                        mode = kwargs['mode']
                        self.add_subdir(mode)
                        self.logger.debug(f'Set subdir to mode {mode}.')
                    else:
                        self.add_subdir(None) # Add subdir with class name
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

        self.alias_dict = {
            'Hoffmansthal_Hugo': ['Hoffmansthal_Hugo-von'], 
            'Schlaf_Johannes': ['Holz-Schlaf_Arno-Johannes'],
            'Arnim_Bettina': ['Arnim-Arnim_Bettina-Gisela'],
            'Stevenson_Robert-Louis': ['Stevenson-Grift_Robert-Louis-Fanny-van-de', 
                                        'Stevenson-Osbourne_Robert-Louis-Lloyde']}
        self.author_filename_mapping, self.nr_texts_per_author = self.create_data()

    def get_filenames(self):
        filenames = os.listdir(self.output_dir)
        filenames = [os.path.splitext(filename)[0] for filename in filenames]
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
                author_filename_mapping[author] = [file_name]
                

        # Aggregate if author has collaborations with others         
        for author, aliases in self.alias_dict.items():
            if author in authors:
                for alias in aliases:
                    if alias in authors:
                        author_filename_mapping[author].extend(author_filename_mapping[alias]) 
                        del author_filename_mapping[alias]
                        authors = [author for author in authors if author != alias]
        
        nr_texts_per_author = Counter(authors)
        # author_filename_mapping: dict{author name: [list of works by author]}
        # nr_texts_per_author: dict{author name: nr texts by author}
        return author_filename_mapping, nr_texts_per_author


class MetadataChecks(DataHandler):
    '''
    Compare the metadata files with each other to check consistency.
    Compare file names of raw docs with metadata.
    '''

    def __init__(self, language):
        super().__init__(language, output_dir='corpus_corrections', data_type='csv') 
        self.dfs = self.load_dfs()

    def run(self):
        # self.print_cols()
        self.compare_df_values()
        self.compare_rawdocs_dfs()
        # self.check_pub_year() # Can only be run if there are no errors in file names

    def load_dfs(self, df_to_load=None):
        '''
        If df_to_load is not None, load only the specified df
        '''
        if df_to_load is None:
            dirs = ['metadata', 'canonscores', 'sentiscores']
        else:
            dirs = [df_to_load]

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
                
                if 'author_viaf' in df.columns:
                    df['author_viaf'] = self.check_av_col(df['author_viaf'])

                # Find all columns that contain year numbers
                # year_column = self.find_year_column(df)
                # if year_column is not None:
                #     print(f"{filename}: The year column is: {year_column}")

                all_dfs[filename] = df
        return all_dfs
    
    def check_av_col(self, col):
        # Remove whitespace from author_viaf column
        col = col.astype(str).str.replace(r'\s', '', regex=True)
        assert not col.astype(str).str.contains(r'\s', na=False).any(), f'The author_viaf column contains whitespace.'
        # Check if author_viaf col contains anything besides numbers and the '|' character
        # rows_with_condition = col[col.astype(str).str.contains(r'[^0-9|]+')]
        # print(rows_with_condition)
        # assert col.astype(str).str.contains(r'[^0-9|]+').any() == False ################################

        return col
    

    def find_year_column(self, df):
        """
        Find column name if df that contains year numbers.
        """
        for column in df.columns:
            # Check if at least one value in the column matches the year pattern
            if df[column].dropna().astype(str).apply(lambda x: bool(re.match(r'\b\d{4}\b', x))).any():
                return column

        # If no year column is found
        return None

                
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
        for (fn1, df1), (fn2, df2) in combinations(self.dfs.items(), 2):
            dfs_dict = {fn1: df1, fn2: df2}
            self.compare_dfs_pairwise(dfs_dict)

    def compare_dfs_pairwise(self, dfs_dict):

        for fn, df in dfs_dict.items():
            if 'text_id' in df.columns and 'id' in df.columns:
                raise ValueError(f"{fn} contains duplicate column names: {', '.join(df.columns)}")
            if 'text_id' in df.columns:
                df.rename(columns={'text_id': 'id'}, inplace=True)

        for fn, df in dfs_dict.items():
            assert not df.columns.duplicated().any()

        file_names = list(dfs_dict.keys())

        df0_cols = set(dfs_dict[file_names[0]].columns)
        df1_cols = set(dfs_dict[file_names[1]].columns)

        unique_cols = ['id', 'file_name'] # Cols which must have a unique value in every row
        main_cols = unique_cols + ['author_viaf', 'pub_year'] # Cols to check
        # Cols from main cols that are in both dfs
        main_cols = list(df0_cols & df1_cols & set(main_cols))
        # Cols from unique cols that are in both dfs
        unique_cols = list(set(unique_cols) & set(main_cols))
        
        # Check if values that should be unique are duplicated
        # (Sentiscores files can have duplicated values because they have multiple entries for the same text if a review appeared in different magazines)
        if unique_cols:
            if not self.is_sentifile(file_names):
                for key, df in dfs_dict.items():
                    for col in unique_cols:
                        assert df[col].nunique(dropna=True) == df[col].size, f'Column "{col}" not unique in {key}.'

                # Some dfs don't have entries for all texts
                # User 'inner' merge to ignore them
                try:
                    df = pd.merge(dfs_dict[file_names[0]][unique_cols], dfs_dict[file_names[1]][unique_cols], on=unique_cols, how='inner', validate='one_to_one')
                    assert not df.isna().any().any()
                except pd.errors.MergeError as e:
                    print(e)
            else:
                try:
                    df = pd.merge(dfs_dict[file_names[0]][unique_cols], dfs_dict[file_names[1]][unique_cols], on=unique_cols, how='outer')
                    assert not df.isna().any().any()
                except pd.errors.MergeError as e:
                    print(e)

        # Merge dfs to check if rows contain the same values in both dfs
        if main_cols:
            try:
                df = pd.merge(dfs_dict[file_names[0]][main_cols], dfs_dict[file_names[1]][main_cols], on=main_cols, how='outer')
                assert not df.isna().any().any()
            except pd.errors.MergeError as e:
                print(e)


        # Ignore death and birth dates, errors in data
        # # Check year numbers
        # # 'pub_year' is not in same df as the other two
        # date_fn = None
        # year_fn = None
        # if 'date_birth' in df0_cols and 'date_death' in df0_cols:
        #     date_fn = file_names[0]
        # elif 'date_birth' in df1_cols and 'date_death' in df1_cols:
        #     date_fn = file_names[1]
        
        # if 'pub_year' in df0_cols:
        #     year_fn = file_names[0]
        # elif 'pub_year' in df1_cols:
        #     year_fn = file_names[1]

        # print(date_fn, year_fn)
        # if (date_fn is not None) and (year_fn is not None):
        #     print('date_fn', date_fn, 'year_fn', year_fn)
        #     dbdf = dfs_dict[date_fn]
        #     dbdf.loc[dbdf['name'] == 'Goldsmith', 'date_birth'] = 1728
        #     dbdf.loc[dbdf['name'] == 'Goldsmith', 'date_death'] = 1774
        #     self.logger.warning(f"Corrected Goldsmith's birth and death dates.")

        #     pydf = dfs_dict[year_fn]
        #     df = dbdf.merge(pydf, how='outer', on='author_viaf')
        #     assert not df['author_viaf'].isnull().any()

        #     df['date_birth'] = self.process_yearcol(df['date_birth'])
        #     df['date_death'] = self.process_yearcol(df['date_death'])
        #     df['pub_year'] = self.process_yearcol(df['pub_year'])


        #     rows = df[pd.notna(df['date_birth']) & pd.notna(df['date_death'])]
        #     rows['diff'] = rows['date_death'] - rows['date_birth']
        #     print(rows.sort_values(by='diff'))
        #     # assert (rows['date_death'] - rows['date_birth']).ge(20).all()


        #     rows = df[pd.notna(df['date_birth']) & pd.notna(df['pub_year'])]
        #     # assert (rows['pub_year'] - rows['date_birth']).ge(15).all()

        #     # Possible errors in this part
        #     rows = df[pd.notna(df['pub_year']) & pd.notna(df['date_death'])] ############################
        #     # print(rows['pub_year'].dtypes, rows['date_death'].dtypes)
        #     # print((rows['date_death'] - rows['pub_year']).sort_values())
        #     df['pub-death-diff'] = (df['date_death'] - df['pub_year'])
        #     df = df[['file_name', 'author_viaf', 'pub_year', 'date_death', 'pub-death-diff']].sort_values(by='pub-death-diff')
        #     self.save_data(data=df, file_name=f'compare-years-{self.language}.csv')
        #     df.to_csv('pub-death-year', index=False)
        #     # assert (rows['pub_year'] <= rows['date_death']).all()



    def process_yearcol(self, input_col):
        def process_string(input_str):
            if pd.isna(input_str):  # Use pd.isna() to check for NaN
                return np.nan
            
            parts = [int(part) for part in re.split('[|?]', str(input_str)) if part.isdigit()]

            if not parts:  # Check if parts is an empty list
                print(f'parts is not Nan and not a digit')
            elif len(parts) == 2:
                return min(parts)
            elif '?' in str(input_str):
                return parts[0]
            else:
                return parts[0]

        if input_col.dtype in [np.int64, np.float64]:
            return input_col
        else:
            return input_col.apply(process_string)




    def compare_rawdocs_dfs(self):
        '''
        Check if file names of raw texts and file names in 'file_name' column of dfs are the same
        '''
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

        fn_mapping = fn_mapping.loc[fn_mapping['edit-dist'] <= dist_cutoff].loc[:, ['metadata-fn', 'rawdocs-fn', 'file']]
        fn_mapping = fn_mapping.sort_values(by='file')
        self.save_data(data=fn_mapping, file_name=f'compare_filenames_{self.language}.csv')

        # Check if all columns contain only unique values
        fn_mapping = fn_mapping[['metadata-fn', 'rawdocs-fn']]
        fn_mapping = fn_mapping.drop_duplicates()
        assert (fn_mapping.nunique() == fn_mapping.shape[0]).all()
        return fn_mapping


    def check_pub_year(self):
        for fn, df in self.dfs.items():
            if 'pub_year' in df.columns:
                print(f'Checking year for {fn}.')

                fn_mapping = self.compare_rawdocs_dfs()
                fn_mapping = dict(zip(fn_mapping['metadata-fn'], fn_mapping['rawdocs-fn']))

                df.loc[:, 'file_name'] = df['file_name'].replace(to_replace=fn_mapping)

                df['fn_year'] = df['file_name'].str[-4:].astype(int)

                rows_not_meeting_condition = df[df['fn_year'] != df['pub_year']]
                print(f"Rows where 'fn-year' and 'pub_year' don't contain the same value in {fn}:")
                print(rows_not_meeting_condition)

                assert (df['fn_year']  == df['pub_year']).all()



class DataLoader(DataHandler):
    '''
    Load various data frames to be used in other classes.
    '''
    def __init__(self, language):
        super().__init__(language, output_dir=None, data_type='csv') 


    def prepare_canon_df(self, fn_mapping):
        # Load data
        file_name = '210907_regression_predict_02_setp3_FINAL.csv'
        path = os.path.join(self.data_dir, 'canonscores', file_name)
        df = pd.read_csv(path, sep=';')

        df = df.loc[:, ['file_name', 'm3']]
        df.loc[:, 'file_name'] = df['file_name'].replace(to_replace=fn_mapping)
        df = df.rename(columns={'m3': 'canon'})
        df = df.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
        df = df.drop_duplicates(subset='file_name')

        # Select rows from current language (df contains data for both eng and ger)
        # Get a list of filenames without the '.txt' extension
        rdfn = [f.rstrip('.txt') for f in os.listdir(self.text_raw_dir) if f.endswith(".txt")]
        # Select rows from the DataFrame where 'file_name' is in the list of filenames
        df = df[df['file_name'].isin(rdfn)]
        # Check if all filenames in the directory are in the DataFrame
        assert all(file_name in df['file_name'].values for file_name in rdfn)
        assert len(df) == self.nr_texts
        return df


    # def set_gender_collaborations(self, df, authors):
    #     assert not df['author_viaf'].isna().any()

    #     # Get author_viaf of authors that are involved in collaborations
    #     coll_avs = df.loc[df['author_viaf'].astype(str).str.contains(r'\|', na=False, regex=True), 'author_viaf'].tolist()
    #     print(coll_avs)
    #     # Set gender for collaboration
    #     for coll_av in coll_avs:
    #         genders = []
    #         av1, av2 = coll_av.split('|')
    #         print(av1, av2)
    #         print(authors.loc[authors['author_viaf']==av1, 'gender'])
    #         genders.append(authors.loc[authors['author_viaf']==av1, 'gender'])
    #         genders.append(authors.loc[authors['author_viaf']==av2, 'gender'])

    #         print('genders', genders)

    #         if all(value == 'm' for value in genders):
    #             g = 'm'
    #         elif all(value == 'f' for value in genders):
    #             g = 'f'
    #         else:
    #             g = 'b'

    #         df.loc[df['author_viaf'] == coll_av, 'gender'] = g

    #     return df




    def prepare_metadata(self, type):
        mc = MetadataChecks(self.language) 
        fn_mapping = mc.compare_rawdocs_dfs()
        fn_mapping = dict(zip(fn_mapping['metadata-fn'], fn_mapping['rawdocs-fn']))

        # There are inconsistencies between the file names of the raw docs and the file names in older data frames.
        # Correct the old file names 
        if type == 'canon':
            df = self.prepare_canon_df(fn_mapping)

        
        elif type == 'meta' or type == 'gender':
            self.output_dir = self.create_output_dir('metadata')
            self.logger.info(f'Created output dir from inside function.')

            file_name = f'{self.language.upper()}_texts_meta.csv'
            df = pd.read_csv(os.path.join(self.output_dir, file_name), sep=';')
            # Remove whitespace from author_viaf column
            df['author_viaf'] = mc.check_av_col(df['author_viaf'])
            df.loc[:, 'file_name'] = df['file_name'].replace(to_replace=fn_mapping)


            if type == 'gender':
                file_name = f'authors_{self.language}.csv'
                authors = pd.read_csv(os.path.join(self.output_dir, file_name), sep=';')

                authors = authors.loc[:, ['author_viaf', 'gender']]
    
                df = df.merge(authors, how='left', on='author_viaf', validate='many_to_one')
                assert len(df) == self.nr_texts

                # Remove whitespace from author_viaf column
                df['author_viaf'] = mc.check_av_col(df['author_viaf'])

                df = df.loc[:, ['file_name', 'gender', 'author_viaf']]
                df.loc[:, 'gender'] = df['gender'].replace(to_replace={'w':'f'})

                # Set anonymous values to 'a'
                mask = df['file_name'].str.contains('anonymous_anonymous', case=False)
                # Set the value 'Female' in the 'Gender' column for the matched rows
                df.loc[mask, 'gender'] = 'a' # a for anonymous


                # Find gender of collaborating authors (easier to do by hand)
                # self.set_gender_collaborations(df, authors)

                # collaboration_rows = df[df['author_viaf'].astype(str).str.contains(r'\|', na=False, regex=True)]
                # for fn in collaboration_rows['file_name'].to_list():
                #     print(fn)


                if self.language == 'ger':
                    df.loc[df['file_name'] == 'Hebel_Johann-Peter_Kannitverstan_1808', 'gender'] = 'm'
                    df.loc[df['file_name'] == 'May_Karl_Ardistan-und-Dschinnistan_1909', 'gender'] = 'm'
                    df.loc[df['file_name'] == 'May_Karl_Das-Waldroeschen_1883', 'gender'] = 'm'
                    df.loc[df['file_name'] == 'Arnim-Arnim_Bettina-Gisela_Rattenzuhausbeiuns_1844', 'gender'] = 'f'
                    df.loc[df['file_name'] == 'Holz-Schlaf_Arno-Johannes_Papa-Hamlet_1889', 'gender'] = 'm'


                # Edith Somerville: f
                # Martin Ross: f (pseudonym of Violet Florence Martin)
                # Robert Louis Stevenson: m
                # Frances "Fanny" Matilda Van de Grift Osbourne Stevenson: f
                # Samuel Lloyd Osbourne: m
                if self.language == 'eng':
                    df.loc[df['file_name'] == 'Somerville-Ross_Edith-Martin_An-Irish-Cousin_1889', 'gender'] = 'f'
                    df.loc[df['file_name'] == 'Somerville-Ross_Edith-Martin_Experience-of-an-Irish-RM_1899', 'gender'] = 'f'
                    df.loc[df['file_name'] == 'Somerville-Ross_Edith-Martin_The-Real-Charlotte_1894', 'gender'] = 'f'
                    df.loc[df['file_name'] == 'Stevenson-Grift_Robert-Louis-Fanny-van-de_The-Dynamiter_1885', 'gender'] = 'b' # Both male and female author
                    df.loc[df['file_name'] == 'Stevenson-Osbourne_Robert-Louis-Lloyde_The-Ebb-Tide_1894', 'gender'] = 'm'
                

                # print(df['gender'].value_counts())
                assert df['gender'].isin(['f', 'm', 'a', 'b']).all()
                assert not df.isna().any().any()
                
        df = df.set_index('file_name', inplace=False)
        assert len(df) == self.nr_texts
        self.logger.info(f'Returning df with file_name as index.')
        return df


    def prepare_features(self):
        '''
        Return features table for full-book features
        '''
        self.output_dir = self.create_output_dir('features')
        self.logger.info(f'Created output dir from inside function.')

        path = os.path.join(self.output_dir, 'book.csv')
        df = pd.read_csv(path, index_col='file_name')

        strings_to_drop = ['average_sbert_embedding_', 'd2v_embedding_', 'pos_bigram_', 'pos_trigram_']
        columns_to_drop = [col for col in df.columns if any(substring in col for substring in strings_to_drop)]
        df = df.drop(columns=columns_to_drop, inplace=False)
        self.logger.info(f'Returning df with file_name as index.')
        return df
    




# mh = MetadataChecks('eng').run()
# dl = DataLoader('ger').prepare_metadata('gender')



# # Provide the directory path and the string to search for
# directory_path = '/home/annina/scripts/great_unread_nlp/data/text_tokenized'
# directory_path = '/home/annina/scripts/great_unread_nlp/src/'
# search_string = "medoid"
# extension = ['.txt', '.py']
# search_string_in_files(directory_path, search_string, extension, full_word=False)


# %%
