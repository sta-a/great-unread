# %load_ext autoreload
# %autoreload 2
import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
import logging
logging.basicConfig(level=logging.DEBUG)


class DataHandler():
    '''
    Base class for creating, saving, and loading data.
    '''
    def __init__(self, language, output_dir, data_type='csv', tokens_per_chunk=500, data_dir='/home/annina/scripts/great_unread_nlp/data'):
        self.language = language
        self.output_dir = output_dir # Name of subdir of data_dir where results are stored
        self.data_dir = data_dir
        self.data_type = data_type
        self.tokens_per_chunk = tokens_per_chunk
        self.logger = logging.getLogger(__name__)

    def create_all_data(self):
        # Implement the distance calculation logic
        raise NotImplementedError

    def create_filename(self, kwargs):
        # Implement the logic to create the file name
        # return os.path.join(f"{kwargs['']}", self.data_type)
        raise NotImplementedError

    def save_data(self, data, file_name, **kwargs):
        file_path  = self.get_file_path(file_name, kwargs)
        dir_path = Path(file_path).parent
        if not os.path.exists(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=False)
        self.save_data_type(data, file_path, kwargs)

    def save_data_type(self, data, file_path, kwargs=None):
        if self.data_type == 'csv':
            data = data.sort_index(axis=0).sort_index(axis=1)
            data.to_csv(file_path, header=True, index=True)

    def load_data(self, file_name=None, **kwargs):
        if file_name is not None and not file_name.endswith(['.npz', '.csv']):
            raise ValueError("The file extensions must be provided. Supported extensions are '.csv' and '.npz'.")

        file_path = self.get_file_path(file_name, kwargs)
        if not os.path.exists(file_path):
            self.logger.info(f'Creating d2v embeddings for all parameters.')
            self.create_all_data()
        
        data = self.load_data_type(file_path, kwargs)
        if file_name is None:
            file_name = self.create_filename(kwargs)
        self.logger.info(f'Loaded {file_name} from file.')
        return data
    
    def load_data_type(self, file_path, kwargs):
        if self.data_type == 'csv':
            data = pd.read_csv(file_path, header=0, index_col=0)

        elif self.data_type == 'np':
            data = load_list_of_lines(file_path, 'np')
        return data

    def get_file_path(self, file_name=None, kwargs=None):
        if file_name is None and not kwargs:
            raise ValueError("Either 'file_name' or 'kwargs' must be provided.")
        if file_name is not None and kwargs:
            raise ValueError("Only one of 'file_name' or 'kwargs' should be provided.")

        if file_name is None:
            file_name = self.create_filename(kwargs)
        file_path = os.path.join(self.data_dir, self.output_dir, self.language, file_name)
        return file_path
    
    def file_exists(self, file_name=None, **kwargs):
        return os.path.exists(self.get_file_path(file_name, kwargs))
#-----------------------------------------------------------------------------------------------------

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
    doc_paths = [os.path.join(docs_dir, doc_name) for doc_name in os.listdir(docs_dir) if Path(doc_name).suffix == '.txt']
    return doc_paths


def df_from_dict(d, keys_as_index, keys_column_name, values_column_value):
    '''Turn both keys and values of a dict into columns of a df.'''
    df = pd.DataFrame(d.items(), columns=[keys_column_name, values_column_value])
    if keys_as_index == True:
        df = df.set_index(keys=keys_column_name)
    return df
    

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


def get_texts_by_author(list_of_filenames):
    '''
    Map the filenames to the authors and count the number of texts per author.
    '''
    
    authors = []
    author_filename_mapping = {}
    #Get texts per authors
    for file_name in list_of_filenames:
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
# %%
