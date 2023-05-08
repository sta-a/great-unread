import os
from pathlib import Path
import numpy as np
import pandas as pd
from unidecode import unidecode
import re
from collections import Counter


def get_bookname(doc_path):
    return Path(doc_path).stem


def get_doc_paths(docs_dir):
    doc_paths = [os.path.join(docs_dir, doc_name) for doc_name in os.listdir(docs_dir) if Path(doc_name).suffix == '.txt']
    return doc_paths


def preprocess_sentences_helper(text):
    text = text.lower()
    text = unidecode(text)
    text = re.sub('[^a-zA-Z]+', ' ', text).strip()
    text = text.split()
    text = ' '.join(text)
    return text


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
        lines = list(np.load(path)['arr_0'])
    else:
        raise Exception(f'Not a valid line_type {line_type}')
    return lines


def save_list_of_lines(lst, path, line_type):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if line_type == 'str':
        with open(path, 'w') as writer:
            for item in lst:
                writer.write(str(item) + '\n')
    elif line_type == 'np':
        np.savez_compressed(path, np.array(lst))
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


def nr_elements_triangular_mx(mx):
      '''
      Calculate the number of elements in one triangular above the diagonal of a symmetric matrix.
      The diagonal is not counted.
      n(n-1)/2
      '''
      return mx.shape[0]*(mx.shape[0]-1)/2