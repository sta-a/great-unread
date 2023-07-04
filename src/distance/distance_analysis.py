import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def check_symmetric(mx):
    print('Checking if matrix is symmetric. Index and column names must be identical.')
    return mx.equals(mx.T)


def nr_elements_triangular(n_or_mx):
    '''
    Calculate the number of elements in one triangular above the diagonal of a symmetric matrix.
    The diagonal is not counted.
    n(n-1)/2
    '''
    if isinstance(n_or_mx, pd.DataFrame):
        n = n_or_mx.shape[0]
    else:
        n = n_or_mx
    return n*(n-1)/2

def set_diagonal(df, value):
        '''
        df: distance or similarity matrix
        value: the value that should be on the diagonal
        '''
        #df.values[[np.arange(df.shape[0])]*2] = value
        for i in range(0, df.shape[0]):
              df.iloc[i, i] = value
        return df

def distance_to_similarity_mx(df):
    '''
    df: distance matrix
    Invert distances to obtain similarities.
    '''
    # Assert that there are no Nan
    assert not df.isnull().values.any()
    # Set diagonal to Nan
    df = set_diagonal(df, np.nan)
    # Assert that there are no zero distances
    assert df.all().all()

    df = df.rdiv(1) # divide 1 by the values in the matrix
    return df

def get_mx_triangular(mx):
    '''
    mx: symmetric matrix
    Return values in one triangular of the matrix as array. Diagonal is ignored
    '''
    vals = np.tril(mx.values, k=-1).flatten() # Return a copy of an array with elements above the k-th diagonal zeroed
    vals = vals[np.nonzero(vals)] # Remove zeros from tril
    # Check number of elements below the diagonal
    assert len(vals) == nr_elements_triangular(mx)
    return vals

def plot_distance_distribution(mx, mx_type, language, filename, data_dir):
    start = time.time()
    vals = get_mx_triangular(mx) # Get triangular matrix, ignore diagonal
    assert not np.any(np.isnan(vals)) # Test whether any element is nan

    # Find most common frequency
    _, counts = np.unique(vals, return_counts=True)
    ind = np.argmax(counts)

    print(f'Minimum {mx_type}: {min(vals)}. Maximum {mx_type}: {max(vals)}. Most common {mx_type}: {vals[ind]}.')

    fig = plt.figure(figsize=(20,6), dpi=300)
    ax = fig.add_subplot(111)
    if mx_type == 'distance':
        binsize = 0.001
        xtick_step = 1
    else:
        binsize = 0.1
        xtick_step = 5
    ax.hist(vals, bins = np.arange(0,max(vals) + 0.1, binsize), log=True, ec='black', color='black') #kwargs set edge color
    ax.set_xlabel(f'{mx_type.capitalize()}')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{mx_type.capitalize()} distribution {filename}')
    plt.xticks(np.arange(0, max(vals) + 0.1, step=xtick_step))
    plt.xticks(rotation=90)
    ax.grid(False)

    print(os.path.join(data_dir, 'distances', language, f'distribution_{mx_type}_{filename}.png'))
    plt.savefig(os.path.join(data_dir, 'distances', language, f'distribution_{mx_type}_{filename}.png'), format="png")
    plt.show()
    print(f'{time.time()-start}s to create {mx_type} distribution plot.')
