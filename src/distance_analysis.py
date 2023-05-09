import pandas as pd
import numpy as np

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