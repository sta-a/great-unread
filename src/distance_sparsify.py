import pandas as pd
import numpy as np
from utils import get_texts_by_author


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


def filter_min_author_similarity(mx):
      '''
      mx: similarity marix
      Edge only if texts are equally or more similar than the least similar text by the same author
      This results in a directed weight matrix
      '''
      directed_mx = []
      author_filename_mapping, _ = get_texts_by_author(list_of_filenames=mx.index)
      for _, list_of_filenames in author_filename_mapping.items():
            author_mx = mx.loc[list_of_filenames, list_of_filenames]
            new_rows = mx.loc[list_of_filenames]
            if author_mx.shape[0] != 1:
                  min_simliarity = np.nanmin(author_mx.to_numpy())
                  # set all distances above max to nan
                  new_rows[new_rows < min_simliarity] = np.nan
            else:
                  new_rows[:] = np.nan
            directed_mx.append(new_rows)

      directed_mx = pd.concat(directed_mx)
      assert directed_mx.shape == mx.shape

      print(f'Number of possible edges: {directed_mx.shape[0]*(directed_mx.shape[0]-1)}')
      print(f'Number of non-nan entries: {np.sum(directed_mx.count())}.')

      return directed_mx
