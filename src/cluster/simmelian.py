# %%
import os
import matplotlib.pyplot as plt

import time
import numpy as np

from copy import deepcopy
import pandas as pd
import numpy as np

import sys
sys.path.append("..")
from helpers import get_simmx_mini, create_simmx
from .network import NXNetwork
from .create import SimMx


class Simmelian():
    '''
    Calculate parametric version of Simmelian backbones.
    Check results with Visone.
    Visone's 'identified' and 'conditioned' options are implemented.
    '''

    def __init__(self, mx, conditioned=True):
        self.mx = mx # Similarity matrix
        self.conditioned = conditioned # If conditioned, the overlap calculation is only performed for those links which have been top-ranked themselves.
        self.set_diagonal(0) # no self-loops


    def set_diagonal(self, value):
        for i in range(0, self.mx.shape[0]):
            self.mx.iloc[i, i] = value


    def sort_all_neighbours(self, row):
        # Sort each row in descending order and return a list of tuples (neighbor name, neighbor distance)
        sorted_values = row.sort_values(ascending=False)
        sorted_tuples = [(index, value) for index, value in zip(sorted_values.index.tolist(), sorted_values.tolist())]
        return sorted_tuples


    def get_neighbours(self):
        # Create df where the index is the nodes and each row represents its sorted neighbours
        # The leftmost column is the closest neighbour
        df = self.mx.apply(self.sort_all_neighbours, axis=1, result_type='expand')
        return df
    
    
    def compare_cols(self, col, col_left):
        # Compare two columns and replace tuples with np.nan
        # Values have to be equal to the last column of the topk neighbourhood to be included (equal with no tolerance)
        col_vals = col.apply(lambda x: x[1]).to_numpy()
        col_left_vals = col_left.apply(lambda x: x[1]).to_numpy()
        mask = col_left_vals == col_vals
        col[mask == False] = np.nan
        return col

    
    def get_topk(self, df, k):
        topk = df.iloc[:,:k]

        # Include more neighbours if they have the same distance as the values in the last column of the selected df
        other = df.iloc[:,k:]

        new_df = {}
        for col_name in other.columns:
            # Access the column using df[col_name]
            current_col = other[col_name]
            current_col = self.compare_cols(current_col, topk.iloc[:,-1])
            if current_col.isna().all():
                break
            else:
                new_df[col_name] = current_col

        # Evaluate neighbours that were added from other because they have the same similarity as the kth neighbour
        # If k is high, the number of added cols can be high
        # for col_name, col_values in new_df.items():
        #     non_nan_values = [val for val in col_values if not pd.isna(val)]
        #     print(f"Nr. Non-NaN values in '{col_name}': {len(non_nan_values)}")

        if bool(new_df):
            new_df = pd.DataFrame(new_df)
            assert not new_df.isna().all().all()
            # print('new_df\n', new_df.shape, new_df)

            topk = pd.concat([topk, new_df], axis=1)

        # Return only names of neighbours and not values
        topk = topk.applymap(lambda x: x[0] if isinstance(x, tuple) else x)
        return topk
    

    def get_intersection(self, row1, row_const):
        if row1.equals(row_const):
            return 0
        else:
            # Check for reciprocity ('identified' = True in Visone)
            # If the rows that are being compared are are among each others top-k neighbours, intersection is increased by 1
            reciprocity = 0
            if (row1.name in row_const.values) and (row_const.name in row1.values):
                reciprocity = 1

            intersection = set(row1).intersection(set(row_const))
            if np.nan in intersection:
                intersection.remove(np.nan)

            intersection = len(intersection) + reciprocity
            return intersection


    def get_dist_to_row(self, row, df):
        df = df.apply(self.get_intersection, axis=1, row_const=row)
        return df


    def get_overlap(self, df):
        df = df.apply(self.get_dist_to_row, axis=1, df=df)
        return df
    
    def set_diagonal_return(self, value, df):
        for i in range(0, self.mx.shape[0]):
            df.iloc[i, i] = value

        return df
    
    def filter_simmx(self, df, min_overlap):
        assert df.shape == self.mx.shape
        assert df.index.is_monotonic_increasing, "DataFrame index is not alphabetically sorted."
        assert self.mx.columns.equals(self.mx.index), "Column names do not match the index in the DataFrame 'mx'"
        assert df.index.equals(self.mx.index), "Index mismatch between df and self.mx"
        assert df.columns.equals(self.mx.columns), "Column names mismatch between df and self.mx"
        assert not df.isna().any().any()
        assert not self.mx.isna().any().any()

        mx = deepcopy(self.mx).astype(float)
        #df = df.astype(float)
        
        # Set values below min_overlap to 0 and others to 1
        df[df < min_overlap] = 0.0
        df[df >= min_overlap] = 1.0

        # df = self.set_diagonal_return(0.0, df)


        # Set values in similarity matrix to 0 where filtered overlap matrix is below min_overlap

        # Set values in self_mx to zero where df is 0
        filtered_simmx = np.where(df == 0, 0, mx)

        # Convert the result back to DataFrame
        filtered_simmx = pd.DataFrame(filtered_simmx, columns=df.columns, index=df.index)
        # filtered_simmx = self.set_diagonal_return(0.0, filtered_simmx)
        #filtered_simmx = filtered_simmx.astype(float)
        

        check = False
        if check:
            # filtered_simmx can be 0 where df is not 0, because 0 is at the corresponding position in mx
            # 0 in argamon_quadratic-500 at positions Kipling_Rudyard_How-the-Camel-Got-His-Hump_1902, Kipling_Rudyard_The-Sing-Song-of-the-Old-Man-Kangaroo_1902
            print('mx', self.mx.loc['Kipling_Rudyard_How-the-Camel-Got-His-Hump_1902', 'Kipling_Rudyard_The-Sing-Song-of-the-Old-Man-Kangaroo_1902'])
            print('df', df.loc['Kipling_Rudyard_How-the-Camel-Got-His-Hump_1902', 'Kipling_Rudyard_The-Sing-Song-of-the-Old-Man-Kangaroo_1902'])
            print('filtered_simmx', filtered_simmx.loc['Kipling_Rudyard_How-the-Camel-Got-His-Hump_1902', 'Kipling_Rudyard_The-Sing-Song-of-the-Old-Man-Kangaroo_1902'])

            atol = 0.001
            comparison_result = np.isclose(df, 0, atol=atol) == np.isclose(filtered_simmx, 0, atol=atol)

            # Find positions where the comparison is false
            false_positions = np.where(~comparison_result)


            print("\nPositions where the comparison is false:")
            print(false_positions)

            fdf = pd.DataFrame(comparison_result, index=df.index, columns=df.columns)
            print(fdf.iloc[335, 366])
            print(df.iloc[335, 366])
            print(filtered_simmx.iloc[335, 366])
            
            print(df.loc[df.index[335], df.columns[366]])
            print(df.loc[df.index[366], df.columns[335]])

            print(df.index[335], df.columns[366])
            print(df.index[366], df.columns[335])

            print(filtered_simmx.loc[filtered_simmx.index[335], filtered_simmx.columns[366]])
            print(filtered_simmx.loc[filtered_simmx.index[366], filtered_simmx.columns[335]])

            print(filtered_simmx.index[335], filtered_simmx.columns[366])
            print(filtered_simmx.index[366], filtered_simmx.columns[335])
            print('Nr 0 in df: ', np.sum(df.values == 0))
            print('Nr 0 in filtered_simmx: ', np.sum(filtered_simmx.values == 0))



        print('0 at the same positions:', np.all(filtered_simmx[df == 0] == 0))
        return filtered_simmx
    
    
    def filter_conditioned(self, df, topk):
        df_original = deepcopy(df)

        for row_index, row_values in df.iterrows():
            neighbours = topk.loc[row_index].tolist()

            for col_index, value in row_values.items():
                if value != 0:
                    if col_index not in neighbours:
                        df_original.loc[row_index, col_index] = 0
        return df_original


    def run_parametric(self, min_overlap, k):
        prints = False # print intermediate results
        s = time.time()
        # k is called max ranking in Visone

        df = self.get_neighbours()
        if prints:
            print(df.iloc[:5, :], '\n\n')

        topk = self.get_topk(df, k)
        if prints:
            print(topk.iloc[:5, :].to_markdown(), '\n\n')

        df = self.get_overlap(topk)
        if prints:
            print(df.iloc[:5, :].to_markdown(), '\n\n')

        df = self.filter_simmx(df, min_overlap)

        if self.conditioned:
            df = self.filter_conditioned(df, topk)
            if prints:
                print(df.iloc[:5, :].to_markdown(), '\n\n')
        
        print(f'{time.time()-s}s to calculate Simmelian backbone.')
        return df



def check_visone():
    dirpath = '/home/annina/scripts/great_unread_nlp/data/similarity/eng/visone-backbones/'
    minimx = False
    conditioned = True
    if minimx:
        params = {'minimx': {'min_overlap': 5, 'k': 10}}
    else:
        params = {'burrows-500': {'min_overlap': 5, 'k': 10}}


    for mxname, vals in params.items():
        min_overlap = vals['min_overlap']
        k = vals['k']

        if minimx:
            simmx = create_simmx(30)
            # Save matrix as input for Visone
        else:
            simmx = pd.read_csv(os.path.join(dirpath, 'burrows-500.csv'), index_col='file_name')
            for i in range(len(simmx.index)):
                simmx.iloc[i, i] = 0


        # Visone: conditioned=True, identified=True
        if conditioned:
            vname = f'vi-{mxname}-{min_overlap}-{k}_conditioned.csv'
        else:
            vname = f'vi-{mxname}-{min_overlap}-{k}_unconditioned.csv'

        vmx = pd.read_csv(os.path.join(dirpath, vname), index_col=0)


        # Calculate backbones with this class
        s = Simmelian(simmx, conditioned=conditioned)
        pmx = s.run_parametric(min_overlap=min_overlap, k=k)
        pmx = pmx.reindex(index=vmx.index, columns=vmx.columns)
        
        tol = 1e-5
        print(f'All values in the python and visone matrix are close within tolerance {tol}: ', np.allclose(pmx, vmx, atol=tol))

        equal_positions = (pmx == vmx).stack()
        i = 0
        # Print the positions and values where they are not equal
        for position, value in equal_positions.items():
            if value == False:
                row, col = position
                if i < 10:
                    print(f"At position ({row}, {col}), values are not equal: {pmx.at[row, col]} != {vmx.at[row, col]}")
                    i += 1

# check_visone()
