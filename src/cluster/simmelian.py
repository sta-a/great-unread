# %%
import networkit as nk
import matplotlib.pyplot as plt
from cluster.network import NXNetwork
from cluster.create import SimMx

import networkx as nx
import time
import numpy as np

from copy import deepcopy
import pandas as pd
import numpy as np
import networkx as nx
from helpers import get_simmx_mini, create_simmx
import os


class Simmelian():
    '''
    Calculate parametric version of Simmelian backbones.
    Check results with Visone.
    Visone's 'identified' and 'conditioned' options are implemented.
    '''

    def __init__(self, mx, conditioned=True):
        self.mx = mx # Similarity matrix
        self.conditioned = conditioned # If conditioned, the overlap calculation is only performed for those links which have been top-ranked themselves.


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
    
    
    def compare_cols(self, col, col_left, tolerance=0.001):
        # Compare two columns and replace tuples with np.nan if the difference is outside the tolerance.
        col_vals = col.apply(lambda x: x[1]).to_numpy()
        col_left_vals = col_left.apply(lambda x: x[1]).to_numpy()
        mask = np.abs(col_left_vals - col_vals) < tolerance
        col[mask == False] = np.nan
        return col
    
    
    def get_topk(self, df, k):
        topk = df.iloc[:,:k]

        # Include more neighbours if they have the same distance (within tolerance) as the values in the last column of the selected df
        other = df.iloc[:,k:]

        new_df = pd.DataFrame() 
        for col_name in other.columns:
            # Access the column using df[col_name]
            current_col = other[col_name]
            current_col = self.compare_cols(current_col, topk.iloc[:,-1])
            if current_col.isna().all():
                break
            else:
                new_df[col_name] = current_col

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
    
    
    def filter_simmx(self, df, min_overlap):
        # Set values in similarity matrix to 0 where filtered overlap matrix is 0
        if df.shape != self.mx.shape:
            raise ValueError("Input DataFrame shape does not match the shape of self.mx")
        
        # Set values below min_overlap to 0 and others to 1
        df[df < min_overlap] = 0
        df[df >= min_overlap] = 1

        # Set values in similarity matrix to 0 where filtered overlap matrix is below min_overlap
        filtered_simmx = self.mx * df

        assert (df == 0).equals(filtered_simmx == 0), 'Dfs have different positions with 0 values.'
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
        topk.to_csv('/home/annina/scripts/great_unread_nlp/data/similarity/eng/visone-backbones/topk.csv', index=True)
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
    minimx = True
    if minimx:
        params = {'minimx': {'min_overlap': 5, 'k': 10}}
    else:
        params = {'burrows-500': {'min_overlap': 5, 'k': 10}}


    for mxname, vals in params.items():
        min_overlap = vals['min_overlap']
        k = vals['k']
        printn = 5

        if minimx:
            simmx = create_simmx(30)
            # Save matrix as input for Visone
            simmx.to_csv(os.path.join(dirpath, 'minimx.csv'), index=True)
        else:
            simmx = pd.read_csv('/home/annina/scripts/great_unread_nlp/data/similarity/eng/simmxs/burrows-500.csv', index_col='file_name')
            for i in range(len(simmx.index)):
                simmx.iloc[i, i] = 0
            # print(simmx.iloc[:printn, :printn].to_markdown())
            # simmx.to_csv(os.path.join(dirpath, f'{mxname}.csv'), index=True)


        # Visone: conditioned=True, identified=True
        vname = f'vi-{mxname}-{min_overlap}-{k}_unconditioned.csv' ########################
        vmx = pd.read_csv(os.path.join(dirpath, vname), index_col=0)
        vmx = vmx.sort_index(axis=0).sort_index(axis=1)
        vmx.to_csv(os.path.join(dirpath, vname), index=True)
        vmx = vmx.replace({0: 0.0, 0.000: 0.0})
        print('vmx symmetric: ', vmx.equals(vmx.T))
        print(vmx.iloc[:printn, :printn])


        # Calculate backbones with this class
        s = Simmelian(simmx, conditioned=False) #############################
        pmx = s.run_parametric(min_overlap=min_overlap, k=k)
        pmx = pmx.replace({0: 0.0, 0.000: 0.0})
        pmx = pmx.reindex(index=vmx.index, columns=vmx.columns)
        print('\n---------------\npmx symmetric: ', pmx.equals(pmx.T))
        print(pmx.iloc[:printn, :printn], '\n\n---------------------\n\n')
        pmx.to_csv(os.path.join(dirpath, 'pmx.csv'), index=True)
        

        print('All values in the python and visone matrix are close within tolerance: ', np.allclose(pmx, vmx, atol=1e-2))

        equal_positions = (pmx == vmx).stack()
        i = 0
        # Print the positions and values where they are not equal
        for position, value in equal_positions.items():
            if value == False:
                row, col = position
                if i < 10:
                    print(f"At position ({row}, {col}), values are not equal: {pmx.at[row, col]} != {vmx.at[row, col]}")
                    i += 1

    # With burrows-500, python implementation kepps more edges than visone implelmentation 
    # Different values probably stem from handling neighbours with equal distances

# check_visone()

