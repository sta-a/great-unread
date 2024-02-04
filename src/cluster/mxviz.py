
import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
import pickle
from copy import deepcopy
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.gridspec as gridspec
import textwrap
import time
import random
random.seed(9)

from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform

import sys
sys.path.append("..")
from utils import DataHandler
from .create import SimMx
from .cluster_utils import VizBase, Colors
import logging
logging.basicConfig(level=logging.DEBUG)


class MxReorder():
    '''Sort row and column indices so that clusters are visible in heatmap.'''

    ORDERS = ['fn', 'olo']

    def __init__(self, language, mx, info):
        self.language = language
        self.mx = mx
        self.info = info
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)


    def order(self):
        order_methods = {
            'fn': self.order_fn,
            'olo': self.order_olo,
            'continuous': self.order_cont,
        }

        if self.info.order in order_methods:
            ordmx = order_methods[self.info.order]()
            if isinstance(ordmx, SimMx):
                ordmx = ordmx.mx
        else:
            raise ValueError(f"Invalid order value: {self.info.order}")

        assert self.mx.mx.shape == ordmx.shape
        assert self.mx.mx.equals(self.mx.mx.T)
        assert ordmx.index.equals(ordmx.columns), 'Index and columns of ordmx must be equal.'

        return ordmx

    
    def order_cont(self):
        df = self.info.metadf.copy(deep=True)
        file_names = df.sort_values(by=self.info.attr).index.tolist()
        return self.mx.mx.loc[file_names, file_names]
    

    def order_fn(self):
        # Sort rows and columns of each cluster (respectively attribute value) according to file name, which starts with the name of the author
        ordmxs = []

        # Get index labels belonging to the current cluster
        for cluster in self.info.metadf[self.info.attr].unique():
            file_names = self.info.metadf[self.info.metadf[self.info.attr] == cluster].index.tolist()

            df = self.mx.mx.loc[:, file_names].sort_index(axis=1)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=1)

        ordmxs = []
        for cluster in self.info.metadf[self.info.attr].unique():
            file_names = self.info.metadf[self.info.metadf[self.info.attr] == cluster].index.tolist()
            df = ordmx.loc[file_names, :].sort_index(axis=0)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=0)
        return ordmx

    
    def order_olo(self):
        ordered_fns = []
        # Get unique cluster lables, sorted from rarest to most common label
        unique_clust = self.info.metadf[self.info.attr].value_counts().sort_values().index.tolist()

        # Iterate over unique attribute values
        for cluster in unique_clust:

            # Extract file names for the current cluster
            file_names = self.info.metadf[self.info.metadf[self.info.attr] == cluster].index.tolist()

            # If 1 or 2 elements in cluster, order doesn't matter
            if len(file_names) <=2:
                ordered_fns.extend(file_names)

            # Get OLO for current cluster
            else:
                # Subset the similarity matrix for the current cluster
                cmx = self.mx.dmx.loc[file_names, file_names]
                sq_cmx = squareform(cmx)
                
                cluster_linkage = linkage(sq_cmx, method='average')
                
                order = leaves_list(optimal_leaf_ordering(cluster_linkage, sq_cmx))
                # Map integer indices to string indices
                ordered_fn = cmx.index[order].tolist()
                ordered_fns.extend(ordered_fn)

        # Check that there are no duplicated values
        assert len(set(ordered_fns)) == len(ordered_fns)
        ordmx = self.mx.mx.loc[ordered_fns, ordered_fns]
        ordmx = SimMx(self.language, name='olo', mx=ordmx, normalized=True, is_sim=True, is_directed = self.mx.is_directed, is_condensed=False)

        nr_texts = DataHandler(self.language).nr_texts
        assert (ordmx.mx.shape[0] == nr_texts) and (ordmx.mx.shape[1] == nr_texts) 

        return ordmx


class MxViz(VizBase):
    def __init__(self, language, mx, info, plttitle, expname):
        self.cmode = 'mx'
        super().__init__(language, self.cmode, info, plttitle, expname)
        self.mx = mx
        self.n_jobs = 1

        # Whitespace
        self.ws_left = 0.03
        self.ws_right = 0.97
        self.ws_bottom = 0.05
        self.ws_top = 0.95
        self.ws_wspace = 0.2
        self.ws_hspace = 0.2


    def visualize(self):
        self.pos = self.get_mds_positions()
        self.df = self.prepare_metadata()
        self.get_figure()
        self.fill_subplots()
        self.add_legends_and_titles()
        path = self.get_path(data_type=self.data_type)
        self.save_data(data=plt, data_type=self.data_type, file_name=None, file_path=path, plt_kwargs={'dpi': 600})
        plt.show()
  
        
    def get_figure(self):
        ncol = 4
        if self.is_cat:
            ncol += 1

        if self.has_special:
            ncol += 1

        width = 5
        self.fig, self.axs = plt.subplots(2, ncol, figsize=(ncol*width, 2*width))

        gridpos = ncol + 2
        for i in range(1, ncol):
            # In add_subplot, speciy nrows, ncols, position of subplot in the grid
            # The first index is in the second row and the second col
            self.axs[1, i].axis('off')
            self.axs[1, i] = self.fig.add_subplot(2, ncol, gridpos, projection='3d')
            gridpos += 1

        # # Set aspect ratio to 'equal' for 3D subplots
        # axs3d = [self.axs[1, 1], self.axs[1, 2], self.axs[1, 3]]
        # for ax in axs3d:
        #     ax.set_box_aspect([1, 1, 1])

        self.axs[1, 0].axis('off')
        self.fig.subplots_adjust(
            left=self.ws_left,
            right=self.ws_right,
            bottom=self.ws_bottom,
            top=self.ws_top,
            wspace=self.ws_wspace,
            hspace=self.ws_hspace
        )
            
        axs2d = [self.axs[0, 1], self.axs[0, 2], self.axs[0, 3]]
        if self.is_cat:
            axs2d.append(self.axs[0, 4])
        # Set aspect ratio to 'equal'
        for ax in axs2d:
            ax.set_aspect('equal', adjustable='datalim')


        self.attrix = [0, 1]
        self.clstix = [0, 2]
        self.shapeix = [0, 3]

        if self.is_cat and self.has_special:
            self.combix = [0,4]
            self.specix = [0,5]
        if self.is_cat and not self.has_special:
            self.combix = [0,4]
            self.specix = None
        if not self.is_cat and self.has_special:
            self.combix = None
            self.specix = [0,4]
        if not self.is_cat and not self.has_special:
            self.combix = None
            self.specix = None


    def add_legends_and_titles(self):
        # if self.needs_cbar:
        #     self.add_cbar(self.axs[-2, -1])

        self.add_legend(self.fig, 'cluster', label='size', loc='upper left', boxx=self.ws_left, boxy=0.4, boxwidth=0.05, boxheight=0.1, fontsize=self.fontsize, use_shapes=True)
        if self.is_cat:
            self.add_legend(self.fig, self.info.attr, label='attr', loc='upper left', boxx=self.ws_left + 0.05, boxy=0.4, boxwidth=0.1, boxheight=0.1, fontsize=self.fontsize)

        self.add_subtitles(self.attrix, self.clstix, self.shapeix, self.combix, self.specix)
        # Place the title at the bottom left below the heatmap
        self.add_text(self.axs[1,0], x=self.ws_left, y=self.ws_bottom, width=50)


    def fill_subplots(self):
        # attr
        self.draw_mds(self.attrix, color_col=self.info.attr, use_different_shapes=False)
        self.draw_heatmap(self.axs[0, 0])

        # cluster
        self.draw_mds(self.clstix, color_col='cluster', use_different_shapes=False)

        # attr as color, cluster as shapes
        self.draw_mds(self.shapeix, color_col=self.info.attr, use_different_shapes=True)

        # cluster and attr combined as colors
        if self.is_cat:
            self.draw_mds(self.combix, color_col=f'{self.info.attr}_cluster', use_different_shapes=False)

        if self.has_special:
            self.draw_mds(self.specix, color_col=self.info.special, use_different_shapes=False)


    def get_mds_positions(self):
        # Store layouts because it takes a lot of time to calculate them
        pkl_path = self.get_file_path(file_name=f'mds-{self.mx.name}.pkl', subdir=True) 
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                df = pickle.load(f)

        else:
            # Apply classical MDS
            mds_2d = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto', n_jobs=self.n_jobs, random_state=8)
            X_mds_2d = mds_2d.fit_transform(self.mx.dmx)

            # Apply classical MDS in 3D
            mds_3d = MDS(n_components=3, dissimilarity='precomputed', normalized_stress='auto', n_jobs=self.n_jobs, random_state=8)
            X_mds_3d = mds_3d.fit_transform(self.mx.dmx)

            df = pd.DataFrame({
                'X_mds_2d_0': X_mds_2d[:, 0],
                'X_mds_2d_1': X_mds_2d[:, 1],
                'X_mds_3d_0': X_mds_3d[:, 0],
                'X_mds_3d_1': X_mds_3d[:, 1],
                'X_mds_3d_2': X_mds_3d[:, 2],
                })
            
            df.index = self.mx.dmx.index

            with open(pkl_path, 'wb') as f:
                pickle.dump(df, f)
        return df


    def prepare_metadata(self):
        df = self.info.metadf.merge(self.pos, how='inner', left_index=True, right_index=True, validate='1:1')
        return df


    def draw_mds(self, ix, color_col=None, use_different_shapes=False):
        color_col = f'{color_col}_color'
        
        df = self.df.copy() # Avoid chained assingment warning
        # Iterate through shapes because only one shape can be passed at a time, no lists
        if not use_different_shapes:
            df['clst_shape'] = 'o'
        shapes = df['clst_shape'].unique()

        for shape in shapes:
            sdf = df[df['clst_shape'] == shape]
            kwargs = {'c': sdf[color_col], 'marker': shape, 's': 30, 'edgecolor': 'black', 'linewidth': 0.2} ################10
            self.axs[0, ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)
            self.axs[1, ix[1]].scatter(sdf['X_mds_3d_0'], sdf['X_mds_3d_1'], sdf['X_mds_3d_2'], **kwargs)


    def draw_heatmap(self, ax):
        # Draw heatmap
        ordmx = MxReorder(self.language, self.mx, self.info).order()

        im = ax.imshow(ordmx, cmap=Colors.CMAP, interpolation='nearest')
        ax.axis('off')  # Remove the axis/grid
        ax.set_title('Attribute', fontsize=self.fontsize)

        # Add a color bar to the heatmap for better understanding of the similarity values
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.1, location='left')

