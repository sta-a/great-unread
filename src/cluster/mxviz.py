
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
from .cluster_utils import VizBase
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
    def __init__(self, language, mx, info, plttitle):
        self.cmode = 'mx'
        super().__init__(language, self.cmode, info, plttitle)
        self.mx = mx
        self.n_jobs = 1


    def visualize(self):
        self.fill_subplots()

  
    def get_figure(self):
        if self.info.attr in self.cat_attrs:
            fig_width = 20
            fig_height = 0.4 * fig_width
            self.fig = plt.figure(figsize=(fig_width, fig_height))
            gs = self.fig.add_gridspec(2, 5)
        else:
            fig_width = 20
            fig_height = 0.5 * fig_width
            self.fig = plt.figure(figsize=(fig_width, fig_height))
            gs = self.fig.add_gridspec(2, 4)

        self.ax0 = self.fig.add_subplot(gs[0,0])

        self.ax1 = self.fig.add_subplot(gs[0,1])
        self.ax2 = self.fig.add_subplot(gs[1,1], projection='3d')

        self.ax3 = self.fig.add_subplot(gs[0,2])
        self.ax4 = self.fig.add_subplot(gs[1,2], projection='3d')

        self.ax5 = self.fig.add_subplot(gs[0,3])
        self.ax6 = self.fig.add_subplot(gs[1,3], projection='3d')

        if self.info.attr in self.cat_attrs:
            self.ax7 = self.fig.add_subplot(gs[0,4])
            self.ax8 = self.fig.add_subplot(gs[1,4], projection='3d')


    def fill_subplots(self):
        self.get_figure()
        # kwargs = {'aspect': 'auto'} ###################
        # constrained_layout, tight_layout

        # attr
        self.draw_mds(self.ax1, self.ax2, color_col=self.info.attr, use_different_shapes=False)
        self.draw_heatmap(self.ax0)

        # cluster
        self.draw_mds(self.ax3, self.ax4, color_col='cluster', use_different_shapes=False)

        # attr as color, cluster as shapes
        self.draw_mds(self.ax5, self.ax6, color_col=self.info.attr, use_different_shapes=True)

        # cluster and attr combined as colors
        if self.info.attr in self.cat_attrs:
            self.draw_mds(self.ax7, self.ax8, color_col=f'{self.info.attr}_cluster', use_different_shapes=False)

        self.add_subplot_titles(attrax=self.ax1, clstax=self.ax3, shapeax=self.ax5, combax=self.ax7)
        self.manage_big_figure()
        fig_path = self.get_path('big', omit=[], data_type=self.data_type)
        # self.save_plot(plt, file_name=None, file_path=fig_path)
        # plt.close()
        plt.show()


    def draw_mds(self, ax1, ax2, color_col=None, use_different_shapes=False):
            color_col = f'{color_col}_color'

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

                with open(pkl_path, 'wb') as f:
                    pickle.dump(df, f)


            # Shapes
            df = df.assign(shape=['o']*len(self.mx.dmx))
            if use_different_shapes:
                df = df.assign(shape=self.info.metadf.loc[self.mx.dmx.index, 'clst_shape'].values)

            # Colors
            df = df.assign(color=self.info.metadf.loc[self.mx.dmx.index, color_col].values)


            for shape in df['shape'].unique():
                sdf = df[df['shape'] == shape]
                kwargs = {'c': sdf['color'], 'marker': shape, 's': 10}
                ax1.scatter(sdf['X_mds_2d_0'], sdf['X_mds_2d_1'], **kwargs)
                ax2.scatter(sdf['X_mds_3d_0'], sdf['X_mds_3d_1'], sdf['X_mds_3d_2'], **kwargs)



    def draw_heatmap(self, ax5):
        # Draw heatmap
        ordmx = MxReorder(self.language, self.mx, self.info).order()

        # hot_r, viridis, plasma, inferno
        # ordmx = np.triu(ordmx) ####################
        im = ax5.imshow(ordmx, cmap='coolwarm', interpolation='nearest')
        ax5.axis('off')  # Remove the axis/grid

        # Add a color bar to the heatmap for better understanding of the similarity values
        cbar = plt.colorbar(im, ax=ax5, fraction=0.05, pad=0.1)

