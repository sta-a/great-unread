import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import os
import random
random.seed(9)
from tqdm import tqdm

from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform

import sys
sys.path.append("..")
from .viz_utils import VizBase, ImageGrid
from .nkviz import NkSingleVizAttr
from cluster.create import SimMx
from cluster.cluster_utils import Colors
from cluster.combinations import InfoHandler
from typing import List

import logging
logging.basicConfig(level=logging.DEBUG)


class MxReorder():
    '''Sort row and column indices so that clusters are visible in heatmap.'''

    ORDERS = ['fn', 'olo']

    def __init__(self, language, mx, info, by_author):
        self.language = language
        self.mx = mx
        self.info = info
        self.by_author = by_author
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

        # assert self.mx.mx.shape == ordmx.shape
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
        ordmx = SimMx(self.language, name='olo', mx=ordmx, normalized=True, is_sim=True, is_directed = self.mx.is_directed, is_condensed=False, by_author=self.by_author)
        return ordmx



class MxVizBase(VizBase):
    def __init__(self, language, output_dir, mx, info, plttitle, exp, by_author=False):
        self.cmode = 'mx'
        super().__init__(language, output_dir, self.cmode, info, plttitle, exp, by_author)
        self.mx = mx
        self.n_jobs = 1

        # Whitespace
        self.ws_left = 0.03
        self.ws_right = 0.97
        self.ws_bottom = 0.05
        self.ws_top = 0.95
        self.ws_wspace = 0.2
        self.ws_hspace = 0.2
        self.is_topattr_viz = True ###################################3


    def visualize(self, vizname='viz', omit=[]):
        self.vizpath = self.get_path(name=vizname, omit=omit)
        if not os.path.exists(self.vizpath):
            self.pos = self.get_mds_positions()
            self.add_positions_to_metadf()
            self.get_figure()
            self.fill_subplots()
            self.add_legends_and_titles()
            self.save_plot(plt)
            # plt.show()


    def get_mds_path(self):
        mds_dir = os.path.join(self.output_dir, 'mds')
        if not os.path.exists(mds_dir):
            self.create_dir(mds_dir)
        path = os.path.join(mds_dir, f'mds-{self.mx.name}.pkl')
        return path


    def get_mds_positions(self):
        pkl_path = self.get_mds_path() 
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                df = pickle.load(f)

        else:
            # Apply classical MDS
            mds_2d = MDS(n_components=2, dissimilarity='precomputed',  n_jobs=self.n_jobs, random_state=8)
            X_mds_2d = mds_2d.fit_transform(self.mx.dmx)

            # Apply classical MDS in 3D
            mds_3d = MDS(n_components=3, dissimilarity='precomputed', n_jobs=self.n_jobs, random_state=8)
            X_mds_3d = mds_3d.fit_transform(self.mx.dmx)

            df = pd.DataFrame({
                'X_mds_2d_0': X_mds_2d[:, 0],
                'X_mds_2d_1': X_mds_2d[:, 1],
                'X_mds_3d_0': X_mds_3d[:, 0],
                'X_mds_3d_1': X_mds_3d[:, 1],
                'X_mds_3d_2': X_mds_3d[:, 2],
                })
            
            df.index = self.mx.dmx.index
            df.index = df.index.astype(str)

            with open(pkl_path, 'wb') as f:
                pickle.dump(df, f)
        return df


    def add_positions_to_metadf(self, load_metadf=True):
        print('add_positions_to_metadf')
        # Combine positions and metadata
        if load_metadf:
            self.get_metadf()
        self.df = self.df.merge(self.pos, how='inner', left_index=True, right_index=True, validate='1:1', suffixes = ['_xsuffix', '_ysuffix'])


    def draw_mds(self, ix, color_col=None, use_different_shapes=False, s=30, edgecolor='black', linewidth=0.2):
        scatter_kwargs = {'s': s, 'edgecolor': edgecolor, 'linewidth': linewidth}
        color_col = f'{color_col}_color'
        
        df = self.df.copy() # Avoid chained assingment warning
        # Iterate through shapes because only one shape can be passed at a time, no lists
        if not use_different_shapes:
            df['clst_shape'] = 'o'
        shapes = df['clst_shape'].unique()

        for shape in shapes:
            sdf = df[df['clst_shape'] == shape]
            kwargs = {'c': sdf[color_col], 'marker': shape, **scatter_kwargs}
            self.axs[ix[0], ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)
            self.axs[ix[0]+1, ix[1]].scatter(sdf['X_mds_3d_0'], sdf['X_mds_3d_1'], sdf['X_mds_3d_2'], **kwargs)


    def draw_heatmap(self, ax):
        # Draw heatmap
        ordmx = MxReorder(self.language, self.mx, self.info, by_author=self.by_author).order()

        im = ax.imshow(ordmx, cmap=Colors.CMAP, interpolation='nearest')
        ax.axis('off')  # Remove the axis/grid
        ax.set_title('Attribute', fontsize=self.fontsize)

        # Add a color bar to the heatmap for better understanding of the similarity values
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.1, location='left')


    def adjust_subplots(self):
        self.fig.subplots_adjust(
            left=self.ws_left,
            right=self.ws_right,
            bottom=self.ws_bottom,
            top=self.ws_top,
            wspace=self.ws_wspace,
            hspace=self.ws_hspace)  
        


class MxKeyAttrViz(MxVizBase):
    def __init__(self, language, output_dir, mx, info, plttitle, exp, by_author=False):
        super().__init__(language, output_dir, mx, info, plttitle, exp, by_author)


    def get_figure(self):
        self.ncol = 4
        if self.is_cat:
            self.ncol += 1

        width = 5
        self.nrow = 2
        if self.is_topattr_viz: #########################3
            self.nrow += 2

        self.fig, self.axs = plt.subplots(self.nrow, self.ncol, figsize=(self.ncol*width, self.nrow*width))

        gridpos = self.ncol + 2
        for j in range(1, self.nrow, 2):
            for i in range(1, self.ncol):
                # In add_subplot, speciy nrow, ncol, position of subplot in the grid
                # The first index is in the second row and the second col
                self.axs[j, i].axis('off')
                self.axs[j, i] = self.fig.add_subplot(self.nrow, self.ncol, gridpos, projection='3d')
                gridpos += 1
            gridpos += (self.ncol + 1)

        # # Set aspect ratio to 'equal' for 3D subplots
        # axs3d = [self.axs[1, 1], self.axs[1, 2], self.axs[1, 3]]
        # for ax in axs3d:
        #     ax.set_box_aspect([1, 1, 1])

        for j in range(1, self.nrow):
            self.axs[j, 0].axis('off')

        self.adjust_subplots()
            
        axs2d = [self.axs[0, 1], self.axs[0, 2], self.axs[0, 3]]
        if self.is_cat:
            axs2d.append(self.axs[0, 4])
        # Set aspect ratio to 'equal'
        for ax in axs2d:
            ax.set_aspect('equal', adjustable='datalim')


        self.attrix = [0, 1]
        self.clstix = [0, 2]
        self.shapeix = [0, 3]

        if self.is_cat:
            self.combix = [0,4]
        else:
            self.combix = None


    def add_legends_and_titles(self):
        # if self.needs_cbar:
        #     self.add_cbar(self.axs[-2, -1])

        self.add_legend(self.fig, 'cluster', label='size', loc='upper left', boxx=self.ws_left, boxy=0.4, boxwidth=0.05, boxheight=0.1, fontsize=self.fontsize, use_shapes=True)
        if self.is_cat:
            self.add_legend(self.fig, self.info.attr, label='attr', loc='upper left', boxx=self.ws_left + 0.05, boxy=0.4, boxwidth=0.1, boxheight=0.1, fontsize=self.fontsize)

        self.add_subtitles(self.attrix, self.clstix, self.shapeix, self.combix)
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

        if self.is_topattr_viz:
            self.draw_mds([2, 1], color_col=self.key_attrs[0], use_different_shapes=True)
            self.draw_mds([2, 2], color_col=self.key_attrs[1], use_different_shapes=True)
            self.draw_mds([2, 3], color_col=self.key_attrs[2], use_different_shapes=True)
            self.get_ax([2, 1]).set_title(self.key_attrs[0], fontsize=self.fontsize)
            self.get_ax([2, 2]).set_title(self.key_attrs[1], fontsize=self.fontsize)
            self.get_ax([2, 3]).set_title(self.key_attrs[2], fontsize=self.fontsize)

        for j in range(1, self.nrow):
            for i in range(1, self.ncol):
                if not self.axs[j,i].has_data():
                    self.axs[j,i].axis('off')



class MxAttrGridViz(MxVizBase):
    def __init__(self, language, output_dir, mx, info, plttitle, exp, by_author=False):
        super().__init__(language, output_dir, mx, info, plttitle, exp, by_author)
        self.data_type = 'svg'
        self.fontsize = 5
        self.nrow = 2
        self.ncol = 4


    def create_logfile(self, all_cols, nfields):
        df = pd.DataFrame({'feature': all_cols})

        df['cmode'] = self.cmode
        df['mxname'] = self.info.mxname
        df['distinctive'] = ''

        # Nr of plot that contains feature
        viznr_values = []
        group_size = int(nfields/2)
        for i in range(len(df)):
            viznr_values.append(i // group_size)
        df['viznr'] = viznr_values

        df = df[['cmode', 'mxname', 'viznr', 'feature', 'distinctive']]
        self.save_data(data=df, subdir=True, file_name='visual-assessment.csv', data_type='csv')


    def get_feature_columns(self, df):
        df = deepcopy(df)
        special_cols = ['cluster', 'clst_shape', 'gender_cluster', 'author_cluster']
        # Get list of attributes in interesting order
        if self.exp['name'] == 'attrviz':
            cols = self.key_attrs + [col for col in df.columns if col not in self.key_attrs and col not in special_cols and ('_color' not in col)]
        else:
            cols = self.key_attrs
        return cols


    def visualize(self, vizname='viz'): # vizname for compatibility
        all_cols = self.get_feature_columns(self.info.metadf)
        nfields = self.nrow * self.ncol # fields per plot
        nplots = len(all_cols)*2 # 2 fields used for each feature
        nfig = nplots // nfields 
        if nplots % nfields != 0:
            nfig += 1
        self.create_logfile(all_cols, nfields)

        ix = 0
        for i in range(nfig):
            self.cols = all_cols[ix: ix + int((nfields)/2)]
            # print(ix, ix + int((nfields)/2))
            ix += int(nfields/2)
            
            if nfig == 1:
                vizname = vizname
            else:
                vizname = f'viz{i}'
            super().visualize(vizname=vizname, omit=['clst_alg_params', 'attr'])


    def get_figure(self):
        self.fig, self.axs = plt.subplots(self.nrow, self.ncol, figsize=(15, 7.5))

        for i in range(self.nrow):
            for j in range(self.ncol):
                # Check if the subplot index is even
                if j % 2 != 0:
                    # Plot a 3D plot
                    self.axs[i, j].axis('off')
                    self.axs[i, j] = self.fig.add_subplot(self.nrow, self.ncol, i * self.ncol + j + 1, projection='3d')
                    # ax.plot([0, 1], [0, 1], [0, 1])
                    # self.axs[i, j].axis('off')

        for i in range(self.nrow):
            for j in range(self.ncol):
                self.axs[i, j].set_xticks([])  # Remove x-axis ticks
                self.axs[i, j].set_yticks([])  # Remove y-axis ticks
                self.axs[i, j].set_xticklabels([])  # Remove x-axis tick labels
                self.axs[i, j].set_yticklabels([])  # Remove y-axis tick labels
                self.axs[i, j].set_aspect('equal')  # Set equal aspect ratio for squares
        # plt.subplots_adjust(wspace=0, hspace=0)
    

    def fill_subplots(self):
        for i in range(self.nrow):
            for j in range(self.ncol):
                if j % 2 == 0:
                    index = int((i * self.ncol + j)/2)
                    if index < len(self.cols):
                        self.draw_mds([i, j], color_col=self.cols[index], use_different_shapes=False)



    def draw_mds(self, ix, color_col=None, use_different_shapes=False):
        color_col = f'{color_col}_color'
        
        df = self.df.copy() # Avoid chained assingment warning
        # Iterate through shapes because only one shape can be passed at a time, no lists
        if not use_different_shapes:
            df['clst_shape'] = 'o'
        shapes = df['clst_shape'].unique()

        for shape in shapes:
            sdf = df[df['clst_shape'] == shape]
            kwargs = {'c': sdf[color_col], 'marker': shape, 's': 20, 'edgecolor': 'black', 'linewidth': 0.2}
            self.axs[ix[0], ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)
            self.axs[ix[0], ix[1]+1].scatter(sdf['X_mds_3d_0'], sdf['X_mds_3d_1'], sdf['X_mds_3d_2'], **kwargs)


    def add_legends_and_titles(self):
        for i in range(self.nrow):
            for j in range(self.ncol):
                if j % 2 == 0:
                    index = int((i * self.ncol + j)/2)
                    if index < len(self.cols):
                        self.axs[i, j].set_title(self.cols[index])


class MxSingleViz2D3D(MxVizBase):
    '''
    Create a single plot per matrix, with a 2d and a 3d visualization.
    For each matrix, a seperate plot for each key attribute is created.
    '''
    def __init__(self, language, output_dir, exp, by_author, mc, info=None):
        # language, mx, info, plttitle, exp
        super().__init__(language, output_dir, mx=None, info=info, plttitle=None, exp=exp, by_author=by_author)
        self.mc = mc # EmbMxCombinations or MxCombinations object
        self.fontsize = 6
        self.markersize = 10
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0
        self.ws_hspace = 0
        if info is None:
            self.ih = InfoHandler(language=language, add_color=True, cmode=self.cmode, by_author=by_author)
        self.add_subdir('MxSingleViz2D3D')


    def get_metadf(self):
        self.df = deepcopy(self.ih.metadf)
        self.df['noattr_color'] = 'blue'


    def get_figure(self):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(4, 8))
        self.axs = self.axs.reshape((2, 1))
        self.axs[0, 0].axis('off')
        self.axs[1, 0].axis('off') # has to come before 3d subplots are created
        self.axs[1, 0] = self.fig.add_subplot(2, 1, 2, projection='3d')
        self.adjust_subplots()


    def fill_subplots(self, attr):
        self.draw_mds([0, 0], color_col=attr, use_different_shapes=False, s=self.markersize)


    def visualize(self, vizname='viz'): # vizname for compatibility #########################################
        for mx in tqdm(self.mc.load_mxs()):
            self.mx = mx
            mxname = mx.name
            # Check if plot for last key attr has been created
            vizpath_test = self.get_file_path(f'{mxname}_{self.key_attrs[-1]}', subdir=True)
            print('MxSingleViz2D3D vizpath',  vizpath_test)

            if not os.path.exists(vizpath_test):
                print(vizpath_test)
                self.pos = self.get_mds_positions()
                self.add_positions_to_metadf()

                for curr_attr in self.key_attrs: # + ['noattr']:
                    self.get_figure()
                    self.fill_subplots(curr_attr)
                    self.vizpath = self.get_file_path(f'{mxname}_{curr_attr}', subdir=True)
                    self.save_plot(plt)
                    plt.close()
               
                # self.add_legends_and_titles()



    # def visualize(self, vizname='viz'): # vizname for compatibility
    #     for mx in tqdm(self.mc.load_mxs()):
    #         self.mx = mx
    #         mxname = mx.name
    #         # Check if plot for last key attr has been created

    #         self.pos = self.get_mds_positions()
    #         self.add_positions_to_metadf()

    #         for curr_attr in self.key_attrs + ['noattr']:
    #             self.get_figure()
    #             self.fill_subplots(curr_attr)
    #             self.vizpath = self.get_file_path(f'{mxname}_{curr_attr}', subdir=True)
    #             self.save_plot(plt)
    #             plt.close()
               
                # self.add_legends_and_titles()


class MxSingleViz2D3DHorizontal(MxSingleViz2D3D):
    '''
    The same as MxSingleViz2D3D, but the two plots are aligned in one row.
    '''
    def __init__(self, language, output_dir, exp, by_author, mc, info=None):
        super().__init__(language, output_dir, exp, by_author, mc, info=None)
        self.add_subdir('MxSingleViz2D3DHorizontal')

    def get_figure(self):
        self.fig, self.axs = plt.subplots(1, 2, figsize=(8, 4))
        self.axs = self.axs.reshape((1, 2))
        self.axs[0, 0].axis('off')
        self.axs[0, 1].axis('off')  # has to come before 3d subplots are created
        self.axs[0, 1] = self.fig.add_subplot(1, 2, 2, projection='3d')
        self.adjust_subplots()

    def draw_mds(self, ix, color_col=None, use_different_shapes=False, s=30, edgecolor='black', linewidth=0.2):
        scatter_kwargs = {'s': s, 'edgecolor': edgecolor, 'linewidth': linewidth}
        color_col = f'{color_col}_color'
        
        df = self.df.copy() # Avoid chained assingment warning
        # Iterate through shapes because only one shape can be passed at a time, no lists
        if not use_different_shapes:
            df['clst_shape'] = 'o'
        shapes = df['clst_shape'].unique()

        for shape in shapes:
            sdf = df[df['clst_shape'] == shape]
            kwargs = {'c': sdf[color_col], 'marker': shape, **scatter_kwargs}
            self.axs[ix[0], ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)
            self.axs[ix[0], ix[1]+1].scatter(sdf['X_mds_3d_0'], sdf['X_mds_3d_1'], sdf['X_mds_3d_2'], **kwargs)


class MxSingleViz2d3dSingleAttr(MxSingleViz2D3D):
    '''
    The same as MxSingleViz2D3D, but visualize only the attribute that is passed as 'info.attr'. Don't iterate over all matrices but only use the one passed as 'mx'.
    '''
    def __init__(self, language, output_dir, exp, by_author, mc, info, mx):
        super().__init__(language, output_dir, exp, by_author, mc, info)
        self.attr = self.info.attr
        self.mx = mx
        self.add_subdir('mx_singleimage_s2v')
        self.markersize = 15 # size of scatter points
        self.fontsize = 10


    def get_file_path(self):
        if self.attr == 'cluster':
            file_name = f'{self.info.as_string(omit=["attr"])}.png'
        else:
            file_name = f'{self.mx.name}_{self.attr}'
        return super().get_file_path(file_name, subdir=True)
    

    def get_metadf(self):
        self.df = deepcopy(self.info.metadf)


    def visualize(self, vizname=''): # vizname for compatibility
        self.vizpath = self.get_file_path()
        print(self.vizpath)
        if not os.path.exists(self.vizpath):
            self.pos = self.get_mds_positions()
            self.add_positions_to_metadf()

            for curr_attr in [self.attr]:
                self.get_figure()
                self.fill_subplots(curr_attr)
                self.save_plot(plt, plt_kwargs={'dpi': 200})
                plt.close()


class MxSingleViz2dSingleAttr(MxSingleViz2d3dSingleAttr):
    '''
    Combination of MxSingleViz2d3dSingleAttr with get_figure and draw_mds methods copied from MxSingleViz
    Makes 2d MDS for position clustering, for attrs and clusters
    This class is only used for the thesis presentation.
    '''
    def __init__(self, language, output_dir, exp, by_author, mc, info, mx):
        super().__init__(language, output_dir, exp, by_author, mc, info, mx)
        self.add_subdir('MxSingleViz2dSingleAttr_test')

    def get_figure(self):
        self.fig, self.axs = plt.subplots(1, 1, figsize=(4, 4))
        self.axs = np.reshape(self.axs, (1, 1))
        self.axs[0, 0].axis('off')
        self.adjust_subplots()


    def draw_mds(self, ix, color_col=None, use_different_shapes=False, s=30, edgecolor='black', linewidth=0.2):
        scatter_kwargs = {'s': s, 'edgecolor': edgecolor, 'linewidth': linewidth}
        color_col = f'{color_col}_color'
        
        df = self.df.copy() # Avoid chained assingment warning
        # Iterate through shapes because only one shape can be passed at a time, no lists
        if not use_different_shapes:
            df['clst_shape'] = 'o'
        shapes = df['clst_shape'].unique()

        for shape in shapes:
            sdf = df[df['clst_shape'] == shape]
            kwargs = {'c': sdf[color_col], 'marker': shape, **scatter_kwargs}
            self.axs[ix[0], ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)


class MxSingleViz(MxSingleViz2D3D):
    '''
    Create a single plot per matrix, with a 2d visualization.
    For s2v, the different network positions should be clearly distinguishable in 2D.
    For each matrix, a seperate plot for each key attribute is created.
    This class is only used for the thesis presentation.
    '''
    def __init__(self, language, output_dir, exp, by_author, mc):
        super().__init__(language, output_dir, exp, by_author, mc)
        self.markersize = 20
        self.add_subdir('MxSingleViz_test')

    def get_figure(self):
        self.fig, self.axs = plt.subplots(1, 1, figsize=(4, 4))
        self.axs = np.reshape(self.axs, (1, 1))
        self.axs[0, 0].axis('off')
        self.adjust_subplots()


    def draw_mds(self, ix, color_col=None, use_different_shapes=False, s=30, edgecolor='black', linewidth=0.2):
        scatter_kwargs = {'s': s, 'edgecolor': edgecolor, 'linewidth': linewidth}
        color_col = f'{color_col}_color'
        
        df = self.df.copy() # Avoid chained assingment warning
        # Iterate through shapes because only one shape can be passed at a time, no lists
        if not use_different_shapes:
            df['clst_shape'] = 'o'
        shapes = df['clst_shape'].unique()

        for shape in shapes:
            sdf = df[df['clst_shape'] == shape]
            kwargs = {'c': sdf[color_col], 'marker': shape, **scatter_kwargs}
            self.axs[ix[0], ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)



class MxSingleVizCluster(MxVizBase):
    '''
    Make 2D viz of attr and clusters, only used in additional experiments for thesis
    '''
    def __init__(self, language, output_dir, mx, info, plttitle, exp, by_author):
        super().__init__(language, output_dir, mx=mx, info=info, plttitle=plttitle, exp=exp, by_author=by_author)
        self.markersize = 20
        self.add_subdir('MxSingleVizCluster')


    def get_figure(self):
        self.fig, self.axs = plt.subplots(1, 1, figsize=(4, 4))
        self.axs = np.reshape(self.axs, (1, 1))
        self.axs[0, 0].axis('off')
        self.adjust_subplots()

    def fill_subplots(self):
        print('color col', self.attr)
        self.draw_mds([0, 0], color_col=self.attr, use_different_shapes=False, s=self.markersize)

    def draw_mds(self, ix, color_col=None, use_different_shapes=False, s=30, edgecolor='black', linewidth=0.2):
        scatter_kwargs = {'s': s, 'edgecolor': edgecolor, 'linewidth': linewidth}
        color_col = f'{color_col}_color'
        
        df = self.df.copy() # Avoid chained assingment warning
        # Iterate through shapes because only one shape can be passed at a time, no lists
        if not use_different_shapes:
            df['clst_shape'] = 'o'
        shapes = df['clst_shape'].unique()

        for shape in shapes:
            sdf = df[df['clst_shape'] == shape]
            kwargs = {'c': sdf[color_col], 'marker': shape, **scatter_kwargs}
            self.axs[ix[0], ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)


    # Overwrite method to remove 'viz' from file name
    def get_path(self, omit: List[str]=[], data_type=None):
        if data_type is None:
            data_type = self.data_type
        file_name = f'{self.info.as_string(omit=omit)}.{data_type}'
        return self.get_file_path(file_name, subdir=True)

    def visualize(self, vizname='viz', omit=[]):
        for attr in ['cluster', self.info.attr]:
            self.attr = attr
            self.info.attr = attr
            self.vizpath = self.get_path() # 'cluster' is also added to vizpath
            
            if not os.path.exists(self.vizpath):
                print('################\n', self.vizpath, '\n##################3\n')
                self.pos = self.get_mds_positions()
                self.add_positions_to_metadf()
                self.get_figure()
                self.fill_subplots()
                self.save_plot(plt)
                # plt.show()


class S2vKeyAttrViz(ImageGrid):
    '''
    First row: original network
    Second row: S2v MDS
    '''

    def __init__(self, language, mx, info, plttitle, exp, by_author, subdir=None):
        self.mx = mx
        self.info = info
        self.plttitle = plttitle
        self.exp = exp
        self.by_author = by_author

        self.colnames = ['cluster', 'canon', 'gender', 'year'] 
        if self.by_author:
            self.colnames = self.colnames + ['canon-min', 'canon-max']
        else: 
            # Insert 'author' right after 'canon'
            index = self.colnames.index('canon') + 1
            self.colnames.insert(index, 'author')

        self.subdir = subdir #
        super().__init__(language=language, by_author=by_author, output_dir='analysis_s2v', rowmajor=False, imgs_as_paths=True, subdir=self.subdir) # load_single_images is called in ImageGrid.__init__
        self.nrow = 2
        self.ncol = len(self.colnames)

        self.img_width = 4
        self.img_height = 6 # img_height * 2 rows = 12 for 3 subplots with height 4 (2d3d are in one image)
        self.fontsize = 12


    def get_file_path(self, vizname=None, subdir=None, **kwargs):
        return os.path.join(self.subdir, f'{self.info.as_string(omit=["attr"])}.{self.data_type}')


    def load_single_images(self):
        mxname, spars = self.info.as_string().split('_')[:2]
        self.info.spmx_path = os.path.join(self.data_dir, 'similarity', self.language, 'sparsification', f'sparsmx-{mxname}_{spars}.pkl')
        # self.nk_attr_dir = os.path.join(self.data_dir, 'analysis', self.language, 'nk_singleimage')
        # self.mds_attr_dir = os.path.join(self.output_dir, 'mx_singleimage')

        imgs = []
        for attr in self.colnames:
            self.info.attr = attr


            nkclust = NkSingleVizAttr(self.language, self.output_dir, self.info, plttitle=self.plttitle, exp=self.exp, by_author=self.by_author)
            nkclust_path = nkclust.get_path()
            nkclust.visualize()
            imgs.append(nkclust_path)


            # Just for comparison , delete ·##################################
            # mdsclust = MxSingleVizCluster(self.language, self.output_dir, self.mx, self.info, self.plttitle, self.exp, self.by_author)
            # mxclust_path = mdsclust.get_path(omit=['attr'])
            # if not os.path.exists(mxclust_path):
            #     mdsclust.visualize()


            mdsclust = MxSingleViz2d3dSingleAttr(self.language, self.output_dir, self.exp, self.by_author, mc=None, info=self.info, mx=self.mx)
            mdsclust.visualize()
            mxclust_path = mdsclust.get_file_path()
            imgs.append(mxclust_path)


        return imgs

    def get_title(self, imgpath):
        if 'mx_singleimage_s2v' in imgpath:
            title = ''
        else:
            basename = os.path.splitext(os.path.basename(imgpath))[0]
            title = basename.split('_')[-1] # attr
        return title
    
    def get_figure(self):
        self.fig, self.axs = plt.subplots(self.nrow, self.ncol, figsize=(self.ncol*self.img_width, self.nrow*self.img_height), gridspec_kw={'height_ratios': [1, 2]})
        plt.tight_layout(pad=0)
