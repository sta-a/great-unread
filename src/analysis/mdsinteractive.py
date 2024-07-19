# %%
import pandas as pd
from copy import deepcopy
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm
from typing import List
import mplcursors
from utils import DataHandler

from .mxviz import NkSingleVizAttr
from cluster.combinations import InfoHandler
from cluster.cluster_utils import CombinationInfo
from analysis.mxviz import MxSingleViz2D3DHorizontal


class NkSingleVizAttrAnalysis(NkSingleVizAttr):
    def __init__(self, language, output_dir, info, exp, by_author, labels):
        super().__init__(language, output_dir, info=info, plttitle=None, exp=exp, by_author=by_author)
        self.labels = labels
        self.add_subdir('MxNkAnalysis')

    def get_metadf(self):
        print('info metadf nk', self.info.metadf.shape)
        df = self.info.metadf
        assert df.shape[0] == self.nr_texts
        self.df = df

    def get_path(self):
        file_name = f"{self.info.as_string(omit=['sparsmode'])}_nk.{self.data_type}"
        return self.get_file_path(file_name, subdir=True)

    def fill_subplots(self):
        self.add_nodes_to_ax([0,0], self.df, color_col='label', use_different_shapes=False)

    def visualize(self, vizname='viz', omit=[]):
        # if not self.too_many_edges:
        self.vizpath = self.get_path()
        if not os.path.exists(self.vizpath):
            self.get_graphs()
            self.get_positions()
            self.add_positions_to_metadf()

            self.get_figure()
            self.adjust_subplots()
            self.add_edges()

            self.fill_subplots()
            self.save_plot(plt)
            plt.show()
            # plt.close()


class MxSingleViz2D3DHzAnalysis(MxSingleViz2D3DHorizontal):
    def __init__(self, language, output_dir, exp, by_author, mc, info=None):
        super().__init__(language, output_dir, exp, by_author, mc, info=None)
        self.add_subdir('MxNkAnalysis')
        self.markersize = 20
        self.get_metadf()
        self.key_attrs = ['canon']
        ih = InfoHandler(language=self.language, add_color=False, cmode='mx', by_author=self.by_author)
        self.df_nocolor = deepcopy(ih.metadf) # Df with all nodes, self.df has only nodes with positions (no iso nodes)
        self.get_colors_list()


    def get_colors_list(self):
        self.colors_list = [
            'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown',
            'mediumvioletred', 'white', 'crimson', 'cyan', 'magenta', 'rosybrown', 'lime', 'navy',
            'teal', 'olive', 'gold', 'silver', 'darkgoldenrod', 'coral', 'darkred', 'lightseagreen',
            'turquoise', 'lavender', 'indigo', 'violet', 'salmon', 'darkseagreen']


    def get_metadf(self):
        self.ih = InfoHandler(language=self.language, add_color=True, cmode='mx', by_author=self.by_author)
        self.df = deepcopy(self.ih.metadf)


    def get_networks(self):
        parts = self.mxname.split('_', maxsplit=2)
        distname, spars = parts[:2]
        spmx_path = os.path.join(self.data_dir, 'similarity', self.language, 'sparsification', f'sparsmx-{distname}_{spars}.pkl')
        df_nk = self.get_metadf_for_labels()
        info = CombinationInfo(mxname=self.mxname, sparsmode=spars, spmx_path=spmx_path, attr=self.curr_attr, metadf=df_nk)
        nk = NkSingleVizAttrAnalysis(language=self.language, output_dir=self.output_dir, info=info, exp=self.exp, by_author=self.by_author, labels=self.labels)
        nk.visualize()


    def map_colors_to_nested_list(self, df):
        colors = plt.get_cmap('tab10').colors
        color_map = {tuple(sublist): colors[i % len(colors)] for i, sublist in enumerate(self.labels)}

        # Create a mapping from index to color
        def get_color(index):
            for sublist in self.labels:
                if index in sublist:
                    return color_map[tuple(sublist)] # Use tuple as dict key
            return 'lightgray'

        # Apply color mapping based on index
        df['label_color'] = df.index.to_series().apply(get_color)
        return df


    def get_metadf_for_labels(self):
        '''
        Graph comes from spmx (all nodes). info.df comes from s2v edgelist (only non-iso nodes).
        Iso nodes are missing from metadata.
        '''
        # df with all nodes, for network with iso nodes
        df_nk = deepcopy(self.df_nocolor)
        assert df_nk.shape[0] == self.nr_texts
        df_nk = self.map_colors_to_nested_list(df_nk)

        for sublist in self.labels:
            print(f'Colors for sublist {sublist}:')
            for item in sublist:
                color = df_nk.loc[item, 'label_color']
                print(f'  {item}: {color}')

        self.df = self.map_colors_to_nested_list(self.df)

        for sublist in self.labels:
            print(f'Colors for sublist {sublist}:')
            for item in sublist:
                color = self.df.loc[item, 'label_color']
                print(f'  {item}: {color}')
        return df_nk

            
    def draw_mds(self, ix, color_col=None, use_different_shapes=False, s=30, edgecolor='black', linewidth=0.2):
        self.labels = []
        self.labels2d = []
        self.labels3d = []
        self.get_interactive_plots(ix, color_col, use_different_shapes, s, edgecolor, linewidth)

        if 'label_color' not in self.df.columns:
            assert len(self.labels) == 0, 'There are labels but no label_color column. Network button was not pressed.'
            print('No labels were selected. No plots were saved.')
            with open(self.results_path, 'w') as f: # Create empty file as reminder that combination has been checked
                f.write('')
        else:
            # Create MDS plots where selected points are highlighted
            self.get_figure() # recreate figure
            # super().draw_mds(ix=ix, color_col='label', use_different_shapes=use_different_shapes, s=s, edgecolor=edgecolor, linewidth=linewidth)
            scatter_kwargs = {'s': s, 'edgecolor': edgecolor, 'linewidth': linewidth}
            color_col = f'label_color'
            sdf = self.df.copy() # Avoid chained assingment warning
            kwargs = {'c': sdf[color_col], 'marker': 'o', **scatter_kwargs}
            self.axs[ix[0], ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)
            self.axs[ix[0], ix[1]+1].scatter(sdf['X_mds_3d_0'], sdf['X_mds_3d_1'], sdf['X_mds_3d_2'], **kwargs)
            self.save_plot(plt, plt_kwargs={'dpi': 100})
            plt.close()
    
    def get_interactive_plots(self, ix, color_col, use_different_shapes, s, edgecolor, linewidth):

        def on_button_click(event):
            for i in self.labels2d:
                print(i)
            for j in self.labels3d:
                print(j)
            self.labels.append(list(set(self.labels2d + self.labels3d))) # combine lists
            for sublist in self.labels:
                print(sublist)
            self.get_networks()

        def on_class_button_click(event):
            for i in self.labels2d:
                print('lbl 2d', i)
            for i in self.labels3d:
                print('lbl 3d', i)
            print('\n\n---------------listset', list(set(self.labels2d + self.labels3d)))
            self.labels.append(list(set(self.labels2d + self.labels3d)))
            for sublist in self.labels:
                print('sublist', sublist)
            self.labels2d = []
            self.labels3d = []
            print('new button################')

        def write_labels_to_file(label, row, dim):
            with open(self.results_path, 'a') as f:
                f.write(f"{self.mxname},,{self.curr_attr},{dim},{label},{','.join(map(str, row))}\n") # extra comma for comment column

        def on_close(event):
            # If points were selected, write a comment
            if self.labels and self.labels != [[]]:
                comment = input('Enter a comment. Must not contain any commas! ') # commas used as sep
                # self.get_networks()
                self.write_comment_to_file(comment)
            

        scatter_kwargs = {'s': s, 'edgecolor': edgecolor, 'linewidth': linewidth}
        color_col = f'{color_col}_color'
        
        sdf = self.df.copy() # Avoid chained assingment warning
        kwargs = {'c': sdf[color_col], 'marker': 'o', **scatter_kwargs}
        
        # 2D plot
        scatter_2d = self.axs[ix[0], ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)
        cursor_2d = mplcursors.cursor(scatter_2d, hover=False)
        @cursor_2d.connect('add')
        def on_add_2d(sel):
            i = sel.index
            label = sdf.index[i]
            if label not in self.labels2d:
                self.labels2d.append(label)
                row = self.df_nocolor.loc[label].values.tolist() 
                write_labels_to_file(label, row, dim='2d')
            sel.annotation.set(text=label, fontsize=8, ha='right', color=sdf[color_col].iloc[i])
        
        # 3D plot
        ax_3d = self.axs[ix[0], ix[1] + 1]
        scatter_3d = ax_3d.scatter(sdf['X_mds_3d_0'], sdf['X_mds_3d_1'], sdf['X_mds_3d_2'], **kwargs)
        cursor_3d = mplcursors.cursor(scatter_3d, hover=False)
        @cursor_3d.connect('add')
        def on_add_3d(sel):
            i = sel.index
            label = sdf.index[i]
            if label not in self.labels3d:
                self.labels3d.append(label)
                row = self.df_nocolor.loc[label].values.tolist()
                write_labels_to_file(label, row, dim='3d')
            sel.annotation.set(text=label, fontsize=8, ha='center', color=sdf[color_col].iloc[i])



    # Create a button to trigger network generation
        ax_button = plt.axes([0.81, 0.01, 0.1, 0.075])  # position: [left, bottom, width, height]
        button = Button(ax_button, 'Generate Network')
        button.on_clicked(on_button_click)

        ax_class_button = plt.axes([0.81, 0.09, 0.1, 0.075])  # position: [left, bottom, width, height]
        button_class = Button(ax_class_button, 'New')
        button_class.on_clicked(on_class_button_click)

        # create a non-maximized window with the size of a maximized one
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


        # Pop up window can be closed with ctrl+Q+Q
        plt.gcf().canvas.mpl_connect('close_event', on_close)

        self.fig.suptitle(self.mxname)
        # self.save_plot(plt, plt_kwargs={'dpi': 700}) # save before show to avoid empty plot
        plt.show()


    def write_comment_to_file(self, comment=','):
        comment = comment.replace(',', '.') # replace commas because they are used as seperators in the csv
        with open(self.results_path, 'a') as f:
            ncols = self.df_nocolor.shape[1]
            commas = (ncols+2) * ','
            f.write(f'{self.mxname},{comment}{commas}\n')

    def write_header(self):
        if not os.path.exists(self.results_path):
            with open(self.results_path, 'w') as f:
                f.write(f"mxname,comment,curr_attr,dim,label,{','.join(self.df_nocolor.columns)}\n")

    def get_results_path(self, mxname, curr_attr):
        self.results_path = os.path.join(self.subdir, f'{mxname}_{curr_attr}_results.txt')


    def visualize(self, vizname='viz'):
        for mx in self.mc.load_mxs():
            self.mx = mx
            self.mxname = mx.name
            print('mx dimensions', self.mx.mx.shape)
            # Check if results for last key attr has been created
            # This is faster than adding the positions and then checking the paths
            self.get_results_path(self.mxname, self.key_attrs[-1])
            if os.path.exists(self.results_path):
                print('results already exist', self.results_path)

            if not os.path.exists(self.results_path):
                self.pos = self.get_mds_positions()
                self.add_positions_to_metadf()
                # self.write_header()

                for curr_attr in self.key_attrs: # + ['noattr']:
                    self.curr_attr = curr_attr
                    self.vizpath = self.get_file_path(f'{self.mxname}_{self.curr_attr}', subdir=True)
                    self.get_results_path(self.mxname, self.curr_attr) # recreate the path in case there is more than one key_attr
                    self.get_figure()
                    self.fill_subplots(self.curr_attr)








class MdsResults(DataHandler):
    def __init__(self, language, output_dir='analysis_s2v', by_author=False):
        super().__init__(language, output_dir=output_dir)
        self.by_author = by_author
        self.add_subdir('MxSingleViz2D3DHzAnalysisSelect')
        self.results_path = os.path.join(self.subdir, 'results.txt')
        self.idcols = ['mxname', 'curr_attr', 'dim'] # columns that 
        self.df = self.load_data()
        print(self.df)
        # self.get_nocolor_metadf()

    # only needed if metadata is not already added during selection
    # def get_nocolor_metadf(self):
    #     self.ih = InfoHandler(language=self.language, add_color=False, cmode='mx', by_author=self.by_author)
    #     self.metadf = deepcopy(self.ih.metadf)
    #     print(self.metadf)


    def load_data(self):
        df = pd.read_csv(self.results_path, header=0, index_col=None, sep=',')
        print('df original', df.shape)

        # Split the df into a df with the labels and one with the remaining mxnames and comments
        df_comment = df[df['label'].isna()]
        print("DataFrame with NaN in 'label' column:")
        print(df_comment.shape)
        df_comment = df_comment[['mxname', 'comment']] # keep only one column
        df_comment = df_comment.dropna(subset=['comment']) # Drop rows where comment column is empty
        assert not df_comment['mxname'].duplicated().any() # Assert that there is only one comment per mxname

        df = df[~df['label'].isna()]
        print("\nDataFrame without NaN in 'label' column:")
        print(df.shape)
        # Drop duplicated rows, which means that a data point was clicked multiple times
        df = df.drop_duplicates(subset=self.idcols + ['label'])
        assert df['comment'].isna().all()
        df = df.drop(columns=['comment']) # drop empty column

        # Merge comment df and label df so that each row with a label has the comment
        nrows_before_merge = df.shape[0]
        df = df.merge(df_comment, on='mxname', validate='m:1', how='outer')
        assert df.shape[0] == nrows_before_merge

        # df = df.merge(self.metadf, left_on='mxname', right_index=True, validate='1:1')
        # assert df.shape[0] == nrows_before_merge
        return df


    # def get_networks(self):
    #     grouped = self.df.groupby(self.idcols)
    #     for group_key, group_df in grouped:
    #         # Extract the list of values in the 'label' column for the current group
    #         labels_list = group_df['label'].tolist()
    #         parts = group_df.loc[0, 'mxname'].split('_', maxsplit=2)
    #         mxname, spars = parts[:2]
    #         spmx_path = os.path.join(self.output_dir, 'similarity', self.language, 'sparsification', f'sparsmx-{mxname}_{spars}.pkl')
    #         print(mxname, spars, spmx_path)
    #         info = CombinationInfo(mxname=mxname, sparsmode=spars, spmx_path=spmx_path)

# m = MdsResults('eng', by_author=True)
# m.get_networks()# %%

# %%
