# %%
from copy import deepcopy
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import mplcursors
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np

from .nkviz import NkSingleVizAttr, NkSingleViz
from .mxviz import MxSingleViz2D3DHorizontal
from cluster.combinations import InfoHandler
from cluster.cluster_utils import CombinationInfo
from cluster.network import NXNetwork



class NkSingleVizAttrAnalysis(NkSingleVizAttr):
    def __init__(self, language, output_dir, info, exp, by_author, labels, cmode, subdir):
        super().__init__(language, output_dir, info=info, plttitle=None, exp=exp, by_author=by_author)
        self.labels = labels
        self.cmode = cmode
        self.markersize = 20
        self.subdir = subdir
        print('created NkSingleVizAttrAnalysis')

    def get_metadf(self):
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
        print('vizpath', self.vizpath)
        if not os.path.exists(self.vizpath):
            self.get_graphs()
            self.get_positions()
            self.add_positions_to_metadf()

            self.get_figure()
            self.adjust_subplots()
            self.add_edges()

            self.fill_subplots()
            self.save_plot(plt)
            # Not interesting when in network mode
            # if self.cmode == 'mx':
            plt.show()
            # plt.close()
            print('generated network')


class MxSingleViz2D3DHzAnalysis(MxSingleViz2D3DHorizontal):
    def __init__(self, language, output_dir, exp, by_author, mc, info=None):
        super().__init__(language, output_dir, exp, by_author, mc, info=None)
        self.add_subdir('MxNkAnalysis')
        self.markersize = 20
        self.get_metadf()
        self.key_attrs = ['canon']
        ih = InfoHandler(language=self.language, add_color=False, cmode='mx', by_author=self.by_author)
        self.df_nocolor = deepcopy(ih.metadf) # Df with all nodes, self.df has only nodes with positions (no iso nodes)


    def get_metadf(self):
        self.ih = InfoHandler(language=self.language, add_color=True, cmode='mx', by_author=self.by_author)
        self.df = deepcopy(self.ih.metadf)


    def get_networks(self):
        parts = self.mxname.split('_', maxsplit=2)
        distname, spars = parts[:2]
        spmx_path = os.path.join(self.data_dir, 'similarity', self.language, 'sparsification', f'sparsmx-{distname}_{spars}.pkl')
        df_nk = self.get_metadf_for_labels()
        info = CombinationInfo(mxname=self.mxname, sparsmode=spars, spmx_path=spmx_path, attr=self.curr_attr, metadf=df_nk)
        nk = NkSingleVizAttrAnalysis(language=self.language, output_dir=self.output_dir, info=info, exp=self.exp, by_author=self.by_author, labels=self.labels, cmode=self.cmode, subdir=self.subdir)
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
            for item in sublist:
                color = df_nk.loc[item, 'label_color']

        self.df = self.map_colors_to_nested_list(self.df)

        for sublist in self.labels:
            for item in sublist:
                color = self.df.loc[item, 'label_color']
        return df_nk

            
    def draw_mds(self, ix, color_col=None, use_different_shapes=False, s=30, edgecolor='black', linewidth=0.2):
        self.labels = []
        self.labels2d = []
        self.labels3d = []
        self.class_counter = 0
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

        scatter_kwargs = {'s': s, 'edgecolor': edgecolor, 'linewidth': linewidth}
        color_col = f'{color_col}_color'
        
        sdf = self.df.copy() # Avoid chained assingment warning
        kwargs = {'c': sdf[color_col], 'marker': 'o', **scatter_kwargs}
        
        # 2D plot
        scatter_2d = self.axs[ix[0], ix[1]].scatter(x=sdf['X_mds_2d_0'], y=sdf['X_mds_2d_1'], **kwargs)
        # cursor_2d = mplcursors.cursor(scatter_2d, hover=False)
        
        # @cursor_2d.connect('add')
        # def on_add_2d(sel):
        #     i = sel.index
        #     label = sdf.index[i]
        #     if label not in self.labels2d:
        #         self.labels2d.append(label)
        #         row = self.df_nocolor.loc[label].values.tolist() 
        #         self.write_labels_to_file(label, row, dim='2d')
        #     sel.annotation.set(text=label, fontsize=8, ha='right', color=sdf[color_col].iloc[i])


        # Lasso selector for selecting multiple points
        def on_select_2d(verts):
            path = Path(verts)
            ind = np.nonzero(path.contains_points(sdf[['X_mds_2d_0', 'X_mds_2d_1']]))[0]
            for i in ind:
                label = sdf.index[i]
                if label not in self.labels2d:
                    self.labels2d.append(label)
                    row = self.df_nocolor.loc[label].values.tolist()
                    self.write_labels_to_file(label, row, dim='2d')
                    print(f'Selected: {label}')

        print('Lasso selector for 2D plots. Make sure line is closed!')
        lasso_2d = LassoSelector(self.axs[ix[0], ix[1]], on_select_2d)

        
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
                self.write_labels_to_file(label, row, dim='3d')
            sel.annotation.set(text=label, fontsize=8, ha='center', color=sdf[color_col].iloc[i])



        # Button to trigger network generation
        ax_button = plt.axes([0.81, 0.01, 0.1, 0.075])  # position: [left, bottom, width, height]
        button = Button(ax_button, 'Generate Network')
        button.on_clicked(self.on_button_click)

        ax_class_button = plt.axes([0.81, 0.09, 0.1, 0.075])
        button_class = Button(ax_class_button, 'New')
        button_class.on_clicked(self.on_class_button_click)

        ax_reset_button = plt.axes([0, 0.01, 0.02, 0.03])
        button_reset = Button(ax_reset_button, 'Reset')
        button_reset.on_clicked(self.on_reset_button_click)

        # create a non-maximized window with the size of a maximized one
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


        # Pop up window can be closed with ctrl+Q+Q
        plt.gcf().canvas.mpl_connect('close_event', self.on_close)

        self.fig.suptitle(self.mxname)
        plt.show()


    def on_button_click(self, event):
        self.labels.append(list(set(self.labels2d + self.labels3d))) # combine lists
        print('clicked button network')
        self.get_networks()

    def on_class_button_click(self, event):
        self.labels.append(list(set(self.labels2d + self.labels3d)))
        self.labels2d = []
        self.labels3d = []
        self.class_counter += 1

    def on_reset_button_click(self, event):
        self.labels = []
        self.labels2d = []
        self.labels3d = []
        os.remove(self.results_path)


    def write_labels_to_file(self, label, row, dim):
        with open(self.results_path, 'a') as f:
            # add class counter to keep track of which points is in which cluster
            f.write(f"{self.mxname},,{self.curr_attr},{dim},{self.class_counter},{label},{','.join(map(str, row))}\n") # extra comma for comment column

    def on_close(self, event):
        # If points were selected, write a comment
        if self.labels and self.labels != [[]]:
            comment = input('Enter a comment. Must not contain any commas! ') # commas used as sep
            # self.get_networks()
            self.write_comment_to_file(comment)


    def write_comment_to_file(self, comment=','):
        comment = comment.replace(',', '.') # replace commas because they are used as seperators in the csv
        with open(self.results_path, 'a') as f:
            ncols = self.df_nocolor.shape[1]
            commas = (ncols+3) * ','
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
            print('mxname', self.mxname)
            print('mx dimensions', self.mx.mx.shape)
            # Check if results for last key attr has been created
            # This is faster than adding the positions and then checking the paths
            self.get_results_path(self.mxname, self.key_attrs[-1])
            if os.path.exists(self.results_path):
                print('results already exist', self.results_path)
            else:
                self.pos = self.get_mds_positions()
                self.add_positions_to_metadf()
                # self.write_header()

                for curr_attr in self.key_attrs: # + ['noattr']:
                    self.curr_attr = curr_attr
                    self.vizpath = self.get_file_path(f'{self.mxname}_{self.curr_attr}', subdir=True)
                    self.get_results_path(self.mxname, self.curr_attr) # recreate the path in case there is more than one key_attr
                    self.get_figure()
                    self.fill_subplots(self.curr_attr)





class NkSingleVizAnalysis(NkSingleViz):
    '''
    Copied Code from MxSingleViz2D3DHzAnalysis because multiple inheritance does not work
    '''

    def __init__(self, language, output_dir, exp, by_author, mc):
        NkSingleViz.__init__(self, language=language, output_dir=output_dir, exp=exp, by_author=by_author)
        self.add_subdir('NkAnalysis')
        self.markersize = 70
        self.get_metadf()
        self.key_attrs = ['canon']
        ih = InfoHandler(language=self.language, add_color=False, cmode='nk', by_author=self.by_author)
        self.df_nocolor = deepcopy(ih.metadf) # Df with all nodes, self.df has only nodes with positions (no iso nodes)


    def get_metadf(self):
        ih = InfoHandler(language=self.language, add_color=True, cmode='nk', by_author=self.by_author)
        self.df = deepcopy(ih.metadf)
        

    def visualize(self, vizname='viz'): # vizname for compatibility
        mxs = self.load_mxnames()
        mxs = sorted(mxs)
        for mxname in mxs:
            mxpath = os.path.join(self.mxdir, mxname)
            self.mxname = self.clear_mxname(mxname) # contains mxname_sparsmode
            print('mxname', self.mxname)
            # Check if plot for last key attr has been created
            self.get_results_path(self.mxname, self.key_attrs[-1])
            if os.path.exists(self.results_path):
                print('results already exist')
            else:
                self.get_metadf()
                mxname_only, sparsmode = mxname.split('_')
                self.info = CombinationInfo(mxname=self.mxname, sparsmode=sparsmode)

                i = 0
                j = 0
                self.get_figure()
                
                self.network = NXNetwork(self.language, path=mxpath)
                self.graph = self.network.graph

                if (self.graph.number_of_edges() > 0):
                    self.global_vmax, self.global_vmin = self.get_cmap_params()
                    self.get_graphs()
                    self.get_positions()
                    self.add_positions_to_metadf()
                    self.subplots = [[i, j]]
                    self.add_edges()
                    for curr_attr in self.key_attrs:
                        self.info.attr = curr_attr
                        self.curr_attr = curr_attr
                        self.fill_subplots()
                plt.close()


    def fill_subplots(self):
        self.labels = []
        self.labels2d = []
        self.class_counter = 0
        self.get_interactive_plots()

        if self.labels == [[]]:
            assert 'label_color' not in self.df.columns
            assert self.labels == [[]]
            print('No labels were selected. No plots were saved.')
            # with open(self.results_path, 'w') as f: # Create empty file as reminder that combination has been checked ################################
            #     f.write('')

        else:
            self.get_networks()

            # If points were selected, write a comment
            comment = input('Enter a comment. Must not contain any commas! ') # commas used as sep
            # self.get_networks()
            # self.write_comment_to_file(comment) ###############################
            # print('wrote comment to file')



    def get_interactive_plots(self):
        self.add_nodes_to_ax([0,0], self.df, color_col=self.info.attr, use_different_shapes=False)
        node_positions = np.array([self.pos[node] for node in self.graph.nodes])
        node_indices = list(self.graph.nodes)

        self.labels = []
        self.labels2d = []

        # Lasso selector callback
        def onselect(verts):
            path = Path(verts)
            ind = np.nonzero(path.contains_points(node_positions))[0]
            for i in ind:
                label = node_indices[i]
                if label not in self.labels2d:
                    self.labels2d.append(label)
                    row = self.df_nocolor.loc[label].values.tolist()
                    # self.write_labels_to_file(label, row, dim='2d') ########################33
                    print(f'Selected: {label}')
        lasso = LassoSelector(self.axs[0, 0], onselect)


        # Button to reset lables
        ax_class_button = plt.axes([0.81, 0.09, 0.1, 0.075])
        button_class = Button(ax_class_button, 'New')
        button_class.on_clicked(self.on_class_button_click)

        ax_reset_button = plt.axes([0, 0.01, 0.02, 0.03])
        button_reset = Button(ax_reset_button, 'Reset')
        button_reset.on_clicked(self.on_reset_button_click)

        # create a non-maximized window with the size of a maximized one
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


        # Pop up window can be closed with ctrl+Q+Q
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.gcf().canvas.mpl_connect('close_event', self.on_close)

        self.fig.suptitle(self.mxname)
        plt.show()


    def on_key_press(self, event):
        # Check if the 'n' key is pressed
        if event.key == 'n':
            self.on_class_button_click(event)


    def get_metadf(self):
        self.ih = InfoHandler(language=self.language, add_color=True, cmode='mx', by_author=self.by_author)
        self.df = deepcopy(self.ih.metadf)


    def get_networks(self):
        parts = self.mxname.split('_', maxsplit=2)
        distname, spars = parts[:2]
        spmx_path = os.path.join(self.data_dir, 'similarity', self.language, 'sparsification', f'sparsmx-{distname}_{spars}.pkl')
        df_nk = self.get_metadf_for_labels()
        info = CombinationInfo(mxname=self.mxname, sparsmode=spars, spmx_path=spmx_path, attr=self.curr_attr, metadf=df_nk)
        nk = NkSingleVizAttrAnalysis(language=self.language, output_dir=self.output_dir, info=info, exp=self.exp, by_author=self.by_author, labels=self.labels, cmode=self.cmode, subdir=self.subdir)
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
            for item in sublist:
                color = df_nk.loc[item, 'label_color']

        self.df = self.map_colors_to_nested_list(self.df)

        for sublist in self.labels:
            for item in sublist:
                color = self.df.loc[item, 'label_color']
        return df_nk

    
    def on_class_button_click(self, event):
        self.labels.append(list(set(self.labels2d)))
        self.labels2d = []
        self.class_counter += 1

    def on_reset_button_click(self, event):
        self.labels = []
        self.labels2d = []
        os.remove(self.results_path)


    def write_labels_to_file(self, label, row, dim):
        with open(self.results_path, 'a') as f:
            # add class counter to keep track of which points is in which cluster
            f.write(f"{self.mxname},,{self.curr_attr},{dim},{self.class_counter},{label},{','.join(map(str, row))}\n") # extra comma for comment column

    def on_close(self, event):
        # Automatically generate network without pressing button
        self.labels.append(list(set(self.labels2d))) # combine lists


    def write_comment_to_file(self, comment=','):
        comment = comment.replace(',', '.') # replace commas because they are used as seperators in the csv
        with open(self.results_path, 'a') as f:
            ncols = self.df_nocolor.shape[1]
            commas = (ncols+3) * ','
            f.write(f'{self.mxname},{comment}{commas}\n')

    def write_header(self):
        if not os.path.exists(self.results_path):
            with open(self.results_path, 'w') as f:
                f.write(f"mxname,comment,curr_attr,dim,label,{','.join(self.df_nocolor.columns)}\n")

    def get_results_path(self, mxname, curr_attr):
        self.results_path = os.path.join(self.subdir, f'{mxname}_{curr_attr}_results.txt')
