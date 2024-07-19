
import sys
sys.path.append('..')
from utils import DataHandler
import os
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import pickle
import gc
import pandas as pd
import random
random.seed(99)

from .embedding_utils import EmbeddingBase
from .viz_utils import ImageGrid
from .nkviz import NkVizBase
from .mxviz import MxSingleViz
from cluster.combinations import MxCombinations
from cluster.combinations import InfoHandler
from cluster.create import D2vDist
from cluster.cluster_utils import MetadataHandler, CombinationInfo
from .s2vcreator import S2vCreator
from copy import deepcopy


# use network env

class EmbDist(D2vDist):
    def __init__(self, language, output_dir, modes, map_nodeids_to_names=True):
        super().__init__(language, output_dir=output_dir, modes=modes)
        self.file_string = output_dir
        self.map_nodeids_to_names = map_nodeids_to_names
        self.eb = EmbeddingBase(self.language, output_dir=output_dir)


    def get_embeddings_dict(self, mode):
        '''
        Load embeddings of a single network (mode) in format {node_name: embedding}
        '''
        emb_dict = {}
        df = self.eb.load_embeddings(mode, map_nodeids_to_names=self.map_nodeids_to_names)
        for file_name, row in df.iterrows():
            emb_dict[file_name] = np.array(row)
        return emb_dict
    

class EmbLoader():
    def __init__(self, language, file_string, mode=None, files_substring=None, ec=None, by_author=False, map_nodeids_to_names=True):
        self.language = language
        self.file_string = file_string
        self.mode = mode
        self.ec = ec # Embedding creator class, use S2vCreator by default or pass object of custom class
        self.by_author = by_author
        self.map_nodeids_to_names = map_nodeids_to_names
        self.files_substring = files_substring ####### useless? delete
        self.filter_embedding_files()


    def load_single_mx(self, mxname):
        emdist = EmbDist(language=self.language, output_dir=self.file_string, modes=self.embedding_files, map_nodeids_to_names=self.map_nodeids_to_names)
        mx = emdist.load_data(load=True, mode=mxname, use_kwargs_for_fn='mode', file_string=self.file_string, subdir=True)
        return mx


    def filter_embedding_files(self):
        '''
        Get all embedding files that belong to either the 'run' or the 'params' mode EdgelistHandler, or that are passed as a list of substrings.
        '''
        # if self.file_string == 'n2v':
        #     self.ec = N2vCreator(self.language, mode=self.mode)
        # else:
        if self.ec is None:
            self.ec = S2vCreator(self.language, mode=self.mode, by_author=self.by_author)

        all_embedding_paths = self.ec.get_all_embedding_paths()
        all_embedding_paths = [os.path.basename(x) for x in all_embedding_paths]
        all_embedding_paths = [x.split('.')[0] for x in all_embedding_paths]
        # all_embedding_paths = ['cosinesim-1000_threshold-0%95_dimensions-32_walklength-3_numwalks-20_windowsize-3'] ###############################

        # # Filter list of embedding files
        # embedding_files = [file.split('.')[0] for file in os.listdir(self.ec.subdir) if file.endswith('embeddings')] # remove '.embeddings'
        # if self.mode is not None:
        #     embedding_files = [file for file in embedding_files if any(sub in file for sub in all_embedding_paths)]
        # print('len ebm files', len(embedding_files))

        # print('len ebm files', len(embedding_files))
        # self.embedding_files = embedding_files

        if self.files_substring is not None:
            all_embedding_paths = [x for x in all_embedding_paths if self.files_substring in x]

        self.embedding_files = sorted(all_embedding_paths)


    def load_mxs(self):
        # Load similarity matrices. If they don't exist, they are created
        emdist = EmbDist(language=self.language, output_dir=self.file_string, modes=self.embedding_files, map_nodeids_to_names=self.map_nodeids_to_names)


        total_iterations = len(self.embedding_files)
        for i, embedding_file in enumerate(self.embedding_files):
            mx = emdist.load_data(load=True, mode=embedding_file, use_kwargs_for_fn='mode', file_string=self.file_string, subdir=True)
            iteration_number = i + 1
            mx.name = embedding_file
            if self.mode == 'run' and 'mirror' in embedding_file:
                print('"mirror" in wrong directory.')

            # Print every 100th file
            if iteration_number % 100 == 0 or iteration_number == total_iterations:
                # Print the current iteration number and the total number of iterations
                print(f"Current embedding file: {iteration_number} out of {total_iterations}")
            yield mx



class EmbParamEvalSingleMDS(MxSingleViz):
    def __init__(self, language, output_dir, exp=None, by_author=False, mc=None, df=None, attr=None, mx=None):
        super().__init__(language, output_dir, exp, by_author, mc)
        self.df = df
        self.attr = attr
        self.mx = mx
        self.markersize = 5

    def visualize(self, vizname='mds', plt_kwargs={'dpi': 100}): # vizname for compatibility
        self.vizpath = self.get_file_path(vizname, subdir=True)
        if not os.path.exists(self.vizpath):
            self.pos = self.get_mds_positions()
            self.add_positions_to_metadf(load_metadf=False)           
            self.get_figure()
            self.fill_subplots(self.attr)
            self.save_plot(plt, plt_kwargs=plt_kwargs)
            plt.close()
               

class EmbParamEvalSingleViz(NkVizBase):
    '''
    Visualize each parameter combination in a seperate plot, ignore isolated nodes
    '''
    def __init__(self, language, output_dir, info=None, plttitle=None, exp=None, by_author=False, graph=None, ignore_nodes_removed=True): ###############True
        super().__init__(language=language, output_dir=output_dir, info=info, plttitle=plttitle, exp=exp, by_author=by_author, graph=graph)
        self.ignore_nodes_removed = ignore_nodes_removed 
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.01
        self.ws_hspace = 0.01
        self.markersize = 7
        # Infohandler to load metadf
        self.ih = InfoHandler(language=self.language, add_color=True, cmode=self.cmode, by_author=self.by_author)

    def get_metadf(self):
        # InfoHanlder metadf contains all texts, graph only contains non-isolated nodes used for embeddings
        nodes = set(self.graph.nodes)
        self.ih.metadf = self.ih.metadf[self.ih.metadf.index.isin(nodes)]
        self.df = deepcopy(self.ih.metadf)

    def get_figure(self):
        self.fig, self.axs = plt.subplots(figsize=(5, 5))
        self.axs = np.reshape(self.axs, (1, 1))
        self.axs[0, 0].axis('off')

    def add_custom_subdir(self):
        # self.sc = S2vCreator(self.language, mode='params')
        # self.output_dir=self.sc.output_dir
        self.add_subdir('singleimage')

    def get_graphs(self):
        if self.ignore_nodes_removed:
            # nx.connected_components is only implemented for undirected graphs
            if nx.is_directed(self.graph):
                graph = self.graph.to_undirected()
            else:
                graph = deepcopy(self.graph)
            self.graph_con = self.graph
            self.nodes_removed = []
        else:
            super().get_graphs()

    def add_edges(self):
        # Main plot
        self.draw_edges(self.graph_con, self.pos, [0,0])

    def fill_subplots(self):
        # attr
        self.add_nodes_to_ax([0,0], self.df, color_col=self.info.attr, use_different_shapes=False)

    def add_nodes_to_ax(self, ix, df, color_col, use_different_shapes=False):
        color_col = f'{color_col}_color'
        # Draw connected components with more than 2 nodes
        df_con = df[~df.index.isin(self.nodes_removed)]
        ax = self.get_ax(ix)
        self.draw_nodes(self.graph_con, ax, df_con, color_col, use_different_shapes=use_different_shapes)

    def get_path(self, name, omit=None):
        return os.path.join(self.subdir, f'{name}.{self.data_type}')
    
    def visualize_edges(self, ):
        # Split super().visualize method into two parts to draw edges and nodes seperately
        # if not self.too_many_edges:
        self.get_graphs()
        self.get_positions()
        self.add_positions_to_metadf()

        start = time.time()
        self.logger.debug(f'Nr edges below cutoff for {self.info.as_string()}. Making visualization.')

        self.get_figure()
        self.adjust_subplots()
        self.add_edges()
        calctime = time.time()-start
        if calctime > 10:
            print(f'{calctime}s to visualize.')
        print(f'{calctime}s to draw edges.')

    def visualize_nodes(self, vizname='viz', omit=[], plt_kwargs={'dpi': 100}):
        self.vizpath = self.get_path(name=vizname, omit=omit)
        print(self.vizpath)
        if not os.path.exists(self.vizpath):
            s = time.time()
            self.fill_subplots()
            self.save_plot(plt, plt_kwargs=plt_kwargs)
            # Showing plot does not work when nodes are visualized multiple times without redrawing the edges
            print(f'{time.time()-s}s to draw nodes')



class EmbParamEvalGrid(ImageGrid):
    '''
    One row for each param value of numwalks.
    One column for each param value of walklength.
    Different plot for each dimension, windowsize, untillayer.
    Fill images in column major order to align the number of rows and cols with the number of values for walk-length and num-walks.
    '''
    def __init__(self, language, combs, imgdir='singleimage', subdir_name='gridimage', by_author=False):
        self.combs = combs
        self.subdir_name = subdir_name
        super().__init__(language, attr=None, by_author=by_author, output_dir='s2v', imgdir=imgdir, rowmajor=False)
        self.ncol = 6 # 'walk-length': [3, 5, 8, 15, 30]s
        self.nrow = 3 # 'num-walks': [20, 50, 200]
        self.imgs = self.load_single_images()
        self.fontsize = 6

        self.img_width = 1.7*2 # format of single-network plots stored on file, times 2 to make fig bigger
        self.img_height = 1.7*2
        self.nr_mxs = 58
        self.nr_spars = 9

        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.01
        self.ws_hspace = 0.01

        self.add_subdir(self.subdir_name)


    def load_single_images(self):
        imgs = []
        for comb in self.combs:
            path =  os.path.join(self.imgdir, f'{comb}.png')
            imgs.append(path)
        return imgs
    

    def get_file_path(self, vizname, subdir=None, **kwargs):
        file_name = f"{vizname}_dimensions-{kwargs['dimensions']}_windowsize-{kwargs['windowsize']}_untillayer-{kwargs['untillayer']}.png"
        return os.path.join(self.subdir, file_name)
    

    def get_title(self, imgname):
        img = os.path.basename(imgname)
        mxname, spars, dim, walklength, numwalks, windowsize, untillayer, opt1, opt2, opt3 = img.split('_')
        title = f'{walklength}, {numwalks}' #, {windowsize}'
        return title
    


class MirrorMDSGrid(ImageGrid):
    '''
    Adapted from EmbParamEvalGrid
    Combine MDS plots of mirrored graphs into grid. One grid for each distance-sparsification combination for which mirror graphs were run.
    '''
    def __init__(self, language, imgdir='mx_mirror_singleimage', subdir_name='mx_mirror_gridimage', by_author=False):
        self.subdir_name = subdir_name
        super().__init__(language, attr=None, by_author=by_author, output_dir='s2v', imgdir=imgdir, rowmajor=True)
        self.ncol = 4
        self.nrow = 2
        self.imgs = self.load_single_images()
        self.fontsize = 10

        self.img_width = 5
        self.img_height = 5
        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.01
        self.ws_hspace = 0.01

        self.add_subdir(self.subdir_name)


    def visualize_all(self):
        mxnames = []
        for img in os.listdir(self.imgdir):
            components = img.split('_')[:2]
            mxname = '_'.join(components)
            mxnames.append(mxname)

        mxnames = list(set(mxnames))
        for i in mxnames:
            print(i)

        
        for mxname in mxnames:
            imgs = []
            for img in os.listdir(self.imgdir):
                if mxname in img:
                    imgs.append(img)
            assert len(imgs) == 8 # 8 different combinations of run mode params
            imgs = sorted(imgs)
            super().visualize(vizname=f'{mxname}', imgs=imgs) 


    def get_title(self, imgname):
        img = os.path.basename(imgname)
        mxname, spars, dim, walklength, numwalks, windowsize, untillayer, opt1, opt2, opt3 = img.split('_')
        title = f'{dim}, {walklength}, {windowsize}'
        return title


class CombineParamEvalGrids(ImageGrid):
    '''
    Combine MDS and network parameter grids into one plot.
    Cannot be done with network visualizations because then the nodes are too small to see anything.
    '''
    def __init__(self, language, imgdir='mx_gridimage', subdir='mx_gridimages_combined', by_author=False):
        super().__init__(language, attr=None, by_author=by_author, output_dir='s2v', imgdir=imgdir, rowmajor=True)
        self.ncol = 2
        self.nrow = 2 # 3 for first three dims
        # self.imgs = self.load_single_images() ##################
        self.fontsize = 5

        self.img_width = 10
        self.img_height = 5

        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.01
        self.ws_hspace = 0.01

        self.add_subdir(subdir)
        self.img_name = None


    def visualize_all(self):
        for img in os.listdir(self.imgdir):
            imgs = []
            print('img', img)
            if 'dimensions-32' in img and 'untillayer-5' in img:
                mxname, spars, dim, windowsize, untillayer = img.split('_')
                # untillayer = untillayer.split('.')[0] # remove 'png' extension
                print(mxname, spars, dim, windowsize, untillayer)
                img10 = img.replace('untillayer-5', 'untillayer-10')
                # imgs.append(img) # if uncommented, change to self.nrow = 3
                # imgs.append(img10)
                # imgs.append(img.replace('dimensions-32', 'dimensions-64'))
                # imgs.append(img10.replace('dimensions-32', 'dimensions-64'))
                # imgs.append(img.replace('dimensions-32', 'dimensions-128'))
                # imgs.append(img10.replace('dimensions-32', 'dimensions-128'))
                imgs.append(img.replace('dimensions-32', 'dimensions-3'))
                imgs.append(img10.replace('dimensions-32', 'dimensions-3'))
                imgs.append(img.replace('dimensions-32', 'dimensions-5'))
                imgs.append(img10.replace('dimensions-32', 'dimensions-5'))
                super().visualize(vizname=f'{mxname}_{spars}_dimensions-3-5', imgs=imgs, windowsize=windowsize.split('-')[1]) 
                imgs = []
                imgs.append(img.replace('dimensions-32', 'dimensions-8'))
                imgs.append(img10.replace('dimensions-32', 'dimensions-8'))
                imgs.append(img.replace('dimensions-32', 'dimensions-16'))
                imgs.append(img10.replace('dimensions-32', 'dimensions-16'))
                super().visualize(vizname=f'{mxname}_{spars}_dimensions-8-16', imgs=imgs, windowsize=windowsize.split('-')[1]) 
                


    # def load_single_images(self):
    #     imgs = []
    #     path =  os.path.join(self.imgdir, self.img_name)
    #     imgs.append(path)
    #     path = path.replace('singleimage', 'mx_singleimage')
    #     imgs.append(path)
    #     print('nr images for current grid: ', len(imgs))
    #     return imgs
    

    def get_file_path(self, vizname, subdir=None, **kwargs):
        file_name = f"{vizname}_windowsize-{kwargs['windowsize']}.png"
        return os.path.join(self.subdir, file_name)
    

    def get_title(self, imgname):
        img = os.path.basename(imgname)
        mxname, spars, dim, windowsize, untillayer= img.split('_')
        title = dim
        title = imgname ####################
        return title






class ParamModeEval(S2vCreator):
    '''
    Evaluate different parameter settings for s2v by highlighting nodes in network according to their similarity to a selected node in a prominent position.
    '''
    def __init__(self, language, by_author=False):
        super().__init__(language=language, mode='params', by_author=by_author)


    def create_single_images(self):
        all_emb_paths = self.get_all_embedding_paths()
        all_mxnames = [os.path.basename(x) for x in all_emb_paths]
        all_mxnames = [x.split('.')[0] for x in all_mxnames]
        self.el = EmbLoader(self.language, file_string=self.file_string, mode='params', by_author=self.by_author)

        nkviz_dir = os.path.join(self.output_dir, 'singleimage')
        mdsviz_dir = os.path.join(self.output_dir, 'mx_singleimage')


        for network_name, node in self.examples.items():
            print('network name', network_name)

            # Network is the same for all params
            # Draw network edges only once, repeatedly draw nodes
            network = self.network_from_edgelist(os.path.join(self.edgelist_dir, f'{network_name}.csv'), delimiter=' ', nodes_as_str=True, print_info=False)
            info = CombinationInfo(attr='canon') # info with random attr to init class
            nkviz = EmbParamEvalSingleViz(language=self.language, output_dir=self.output_dir, info=info, exp={'attr': node}, by_author=self.by_author, graph=network)
            nkviz.visualize_edges()
    
    
            mxnames = [x for x in all_mxnames if network_name in x]
            for cmxname in mxnames:
                # Checking paths with class instances is too slow. Instead, create paths directly.
                nkvizpath = os.path.join(nkviz_dir, f'{cmxname}.png')
                mdsvizpath = os.path.join(mdsviz_dir, f'{cmxname}.png')



                # Only create simmxs if necessary
                if not os.path.exists(nkvizpath) or not os.path.exists(mdsvizpath):
                    print('nkvizpath', nkvizpath)
                    print('mdsvizpath', mdsvizpath)
                    print(os.path.exists(nkvizpath), os.path.exists(mdsvizpath))
                    cmx = self.el.load_single_mx(mxname=cmxname)
                    cmx.name = cmxname


                    # Select all col with all distances to 'node'
                    df = cmx.mx.loc[[node]]
                    # Exclude the entry for 'node' from the df so that scaling for color is not affected by similarity=1
                    row_without_node = df.drop(columns=[node])
                    df = row_without_node.transpose()
                    assert len(df) == len(cmx.mx) - 1


                    mh = MetadataHandler(self.language)
                    df = mh.add_color_column(metadf=df, colname=node, use_skewed_colormap=False)
                    df.loc[node, f'{node}_color'] = 'lime'
                    assert len(df) == len(cmx.mx)
                    

                    if not os.path.exists(nkvizpath):
                        # Add nodes to network with colors for current parameter combination
                        info = CombinationInfo(metadf=df, attr=node)
                        nkviz.df = df
                        nkviz.info = info
                        nkviz.visualize_nodes(vizname=cmxname)


                    if not os.path.exists(mdsvizpath):
                        # MDS of similarity matrix
                        mdsviz = EmbParamEvalSingleMDS(language=self.language, output_dir=self.output_dir, exp={'name': 'singleimage'}, by_author=self.by_author, mc=self.el, df=df, attr=node, mx=cmx)
                        mdsviz.visualize(vizname=cmxname)


    def create_grid_images(self):
        for network_name, node in self.examples.items():
            params = self.get_params()
            param_combs =  super().get_param_combinations()


            for dim in params['dimensions']:
                for ws in params['window-size']:
                    for ul in params['until-layer']:
                        combs = [
                            d for d in param_combs
                            if d['dimensions'] == dim and d['window-size'] == ws and d['until-layer'] == ul
                        ]
                        combs = [self.get_param_string(x) for x  in combs]
                        combs = [f'{network_name}_{x}' for x in combs]
                        ignk = EmbParamEvalGrid(self.language, combs, imgdir='singleimage', subdir_name='gridimage', by_author=self.by_author)
                        ignk.visualize(vizname=network_name, dimensions=dim, windowsize=ws, untillayer=ul)
                        del ignk
                        gc.collect()  # Explicitly call garbage collection
                        igmx = EmbParamEvalGrid(self.language, combs, imgdir='mx_singleimage', subdir_name='mx_gridimage', by_author=self.by_author)
                        igmx.visualize(vizname=network_name, dimensions=dim, windowsize=ws, untillayer=ul)
                        del igmx
                        gc.collect()

   

class RunModeParamGridPerAttr(ImageGrid):
    '''
    Place all MDS plots for the same matrix with different s2v paramn next to each other.
    '''
    def __init__(self, language, combs, attr):
        self.combs = combs
        super().__init__(language, attr=attr, by_author=False, output_dir='analysis_s2v', imgdir='mx_singleimage')
        self.nrow = 2
        self.ncol = 2
        assert len(self.combs) == self.nrow * self.ncol
        self.imgs = self.load_single_images()
        self.fontsize = 6

        self.img_width = 6
        self.img_height = 4
        # self.nr_interesting_networks = 304
        # self.nr_param_combs = 6 # 2 values for walk-length, 3 values for window-size

        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.01
        self.ws_hspace = 0.01

        self.add_subdir('run_mode_param_grid_per_attr')


    def load_single_images(self):
        imgs = []
        for comb in self.combs:
            path =  os.path.join(self.imgdir, f'{comb}_{self.attr}.png')
            imgs.append(path)
        return imgs
    

    def get_file_path(self, vizname, subdir=None, **kwargs):
        file_name = f"{vizname}_{self.attr}.png"
        return os.path.join(self.subdir, file_name)
    

    def get_title(self, imgname):
        x = os.path.basename(imgname)
        x = x.split('.')[0]
        mxname, spars, dim, walklength, numwalks, windowsize, untillayer, opt1, opt2, opt3, attr = x.split('_')
        title = f'{walklength}, \n{windowsize}'
        return title
    



class BestparamsAttrGrid(ImageGrid):
    '''
    Place all MDS plots for the same matrix with different s2v params on top of each other, and different attributes next to each other.
    '''
    def __init__(self, language, combs):
        self.combs = combs
        super().__init__(language, by_author=False, output_dir='analysis_s2v', imgdir='mx_singleimage')
        self.nrow = 2
        self.ncol = 2
        self.imgs = self.load_single_images()
        self.fontsize = 10

        self.img_width = 4
        self.img_height = 4

        # self.nr_interesting_networks = 304
        # self.nr_param_combs = 4 # 2 values for walk-length, 2 values for window-size

        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.1
        self.ws_hspace = 0.1

        self.add_subdir('bestparams-attr-grid')

    def get_figure(self):
        super().get_figure()
        
        # Add grid to each subplot
        for ax in self.axs.flat:
            ax.grid(True)


    def load_single_images(self):
        imgs = []
        for comb in self.combs:
            for attr in self.key_attrs:
                path =  os.path.join(self.imgdir, f'{comb}_{attr}.png')
                imgs.append(path)
        return imgs
    

    def get_file_path(self, vizname, subdir=None, **kwargs):
        file_name = f"{vizname}.png"
        return os.path.join(self.subdir, file_name)
    

    def get_title(self, imgname):
        x = os.path.basename(imgname)
        x = x.split('.')[0]
        mxname, spars, dim, walklength, numwalks, windowsize, untillayer, opt1, opt2, opt3, attr = x.split('_')
        # title = f'{mxname}, \n{spars}, \n{walklength}, \n{windowsize}, \n{attr}'
        # title = f'{walklength}, {windowsize}, {attr}'
        title = f'{attr}'
        return title



class RunModeEval(S2vCreator):
    '''
    Combine all images that have the same matrix name and sparsification method but different parameter settings into one image.
    '''
    def __init__(self, language, by_author=False):
        super().__init__(language, mode='run', by_author=by_author)
        self.el = EmbLoader(self.language, self.file_string, mode='run', by_author=self.by_author) # Use parameters from run mode


    def create_param_grid(self):
        # Combine different params, ignore attrs
        for network_name in self.edgelists:
            network_name, extension = network_name.split('.')
            print(network_name, extension)
            param_combs =  super().get_param_combinations()
            combs = [self.get_param_string(x) for x  in param_combs]
            combs = [f'{network_name}_{x}' for x in combs]
            # for attr in ['author', 'canon', 'gender', 'year']:
            ig = RunModeParamGridPerAttr(self.language, combs, 'noattr')
            ig.visualize(vizname=network_name)
            plt.close('all')


class BestParamAnalyis(S2vCreator):
    
    def __init__(self, language, by_author=False):
        super().__init__(language, mode='bestparams', by_author=by_author)
        self.el = EmbLoader(self.language, self.file_string, mode='bestparams', by_author=self.by_author) # Use parameters from run mode


    def create_attr_grid(self):
        # Combine different params and attrs in one plot
        for network_name in self.edgelists:
            network_name, extension = network_name.split('.')
            print(extension)
            param_combs =  super().get_param_combinations()
            print(network_name)
            combs = [self.get_param_string(x) for x  in param_combs]
            combs = [f'{network_name}_{x}' for x in combs]
            ig = BestparamsAttrGrid(self.language, combs)
            ig.visualize(vizname=network_name)
            plt.close('all')
        


class EmbMxCombinations(EmbLoader, MxCombinations):
    '''
    Adapt matrix clustering on similarity matrices created from text distance measures to similarity matrices created from embeddings.
    '''
    def __init__(self, language, output_dir='s2v', add_color=False, by_author=False, eval_only=False):
        self.file_string = output_dir # 'n2v' or 's2v'
        # Inherit 'load_mxs' method
        EmbLoader.__init__(self, language=language, file_string=self.file_string, mode='run', by_author=by_author)
        # Inherit everything else
        MxCombinations.__init__(self, language=language, output_dir=output_dir, add_color=add_color, by_author=by_author, eval_only=eval_only)





class MirrorViz(S2vCreator):
    '''
    Create single MDS and network plot for mirrored graphs
    Class is adapted from ParamModeEval.
    '''
    def __init__(self, language, by_author=False):
        super().__init__(language=language, mode='mirror', by_author=by_author)


    def create_metadf(self, network):
        '''
        Create df with all node ids col and color col, where each pair of mirrored nodes have the same color. 
        The mirrored node has the same node id as the original node plus three trailing zeroes.
        '''
        color_all_nodes = False
        nodeids = network.nodes()
        nodeids = [x for x in nodeids if '000' not in x]

        if color_all_nodes:
            import colorcet as cc # cc is not installed on cluster
            nr_colors = len(nodeids)
            max_colors = len(cc.glasbey_light) # Maximum number of distinct colors available in the colormap
            colors_list = cc.glasbey_light[:nr_colors]
        else:
            nr_colors = 30
            grey_color = 'lightgray'  # Color for non-colored nodes
            # Randomly choose 30 nodes
            selected_nodes = random.sample(nodeids, nr_colors)
            colors_list = [
                'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown',
                'mediumvioletred', 'white', 'crimson', 'cyan', 'magenta', 'rosybrown', 'lime', 'navy',
                'teal', 'olive', 'gold', 'silver', 'darkgoldenrod', 'coral', 'darkred', 'lightseagreen',
                'turquoise', 'lavender', 'indigo', 'violet', 'salmon', 'darkseagreen'
            ]
        
        df_list = []
        counter = 0
        for nodeid in nodeids:
            if color_all_nodes:
                color = colors_list[counter]
                counter += 1
                df_list.append({'nodeid': nodeid, 'node_color': color})
                df_list.append({'nodeid': f'{nodeid}000', 'node_color': color})
                # Colors are repeated when maximum of 256 distinct colors is reached
                if counter == max_colors:
                    counter = 0
            else:
                if nodeid in selected_nodes:
                    color = colors_list[selected_nodes.index(nodeid)]
                else:
                    color = grey_color
                df_list.append({'nodeid': nodeid, 'node_color': color})
                df_list.append({'nodeid': f'{nodeid}000', 'node_color': color})
        
        df = pd.DataFrame(df_list)
        df = df.set_index('nodeid')
        return df
    
    
    def get_all_mxnames(self):
        all_emb_paths = self.get_all_embedding_paths()
        all_mxnames = [os.path.basename(x) for x in all_emb_paths]
        all_mxnames = [x.split('.')[0] for x in all_mxnames]
        print('embs paths', len(all_emb_paths), 'all mxsnames', len(all_mxnames))
        return all_mxnames


    def create_single_images(self):
        all_mxnames = self.get_all_mxnames()
        self.el = EmbLoader(self.language, file_string=self.file_string, mode='mirror', by_author=self.by_author, map_nodeids_to_names=False)

        nkviz_dir = os.path.join(self.output_dir, 'mirror_singleimage')
        mdsviz_dir = os.path.join(self.output_dir, 'mx_mirror_singleimage')

        for network_name, node in self.examples.items():
            print('network name', network_name)
            print('node name', node)

            # Network is the same for all params independent of embeddings, draw it only once
            edgelist_path = os.path.join(self.edgelist_dir, f'{network_name}.csv')
            network = self.network_from_edgelist(edgelist_path, delimiter=' ', nodes_as_str=False, print_info=False)
            metadf = self.create_metadf(network)

            nkvizpath = os.path.join(nkviz_dir, f'{network_name}.png')
            if not os.path.exists(nkvizpath):
                info = CombinationInfo(attr='node') # info with random attr to init class
                nkviz = MirrorSingleViz(language=self.language, output_dir=self.output_dir, info=info, exp={'attr': node}, by_author=self.by_author, graph=network, metadf=metadf)
                nkviz.visualize_edges()
                nkviz.visualize_nodes(vizname=network_name, plt_kwargs={'dpi': 300})
        
        
            mxnames = [x for x in all_mxnames if network_name in x]
            for cmxname in mxnames:
                mdsvizpath = os.path.join(mdsviz_dir, f'{cmxname}.png')
                if not os.path.exists(mdsvizpath):
                    cmx = self.el.load_single_mx(mxname=cmxname)
                    cmx.name = cmxname
                    mdsviz = EmbParamEvalSingleMDS(language=self.language, output_dir=self.output_dir, exp={'name': 'mirror_singleimage'}, by_author=self.by_author, mc=self.el, df=metadf, attr='node', mx=cmx)
                    mdsviz.visualize(vizname=cmxname, plt_kwargs={'dpi': 300})


    def check_mirror_node_similarity(self):
        # Check how similar mirror node is to original node
        all_mxnames = self.get_all_mxnames()
        self.el = EmbLoader(self.language, file_string=self.file_string, mode='mirror', by_author=self.by_author)

        pickle_path = os.path.join(self.output_dir, 'check_mirror_node_similarity.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as file:
                all_min_expected_col_values, all_expected_col_ranks, row_counter, threshold_counters = pickle.load(file)
        else:
            all_min_expected_col_values = {}
            all_expected_col_ranks = {}
            row_counter = 0
            threshold_counters = {
                '99': 0,
                '95': 0,
                '90': 0,
                '80': 0,
            }
            
            for network_name, node in self.examples.items():
                mxnames = [x for x in all_mxnames if network_name in x]
                for cmxname in mxnames:
                    cmx = self.el.load_single_mx(mxname=cmxname)
                    cmx.name = cmxname

                    # Check if the node with the second highest similarity in the similarity matrix is the mirror node
                    # The highest values of 1 are on the diagonal
                    mx = cmx.mx
                    mx.index = mx.index.astype(str)
                    expected_col_values = []
                    for idx, row in mx.iterrows():
                        row_counter += 1
                        if '000' in idx:
                            continue
                        else:
                            max_col = row.idxmax()
                            if not max_col == idx:
                                print('Highest value in row is not on diagonal')

                            # Find the column name with the second highest value, which is expected to be the mirror node
                            second_max_col = row.nlargest(2).index[1]
                            second_max_value = row[second_max_col]
                            expected_col = f'{idx}000'
                            expected_col_value = row[expected_col]

                            # if and not elif to check every threshold
                            if expected_col_value < 0.99:
                                threshold_counters['99'] += 1
                            if expected_col_value < 0.95:
                                threshold_counters['95'] += 1
                            if expected_col_value < 0.90:
                                threshold_counters['90'] += 1
                            if expected_col_value < 0.80:
                                threshold_counters['80'] += 1

                            # Second most similar node is always mirror node
                            if not second_max_col == expected_col:
                                # print('Second highest value in row is not mirror node.')
                                # print(idx, expected_col, second_max_col)
                                # print('second_max_value', second_max_value, 'expected_col_value', expected_col_value)
                                expected_col_values.append(expected_col_value)


                                # Find rank of mirror node
                                # New code to determine the rank of the value in the expected column
                                rank = row.rank(ascending=False, method='min')
                                expected_col_rank = rank[expected_col]
                                expected_col_rank = expected_col_rank- 1 # First rank is always diagonal
                                if expected_col_rank in all_expected_col_ranks:
                                    all_expected_col_ranks[expected_col_rank] += 1
                                else:
                                    all_expected_col_ranks[expected_col_rank] = 1


                    if not expected_col_values: # empty list
                        print(f'{cmx.name}: Second highest value in row is always mirror node.')
                        min_expected_col_value = 1
                    else:
                        min_expected_col_value = min(expected_col_values)
                        print(f'{cmx.name}: Smallest expected_col_value: ', min_expected_col_value)
                    all_min_expected_col_values[cmx.name] = min_expected_col_value

            with open(pickle_path, 'wb') as file:
                pickle.dump((all_min_expected_col_values, all_expected_col_ranks, row_counter, threshold_counters), file)

        
        print(f'Smalles similarity value of a mirror node across all matrices: {min(all_min_expected_col_values)}')
        # Print smallest values
        sorted_items = sorted(all_min_expected_col_values.items(), key=lambda x: x[1])
        
        # Get the top 10 (or fewer if the dictionary has less than 10 items)
        top_10 = sorted_items[:10]
        
        print("Top 10 smallest values:")
        for key, value in top_10:
            print(f"{key}: {value}")


        for k, v in threshold_counters.items():
            print(f'Percent rows below 0.{k}: {v/row_counter}, Total number: {v}')
        print(f'Total nr. rows: {row_counter}')


class MirrorSingleViz(EmbParamEvalSingleViz):
    def __init__(self, language, output_dir, info=None, plttitle=None, exp=None, by_author=False, graph=None, ignore_nodes_removed=True, metadf=None): 
        super().__init__(language=language, output_dir=output_dir, info=info, plttitle=plttitle, exp=exp, by_author=by_author, graph=graph, ignore_nodes_removed=ignore_nodes_removed)
        self.df = metadf

    def get_metadf(self):
        self.df = self.df
    
    def add_custom_subdir(self):
        # self.sc = S2vCreator(self.language, mode='params')
        # self.output_dir=self.sc.output_dir
        self.add_subdir('mirror_singleimage')


    def get_positions(self):
        super().get_positions()

        new_positions = {}
        for k, v in self.pos.items():
            if '000' in k:
                continue
            else:
                new_positions[k] = v
                x = v[0]
                y = v[1]
                new_positions[f'{k}000'] = (-x, y)

        self.pos = new_positions
