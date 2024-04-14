# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")
from utils import DataHandler
import os
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from embedding_utils import EmbeddingBase
from cluster.combinations import MxCombinations
from cluster.create import D2vDist
from cluster.cluster_utils import MetadataHandler, CombinationInfo
from nkselect import ImageGrid
from nkviz import NkVizBase
from n2vcreator import N2vCreator
from s2vcreator import S2vCreator
from copy import deepcopy


# use network env

class EmbDist(D2vDist):
    def __init__(self, language, output_dir, modes):
        super().__init__(language, output_dir=output_dir, modes=modes)
        self.file_string = output_dir
        self.eb = EmbeddingBase(self.language, output_dir=output_dir)


    def get_embeddings_dict(self, mode):
        '''
        Load embeddings of a single network (mode) in format {node_name: embedding}
        '''
        emb_dict = {}
        df = self.eb.load_embeddings(mode)
        for file_name, row in df.iterrows():
            emb_dict[file_name] = np.array(row)
        return emb_dict
    
# line 46, 208, 209, 245 in evaluate assert!! #########################


class EmbLoader():
    def __init__(self, language, file_string):
        self.language = language
        self.file_string = file_string


    def load_mxs(self, as_list=True):
        if self.file_string == 'n2v':
            ec = N2vCreator(self.language)
        else:
            ec = S2vCreator(self.language)

        # Load similarity matrices, if they don't exist, they are created
        embedding_files = [file.split('.')[0] for file in os.listdir(ec.subdir) if file.endswith('embeddings')] # remove '.embeddings'
        embedding_files = [x for x in embedding_files if 'burrows-500_simmel-5-10' in x] ####################3
        # embedding_files = ['burrows-500_simmel-5-10_dimensions-32_walklength-3_numwalks-50_windowsize-3_untillayer-5_OPT1-True_OPT2-True_OPT3-True'] ###########
        emdist = EmbDist(language=self.language, output_dir=self.file_string, modes=embedding_files)
        mxs = emdist.load_all_data(use_kwargs_for_fn='mode', file_string=self.file_string, subdir=True)


        if as_list:
            mxs_list = []
            for name, mx in mxs.items():
                mx.name = name
                mxs_list.append(mx)
        else:
            mxs_list = mxs
        return mxs_list
    


class EmbeddingEvalViz(NkVizBase):
    '''
    Visualize each parameter combination in a seperate plot, ignore isolated nodes
    '''
    def __init__(self, language, info=True, plttitle=None, exp=None, by_author=False, graph=None):
        super().__init__(language, info, plttitle, exp, by_author, graph)

    def get_figure(self):
        self.fig, self.axs = plt.subplots()
        self.axs = np.reshape(self.axs, (1, 1))
        self.axs[0, 0].axis('off')

    def add_custom_subdir(self):
        self.sc = S2vCreator(self.language)
        self.output_dir=self.sc.output_dir
        self.add_subdir('singleimages')
        print(self.subdir)

    def get_graphs(self):
        # nx.connected_components is only implemented for undirected graphs
        if nx.is_directed(self.graph):
            graph = self.graph.to_undirected()
        else:
            graph = deepcopy(self.graph)
        self.graph_con = self.graph
        self.nodes_removed = []


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

    def get_path(self, name, omit):
        return os.path.join(self.subdir, f'{name}.{self.data_type}')



class EvalImageGrid(ImageGrid):
    def __init__(self, language, combs):
        self.combs = combs
        super().__init__(language, attr=None, by_author=False, output_dir='s2v', imgdir='singleimages')
        self.nrow = 3
        self.ncol = 3
        self.imgs = self.load_single_images()
        self.fontsize = 6

        self.img_width = 1.92 # format of single-network plots stored on file
        self.img_height = 1.44
        self.nr_mxs = 58
        self.nr_spars = 9

        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0
        self.ws_hspace = 0.1

        self.add_subdir('gridimage')


    def load_single_images(self):
        imgs = []
        for comb in self.combs:
            path =  os.path.join(self.imgdir, f'{comb}.png')
            imgs.append(path)
        return imgs
    

    def get_file_path(self, vizname, subdir=None, **kwargs):
        file_name = f"{vizname}_dimensions-{kwargs['dimensions']}_untillayer-{kwargs['untillayer']}_windowsize{kwargs['windowsize']}.png"
        return os.path.join(self.subdir, file_name)
    

    def get_title(self, imgname):
        x = os.path.basename(imgname)
        print(x)
        mxname, spars, dim, walklengths, numwalks, windowsize, untillayer, opt1, opt2, opt3 = x.split('_')
        title = f'{walklengths}, {numwalks}' #, {windowsize}'
        return title




class ParamEval(S2vCreator):
    def __init__(self, language):
        super().__init__(language)
        self.el = EmbLoader(self.language, self.file_string)


    def check_embeddings(self):
        nr_comb = self.count_combinations()
        print(f'Expected number of combinations: {nr_comb}')
        nr_embeddings = len(os.listdir(self.subdir))
        print(f'Nr combinations in subdir: {nr_embeddings}')



    def create_single_images(self):
        self.mxs_dict = self.el.load_mxs(as_list=False)


        for network_name, node in self.examples.items():
            if node is not None:
                mxs = {k: v for k, v in self.mxs_dict.items() if network_name in k}
                for cmxname, cmx in mxs.items():
                    df = cmx.mx.loc[node].to_frame()


                    mh = MetadataHandler(self.language)
                    df = mh.add_color_column(metadf=df, colname=node)
                    df.loc[node, f'{node}_color'] = 'green'

                    # path = '/home/annina/scripts/great_unread_nlp/data/s2v/eng/burrows-500_simmel-5-10/viz-Corelli_Marie_The-Sorrows-of-Satan_1895.png'
                    # if os.path.exists(path): ###############
                    #     os.remove(path)
                    info = CombinationInfo(metadf=df, attr=node)
                    network = self.network_from_edgelist(os.path.join(self.edgelist_dir, f'{network_name}.csv'), delimiter=' ', nodes_as_str=True, print_info=False)
                    viz = EmbeddingEvalViz(self.language, info=info, exp={'attr': node}, by_author=False, graph=network)
                    viz.visualize(vizname=cmxname)


    def create_grid_images(self):
        for network_name, node in self.examples.items():
            if node is not None:

                params = self.get_params()
                param_combs =  super().get_param_combinations()
                for dim in params['dimensions']:
                    for ul in params['until-layer']:
                        for ws in params['window-size']:

                            combs = [
                                d for d in param_combs
                                if d['dimensions'] == dim and d['until-layer'] == ul and d['window-size'] == ws
                            ]
                            combs = [self.get_param_string(x) for x  in combs]
                            combs = [f'{network_name}_{x}' for x in combs]
                            ig = EvalImageGrid(self.language, combs)
                            ig.visualize(vizname=network_name, dimensions=dim, untillayer=ul, windowsize=ws)


pe = ParamEval('eng')
pe.create_grid_images()
# pe.check_embeddings()


















class EmbMxCombinations(EmbLoader, MxCombinations):
    def __init__(self, language, output_dir, add_color=False, by_author=False):
        self.file_string = output_dir # 'n2v' or 's2v'
        # Inherit 'load_mxs' method
        EmbLoader.__init__(self, language=language, file_string=self.file_string)
        # Inherit everything else
        MxCombinations.__init__(self, language=language, output_dir=output_dir, add_color=add_color, by_author=by_author)

# print(EmbMxCombinations.mro())
# emc = EmbMxCombinations(language='eng', output_dir='s2v')
# emc.evaluate_all_combinations()
# %%
