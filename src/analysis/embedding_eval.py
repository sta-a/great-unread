
import sys
sys.path.append("..")
from utils import DataHandler
import os
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm

from .embedding_utils import EmbeddingBase
from .nkselect import ImageGrid
from .nkviz import NkVizBase
from cluster.combinations import MxCombinations
from cluster.create import D2vDist
from cluster.cluster_utils import MetadataHandler, CombinationInfo
# from .n2vcreator import N2vCreator
from .s2vcreator import S2vCreator
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
    def __init__(self, language, file_string, mode=None, list_of_substrings=[]):
        '''
        list_of_substrings: embedding files must contain a substring from this list to be included if list is not empty
        '''
        self.language = language
        self.file_string = file_string
        self.mode = mode
        self.list_of_substrings = list_of_substrings
        self.filter_embedding_files()


    def load_single_mx(self, mxname):
        emdist = EmbDist(language=self.language, output_dir=self.file_string, modes=self.embedding_files)
        mx = emdist.load_data(load=True, mode=mxname, use_kwargs_for_fn='mode', file_string=self.file_string, subdir=True)
        print(f'Loaded single mx {mx.name}')
        return mx


    def filter_embedding_files(self):
        '''
        Get all embedding files that belong to either the 'run' or the 'params' mode EdgelistHandler, or that are passed as a list of substrings.
        '''
        # if self.file_string == 'n2v':
        #     self.ec = N2vCreator(self.language, mode=self.mode)
        # else:
        self.ec = S2vCreator(self.language, mode=self.mode)

        all_embedding_paths = self.ec.get_all_embedding_paths()
        all_embedding_paths = [os.path.basename(x) for x in all_embedding_paths]
        print('nr emb paths', len(all_embedding_paths))
        all_embedding_paths = [x.split('.')[0] for x in all_embedding_paths]

        print('all emb path', len(all_embedding_paths))

        # Filter list of embedding files
        embedding_files = [file.split('.')[0] for file in os.listdir(self.ec.subdir) if file.endswith('embeddings')] # remove '.embeddings'
        if self.mode is not None:
            embedding_files = [file for file in embedding_files if any(sub in file for sub in all_embedding_paths)]

        if self.list_of_substrings:
            embedding_files = [file for file in embedding_files if any(sub in file for sub in self.list_of_substrings)]

        print('len ebm files', len(embedding_files))
        self.embedding_files = embedding_files


    def load_mxs(self):
        # Load similarity matrices. If they don't exist, they are created
        emdist = EmbDist(language=self.language, output_dir=self.file_string, modes=self.embedding_files)


        total_iterations = len(self.embedding_files)
        for i, mode in enumerate(self.embedding_files):
            mx = emdist.load_data(load=True, mode=mode, use_kwargs_for_fn='mode', file_string=self.file_string, subdir=True)
            iteration_number = i + 1

            mx.name = mode

            # Print every 100th file
            if iteration_number % 100 == 0 or iteration_number == total_iterations:
                # Print the current iteration number and the total number of iterations
                print(f"Current embedding file: {iteration_number} out of {total_iterations}")
            yield mx


class EmbParamEvalViz(NkVizBase):
    '''
    Visualize each parameter combination in a seperate plot, ignore isolated nodes
    '''
    def __init__(self, language, info=True, plttitle=None, exp=None, by_author=False, graph=None):
        super().__init__(language, info, plttitle, exp, by_author, graph)
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.01
        self.ws_hspace = 0.01
        self.markersize = 7

    def get_figure(self):
        self.fig, self.axs = plt.subplots(figsize=(5, 4))
        self.axs = np.reshape(self.axs, (1, 1))
        self.axs[0, 0].axis('off')

    def add_custom_subdir(self):
        self.sc = S2vCreator(self.language, mode='params')
        self.output_dir=self.sc.output_dir
        self.add_subdir('singleimages')

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



class EmbParamEvalGrid(ImageGrid):
    def __init__(self, language, combs):
        self.combs = combs
        super().__init__(language, attr=None, by_author=False, output_dir='s2v', imgdir='singleimages')
        self.nrow = 3
        self.ncol = 3
        self.imgs = self.load_single_images()
        self.fontsize = 6

        self.img_width = 1.92*2 # format of single-network plots stored on file, times 2 to make fig bigger
        self.img_height = 1.44*2
        self.nr_mxs = 58
        self.nr_spars = 9

        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.01
        self.ws_hspace = 0.01

        self.add_subdir('gridimage')


    def load_single_images(self):
        imgs = []
        for comb in self.combs:
            path =  os.path.join(self.imgdir, f'{comb}.png')
            imgs.append(path)
        return imgs
    

    def get_file_path(self, vizname, subdir=None, **kwargs):
        file_name = f"{vizname}_dimensions-{kwargs['dimensions']}_windowsize{kwargs['windowsize']}.png"
        return os.path.join(self.subdir, file_name)
    

    def get_title(self, imgname):
        x = os.path.basename(imgname)
        mxname, spars, dim, walklengths, numwalks, windowsize = x.split('_')
        title = f'{walklengths}, {numwalks}' #, {windowsize}'
        return title





class ParamEval(S2vCreator):
    '''
    Evaluate different parameter settings for s2v by highlighting nodes in network according to their similarity to a selected node in a prominent position.
    '''
    def __init__(self, language):
        super().__init__(language, mode='params')
        self.el = EmbLoader(self.language, self.file_string, mode='params')


    def check_embeddings(self):
        nr_comb = self.count_combinations()
        print(f'Expected number of combinations: {nr_comb}')
        nr_embeddings = len(os.listdir(self.subdir))
        print(f'Nr combinations in subdir: {nr_embeddings}')


    def create_single_images(self):
        mxs_dict = {}
        mxs_generator = self.load_mxs()
        for i in mxs_generator:
            mxs_dict[i.name] = i
        

        for network_name, node in self.examples.items():
            # Select mxs with different parameters that have the same name as network_name
            mxs = {k: v for k, v in mxs_dict.items() if network_name in k}
            for cmxname, cmx in mxs.items():


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
                

                info = CombinationInfo(metadf=df, attr=node)
                network = self.network_from_edgelist(os.path.join(self.edgelist_dir, f'{network_name}.csv'), delimiter=' ', nodes_as_str=True, print_info=False)
                viz = EmbParamEvalViz(self.language, info=info, exp={'attr': node}, by_author=False, graph=network)
                viz.visualize(vizname=cmxname)


    def create_grid_images(self):
        for network_name, node in self.examples.items():
            # if node == 'Corelli_Marie_The-Sorrows-of-Satan_1895' or node == 'Dronke_Ernst_Polizeigeschichten_1846': ################333
            params = self.get_params()
            param_combs =  super().get_param_combinations()


            for dim in params['dimensions']:
                for ws in params['window-size']:
                    combs = [
                        d for d in param_combs
                        if d['dimensions'] == dim and d['window-size'] == ws
                    ]
                    combs = [self.get_param_string(x) for x  in combs]
                    combs = [f'{network_name}_{x}' for x in combs]
                    ig = EmbParamEvalGrid(self.language, combs)
                    ig.visualize(vizname=network_name, dimensions=dim, windowsize=ws)




class EmbMxGrid(ImageGrid):
    def __init__(self, language, combs, attr):
        self.combs = combs
        super().__init__(language, attr=attr, by_author=False, output_dir='analysis_s2v', imgdir='mx_singleimage')
        self.nrow = 1
        self.ncol = 4
        self.imgs = self.load_single_images()
        self.fontsize = 6

        self.img_width = 1.2*2 # format of single-network plots stored on file, times 2 to make fig bigger
        self.img_height = 2.4*2
        self.nr_interesting_networks = 306
        self.nr_param_combs = 4 # 2 values for walk-length, 2 values for window-size

        # Whitespace
        self.ws_left = 0.01
        self.ws_right = 0.99
        self.ws_bottom = 0.01
        self.ws_top = 0.99
        self.ws_wspace = 0.01
        self.ws_hspace = 0.01

        self.add_subdir('gridimage')


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
        mxname, spars, dim, walklengths, numwalks, windowsize, attr = x.split('_')
        title = f'{walklengths}, {windowsize}'
        return title



class S2vMxvizEval(S2vCreator):
    '''
    Combine all images that have the same matrix name and sparsification method but different parameter settings into one image.
    '''
    def __init__(self, language):
        super().__init__(language, mode='run')
        self.el = EmbLoader(self.language, self.file_string, mode='run') # Parameter for run mode


    def create_grid_images(self):
        for network_name in self.nklist:
            if 'simmel' in network_name: ##################3
                param_combs =  super().get_param_combinations()
                print(network_name)
                combs = [self.get_param_string(x) for x  in param_combs]
                combs = [f'{network_name}_{x}' for x in combs]
                for i in combs:
                    print('comb', i)
                ig = EmbMxGrid(self.language, combs, 'canon')
                ig.visualize(vizname=network_name)





class EmbMxCombinations(EmbLoader, MxCombinations):
    '''
    Adapt matrix clustering on similarity matrices created from text distance measures to similarity matrices created from embeddings.
    '''
    def __init__(self, language, output_dir='s2v', add_color=False, by_author=False):
        self.file_string = output_dir # 'n2v' or 's2v'
        # Inherit 'load_mxs' method
        EmbLoader.__init__(self, language=language, file_string=self.file_string, mode='run')
        # Inherit everything else
        MxCombinations.__init__(self, language=language, output_dir=output_dir, add_color=add_color, by_author=by_author)



