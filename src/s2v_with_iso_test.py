# %%
'''
Create embeddings for a single distance and parameter combination that includes the isolated nodes.
'''
import os
from analysis.s2vcreator import S2vCreator
from analysis.nkviz import NkSingleViz
from matplotlib import pyplot as plt


class S2vCreatorWithIsoNodes(S2vCreator):
    '''
    Activate s2v env!
    '''
    def __init__(self, language, mode, by_author=False):
        print(language, mode, by_author)
        super().__init__(language=language, mode=mode, by_author=by_author)
        self.edgelist_dir = '/home/annina/scripts/great_unread_nlp/data/similarity/eng/sparsification_edgelists_noisotest' # edgelist with isolated nodes, spaces instead of commas for s2v compatibility
        self.edgelists = ['both_simmel-4-6.csv']
        self.remove_iso = False


    def get_embedding_path(self, edgelist, kwargs):
        if '.csv' in edgelist:
            edgelist = os.path.splitext(edgelist)[0]
        param_string = self.get_param_string(kwargs)
        return os.path.join(self.subdir, f'{edgelist}_{param_string}-noisonodes.embeddings')
    
    def get_params_mode_params(self):
        # Return many different parameter combinations for parameter selection
        params = {
            'dimensions': [5],
            'walk-length': [3],
            'num-walks': [20],
            'window-size': [3],
            'until-layer': [5],
            'OPT1': ['True'],
            'OPT2': ['True'],
            'OPT3': ['True']
        }
        return params
    

class ParamModeEvalTest(S2vCreatorWithIsoNodes):
    '''
    Evaluate different parameter settings for s2v by highlighting nodes in network according to their similarity to a selected node in a prominent position.
    '''
    def __init__(self, language, by_author=False):
        super().__init__(language=language, mode='params', by_author=by_author)
        self.examples = {
            'both_simmel-4-6': 'Wells_H-G_Tono-Bungay_1909', 
        }
        self.remove_iso = False

    def create_single_images(self):
        all_emb_paths = self.get_all_embedding_paths()
        all_mxnames = [os.path.basename(x) for x in all_emb_paths]
        all_mxnames = [x.split('.')[0] for x in all_mxnames]
        s2vobj = S2vCreatorWithIsoNodes(self.language, self.mode, self.by_author)
        self.el = EmbLoader(self.language, file_string=self.file_string, mode='params', ec=s2vobj)

        nkviz_dir = os.path.join(self.output_dir, 'singleimages')
        mdsviz_dir = os.path.join(self.output_dir, 'mx_singleimages')


        for network_name, node in self.examples.items():
            # Network is the same for all params
            # Draw network edges only once, repeatedly draw nodes
            network = self.network_from_edgelist(os.path.join(self.edgelist_dir, f'{network_name}.csv'), delimiter=' ', nodes_as_str=True, print_info=False)
            isolates = list(nx.isolates(network))
            # print('nr isolates', len(isolates), nx.number_of_isolates(network))
            # for i in isolates[:5]:
            #     print(i)
            # print('nr nodes', network.number_of_nodes(), 'niso', network.number_of_isolates())
            info = CombinationInfo(attr='canon') # info with random attr to init class
            # nkviz = EmbParamEvalSingleViz(language=self.language, output_dir=self.output_dir, info=info, exp={'attr': node}, by_author=self.by_author, graph=network, ignore_nodes_removed=False)
            # nkviz.visualize_edges()
    
    
            mxnames = [x for x in all_mxnames if network_name in x]
            for cmxname in mxnames:
                # Checking paths with class instances is too slow. Instead, create paths directly.
                nkvizpath = os.path.join(nkviz_dir, f'{cmxname}.png')
                mdsvizpath = os.path.join(mdsviz_dir, f'{cmxname}.png')

                # Only create simmxs if necessary
                # print('nkvizpath', nkvizpath)
                # print('mdsvizpath', mdsvizpath)
                # print(os.path.exists(nkvizpath), os.path.exists(mdsvizpath))
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
                df.loc[df.index.isin(isolates), f'{node}_color'] = 'gold'
                assert len(df) == len(cmx.mx)
                

                # Add nodes to network with colors for current parameter combination
                # info = CombinationInfo(metadf=df, attr=node)
                # nkviz.df = df
                # nkviz.info = info
                # nkviz.visualize_nodes(vizname=cmxname)
                nkviz = NkSingleVizS2vTest(self.language, output_dir=self.output_dir, exp={'name': 'mx_singleimages'}, by_author=self.by_author, network=network, name=cmxname, colorcol=f'{node}', vizpath=nkvizpath, df=df)
                nkviz.visualize()



                # MDS of similarity matrix
                mdsviz = EmbParamEvalSingleMDS(language=self.language, output_dir=self.output_dir, exp={'name': 'singleimages'}, by_author=self.by_author, mc=self.el, df=df, attr=node, mx=cmx)
                mdsviz.visualize(vizname=cmxname)


class NkSingleVizS2vTest(NkSingleViz):
    def __init__(self, language, output_dir, exp, by_author, network, name, colorcol, vizpath, df):
        super().__init__(language, output_dir, exp, by_author)
        self.network = network
        self.name = name
        self.colorcol = colorcol
        self.vizpath = vizpath
        self.df = df
        self.graph = self.network



    def add_positions_to_metadf(self):
        # Combine positions and metadata
        print(self.df.shape)
        print(len(self.pos))
        self.df['pos'] = self.df.index.map(self.pos)
        self.df[['x', 'y']] = pd.DataFrame(self.df['pos'].tolist(), index=self.df.index)

    def get_path(self, name, omit=None):
        return os.path.join(self.subdir, f'{name}.{self.data_type}')


    def visualize(self, vizname='viz'): # vizname for compatibility
        self.vizpath = self.get_path(name=self.name, omit=[])

        i = 0
        j = 0
        self.get_figure()
        
        self.global_vmax, self.global_vmin = self.get_cmap_params()
        self.get_graphs()
        self.get_positions()
        self.add_positions_to_metadf()
        self.subplots = [[i, j]]
        self.add_edges()
        self.add_nodes_to_ax([i,j], self.df, color_col=self.colorcol, use_different_shapes=False)
        self.save_plot(plt)
        plt.savefig(self.vizpath)

# def get_iso_nodes():
# Compare embeddings for iso nodes, does not work
#     path = '/home/annina/scripts/great_unread_nlp/data/similarity/eng/sparsification_edgelists_noisotest/both_simmel-4-6.csv'
#     df = pd.read_csv(path, header=None, sep=' ')
#     isonodes = [row[0] for _, row in df.iterrows() if row[0] == row[1] and row[2] == 1]
#     isonodes = list(set(isonodes))
#     isonodes = [str(int(x)) for x in isonodes] #############3all nodes have selfloop 
#     print(isonodes)

#     embpath = '/home/annina/scripts/great_unread_nlp/data/s2v/eng/embeddings/both_simmel-4-6_dimensions-5_walklength-3_numwalks-20_windowsize-3_untillayer-5_OPT1-True_OPT2-True_OPT3-True-noisonodes.embeddings'
#     df = pd.read_csv(embpath, skiprows=1, header=None, sep=' ', index_col=0, dtype={0: str})
    
#     print(df.index.isin(isonodes))
#     selected_rows = df.loc[df.index.isin(isonodes)]
#     print(selected_rows)



# Create embedding
# sc = S2vCreatorWithIsoNodes(language='eng', mode='params', by_author=False)
# paths = sc.get_all_embedding_paths()
# for p in paths:
#     print(p)
# sc.run_combinations()



# Create visualizations
from analysis.embedding_eval import ParamModeEval, EmbLoader, EmbParamEvalSingleViz, EmbParamEvalSingleMDS
from cluster.cluster_utils import CombinationInfo, MetadataHandler
import pandas as pd
import networkx as nx
import shutil
source_dir = '/home/annina/scripts/great_unread_nlp/data/s2v/eng/embeddings_noiso'
dest_dir = '/home/annina/scripts/great_unread_nlp/data/s2v/eng/embeddings'
filename = 'both_simmel-4-6_dimensions-5_walklength-3_numwalks-20_windowsize-3_untillayer-5_OPT1-True_OPT2-True_OPT3-True-noisonodes.embeddings'

# Construct full file paths
source_file = os.path.join(source_dir, filename)
dest_file = os.path.join(dest_dir, filename)

# Copy the file
shutil.copy2(source_file, dest_file)

pe = ParamModeEvalTest(language='eng', by_author=False)
paths = pe.get_all_embedding_paths()
# for p in paths:
#     print(p)
pe.create_single_images()
# get_iso_nodes()
os.remove(dest_file)

# %%
