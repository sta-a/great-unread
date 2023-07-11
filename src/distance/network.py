
import numpy as np
import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
import random
import pandas as pd
from .distance_visualization import ColorMap


class Network():
    def __init__(self, language, output_dir='distance', mx = None, G = None, draw = True, cluster_alg = None, attribute_name = None): # attribute: labeled groups, i.e. 'm', 'f' #################################):
        super().__init__(language, output_dir)
        self.mx = mx
        self.G = G
        if (self.mx is not None and self.G is not None) or (self.mx is None and self.G is None):
            raise ValueError('Pass either matrix or graph.')
        if self.G is None:
            self.G = self.network_from_mx()
        self.get_graph_info()
        self.draw = draw,
        self.cluster_alg = cluster_alg
        self.attribute_name = attribute_name


        # Create clustering if name of clustering algorithm is passed
        if self.cluster_alg is not None:
            self.clusters = self.get_communities()
        if self.draw:
            self.draw_graph()
            self.get_colormap = self.get_colormap()

    def get_graph_info(self):
        raise NotImplementedError

    def get_colormap(self):
        if self.cluster_alg is not None or self.attribute_name is not None: ############################3
            if (self.attribute_name == 'author') or (self.attribute_name == 'unread') or (self.attribute_name == 'gender'):
                cluster_list = list(self.G.nodes(data=False)) # get list of file names
                color_group= self.attribute_name
            else:
                cluster_list = self.clusters
                color_group = self.cluster_alg

            fn_color_mapping = ColorMap(
                color_group,
                cluster_list,
                self.language).get_color_map()
            assert self.G.number_of_nodes() == len(fn_color_mapping)
            color_map = [fn_color_mapping[node] for node in self.G]
        else:
            color_map = 'blue'

    # def get_jazz_network(self):
    #     print('Jazz network as networkit grap.')
    #     # Jazz network (unweighted edge list)
    #     jazz_path = '/home/annina/scripts/great_unread_nlp/src/download.tsv.arenas-jazz/arenas-jazz/out.arenas-jazz'
    #     # col_names = ['source', 'target'] #see readme of dataset: 'weight', 'timestamp' should also exist but don't
    #     # df = pd.read_csv(jazz_path, sep='\t', header=0, names=col_names)
    #     # print(df)
    #     edgeListReader = nk.graphio.EdgeListReader('\t', 1, '%')
    #     G = edgeListReader.read(jazz_path)
    #     return G



class NXNetwork(Network):
    def __init__(self):
        super().__init__()

    def network_from_mx(self):
        '''
        mx: Adjacency or directed or undirected weight matrix
        '''
        mx = self.mx.fillna(0) # Adjacency matrix must not contain Nan!
        if mx.equals(mx.T) :
            G = nx.from_pandas_adjacency(mx)
            print('Matrix is symmetric.')
        else:
            G = nx.from_pandas_adjacency(mx, create_using=nx.DiGraph) 
            print('Matrix is not symmetric but directed.')
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    def get_communities(self):
        if self.cluster_alg == 'louvain':
            c = nx.community.louvain_communities(self.G, weight='weight', seed=11, resolution=0.1)
            return c
    
    def draw_graph(self):
        # No graph drawing possible in Networkit
        pos = nx.spring_layout(self.G, seed=8)
        nx.draw_networkx(self.G, pos, node_size=30, with_labels=False, node_color=self.color_map)
        plt.show()


    def get_graph_info(self):
        print(f'Nr. of nodes: {self.G.number_of_nodes()}, \
                \nNr. of edges: {self.G.number_of_edges()}, \
                \nis weighted: {nx.is_weighted(self.G)}, \
                \nis directed: {nx.is_directed(self.G)}, \
                \nNr. of Selfloops: {nx.number_of_selfloops(self.G)}, \
                \n------------------------------------\n')
        
    def nx_check_nan_edge_weights(self):
        '''
        Return True if a graph has nan edge weights.
        '''
        edge_labels = nx.get_edge_attributes(self.G,'weight')
        weights = [weight for tup, weight in edge_labels.items()]
        nan_bool = np.any(np.isnan(weights)) # True if there are nan
        return nan_bool


    def replace_nan_edge_weights(self, replacement_value=0):
        '''
        G: nx graph
        replacement_value: value with which np.nans should be replaced
        '''
        edge_labels = nx.get_edge_attributes(self.G,'weight')
        new_edge_labels = {tup: replacement_value if np.isnan(weight) else weight for tup, weight in edge_labels.items()}
        
        nx.set_edge_attributes(self.G, new_edge_labels, "weight") # inplace!
        self.get_graph_info(self.G)

    def nx_to_nk(self, G):
        return nk.nxadapter.nx2nk(G, weightAttr='weight') #weightAttr: The edge attribute which should be treated as the edge weight
    
    def nk_to_nx(self, G):
        return nk.nxadapter.nk2nx(G)

    def get_nk_graph_info(self):
        print(nk.overview(self.G))

    def nk_iterate_edges(self):
        def edgeFunc(u, v, weight, edgeId):
                print("Edge from {} to {} has weight {} and id {}".format(u, v, weight, edgeId))
        # Using iterator with callback function.
        self.G.forEdges(edgeFunc) # forEdges iterator from nk

    def nk_complete_weighted_graph(self, n):
        G = nx.complete_graph(n)
        G = nk.nxadapter.nx2nk(G)
        # Iterate over the edges of G and add weights
        G = nk.graphtools.toWeighted(G) # weights = 1
        for u, v, w in G.iterEdgesWeights():
                w = random.random()
                G.setWeight(u, v, w)
        return G

    def random_weighted_undirected_graph(self):
        nkG = nk.generators.MocnikGenerator(dim=2, n=100, k=2, weighted=True).generate() # Density parameter, determining the ratio of edges to nodes
        # print(nk.overview(nkG))
        nxG = self.nk_to_nx(nkG).to_undirected()

        plt.figure(1, figsize=(11, 11))
        #nx.draw_networkx(nxG)
        #pos = nx.circular_layout(nxG)
        pos = nx.spring_layout(nxG)
        #nx.draw_networkx(nxG, pos=pos, node_size=30, with_labels=False, node_color='red')
        edge_labels = nx.get_edge_attributes(nxG,'weight')
        for currtuple, weight in edge_labels.items():
                edge_labels[currtuple] = round(weight, 3)
        nx.draw_networkx_nodes(nxG, pos, node_size=20, node_color='red')
        nx.draw_networkx_edges(nxG, pos, edge_color="blue", width=1)
        nx.draw_networkx_edge_labels(nxG, pos=pos, edge_labels = edge_labels, font_size=6, verticalalignment='top')
        plt.show()
