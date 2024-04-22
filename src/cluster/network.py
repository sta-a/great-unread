import numpy as np
import networkx as nx
import sys
sys.path.append("..")
from .sparsifier import Sparsifier


class NXNetwork():
    '''
    Class for basic network handling.
    '''
    def __init__(self, language, mx=None, path=None, graph=None):
        self.language = language
        self.mx = mx # self.mx is a create.SimMx object, not a df
        self.path = path
        self.graph = graph


        if (self.mx is not None and self.path is not None):
            raise ValueError('Pass either matrix or path.')
        if self.mx is None and self.path is not None:
            self.mx = Sparsifier.load_pkl(self.path)

        if (self.mx is not None and self.graph is not None) or (self.mx is None and self.graph is None):
            raise ValueError('Pass either matrix or graph.')
        if self.graph is None and self.mx is not None:
            self.graph = self.network_from_mx()


    def network_from_mx(self):
        if self.mx.mx.equals(self.mx.mx.T):
            graph = nx.from_pandas_adjacency(self.mx.mx)
            self.logger.debug(f'Created undirected graph from symmetric matrix.')
        else:
            graph = nx.from_pandas_adjacency(self.mx.mx, create_using=nx.DiGraph) 
            self.logger.debug(f'Created directed graph from non-symmetric matrix.')
        # Remove selfloops
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph
     

    def get_graph_info(self):
        if isinstance(self.graph, nx.Graph):
            print(f'--------------\n'
                f'nx Graph Overview: \n'
                f'Nr. of nodes: {self.graph.number_of_nodes()} \n'
                f'Nr. of edges: {self.graph.number_of_edges()} \n'
                f'is weighted: {nx.is_weighted(self.graph)} \n'
                f'is directed: {nx.is_directed(self.graph)} \n'
                f'Nr. of Selfloops: {nx.number_of_selfloops(self.graph)} \n'
                f'--------------\n')


    def nx_check_nan_edge_weights(self):
        '''
        Return True if a graph has nan edge weights.
        '''
        edge_labels = nx.get_edge_attributes(self.graph,'weight')
        weights = [weight for tup, weight in edge_labels.items()]
        nan_bool = np.any(np.isnan(weights)) # True if there are nan
        return nan_bool

    def replace_nan_edge_weights(self, replacement_value=0):
        '''
        graph: nx graph
        replacement_value: value with which np.nans should be replaced
        '''
        edge_labels = nx.get_edge_attributes(self.graph,'weight')
        new_edge_labels = {tup: replacement_value if np.isnan(weight) else weight for tup, weight in edge_labels.items()}
        
        nx.set_edge_attributes(self.graph, new_edge_labels, "weight") # inplace!
        # self.get_graph_info(self.graph)
