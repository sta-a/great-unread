import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
import random
from network import NXNetwork
import pygraphviz as pgv
import sys
sys.path.append("..")


class NkNetwork(NXNetwork):
    '''
    Class for using networkit functionalities.
    '''
    def __init__(self, language, mx=None, path=None, graph=None):
        super().__init__(language, mx, path, graph)



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
        elif isinstance(self.graph, nk.Graph):
            print('nk Graph Overview:', nk.overview(self.graph))
        elif isinstance(self.graph, pgv.AGraph):
            print(f'--------------\n'
                f'PGV Graph Overview\n'
                f'Number of Nodes: {self.graph.number_of_nodes()}\n'
                f'Number of Edges: {self.graph.number_of_edges()}\n'
                f'Is Directed: {self.graph.is_directed()}\n'
                f'Layout Program: {self.graph.layout()}\n'
                f'--------------\n')




    def nx_to_nk(self, graph):
        return nk.nxadapter.nx2nk(graph, weightAttr='weight') #weightAttr: The edge attribute which should be treated as the edge weight
    
    def nk_to_nx(self, graph):
        return nk.nxadapter.nk2nx(graph)      

    def nk_iterate_edges(self):
        def edgeFunc(u, v, weight, edgeId):
                print("Edge from {} to {} has weight {} and id {}".format(u, v, weight, edgeId))
        # Using iterator with callback function.
        self.graph.forEdges(edgeFunc) # forEdges iterator from nk

    def nk_complete_weighted_graph(self, n):
        graph = nx.complete_graph(n)
        graph = nk.nxadapter.nx2nk(graph)
        # Iterate over the edges of graph and add weights
        graph = nk.graphtools.toWeighted(graph) # weights = 1
        for u, v, w in graph.iterEdgesWeights():
                w = random.random()
                graph.setWeight(u, v, w)
        return graph

    def random_weighted_undirected_graph(self):
        nkgraph = nk.generators.MocnikGenerator(dim=2, n=100, k=2, weighted=True).generate() # Density parameter, determining the ratio of edges to nodes
        nxgraph = self.nk_to_nx(nkgraph).to_undirected()

        plt.figure(1, figsize=(11, 11))
        #nx.draw_networkx(nxgraph)
        #pos = nx.circular_layout(nxgraph)
        pos = nx.spring_layout(nxgraph)
        #nx.draw_networkx(nxgraph, pos=pos, node_size=30, with_labels=False, node_color='red')
        edge_labels = nx.get_edge_attributes(nxgraph,'weight')
        for currtuple, weight in edge_labels.items():
                edge_labels[currtuple] = round(weight, 3)
        nx.draw_networkx_nodes(nxgraph, pos, node_size=20, node_color='red')
        nx.draw_networkx_edges(nxgraph, pos, edge_color="blue", width=1)
        nx.draw_networkx_edge_labels(nxgraph, pos=pos, edge_labels = edge_labels, font_size=6, verticalalignment='top')
        plt.show()

