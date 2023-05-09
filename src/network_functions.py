import networkit as nk
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from distance_analysis import check_symmetric


# Convert pandas adjacency matrix to nk graph
def nk_iterate_edges(G):
      def edgeFunc(u, v, weight, edgeId):
            print("Edge from {} to {} has weight {} and id {}".format(u, v, weight, edgeId))
      # Using iterator with callback function.
      G.forEdges(edgeFunc) # forEdges iterator from nk


def nx_print_graph_info(G):
      print(f'Nr. of nodes: {G.number_of_nodes()}, \
            \nNr. of edges: {G.number_of_edges()}, \
            \nis weighted: {nx.is_weighted(G)}, \
            \nis directed: {nx.is_directed(G)}, \
            \nNr. of Selfloops: {nx.number_of_selfloops(G)}, \
            \n------------------------------------\n\n')
      

def nk_visualize_graph(nkG):
      nxG = nk.nxadapter.nk2nx(nkG)
      nx.draw_networkx(nxG)


def nx_graph_from_mx(mx, plot=True, color_map_func=None):
      '''
      mx: Assymetric weight or adjacency matrix
      '''
      # Visualization
      mx = mx.fillna(0) # Adjacency matrix must not contain Nan!
      if check_symmetric(mx):
            G = nx.from_pandas_adjacency(mx)
            print('Matrix is symmetric')
      else:
            G = nx.from_pandas_adjacency(mx, create_using=nx.DiGraph) 
            print('Matrix is not symmetric but directed.')
      G.remove_edges_from(nx.selfloop_edges(G))

      if plot:
            if color_map_func is not None:
                  fn_colors_map = color_map_func(list_of_filenames=mx.index)
                  color_map = [fn_colors_map[node] for node in G]
            else:
                  color_map = 'blue'

            pos = nx.random_layout(G)
            nx.draw_networkx(G, pos, node_size=30, with_labels=False, node_color=color_map)
            plt.show()
      return G




def nk_graph_from_adjacency(mx):
      print('Converting adjacency or weight matrix to networkit graph.')
      # Check if matrix is symmetric
      nxG = nx_graph_from_mx(mx, plot=False, color_map_func=None)

      nkG = nk.nxadapter.nx2nk(nxG, weightAttr='weight') #weightAttr: The edge attribute which should be treated as the edge weight
      print(nk.overview(nkG))
      # nk_iterate_edges(nkG)
      # Test by plotting networkx
      nk_visualize_graph(nkG)
      return nkG

def nx_check_nan_edge_weights(G):
      '''
      Return True if a graph has nan edge weights.
      '''
      edge_labels = nx.get_edge_attributes(G,'weight')
      weights = [weight for tup, weight in edge_labels.items()]
      nan_bool = np.any(np.isnan(weights)) # True if there are nan
      return nan_bool


def nx_replace_nan_edge_weights(G, replacement_value=0):
      '''
      G: nx graph
      replacement_value: value with which np.nans should be replaced
      '''
      edge_labels = nx.get_edge_attributes(G,'weight')
      new_edge_labels = {tup: replacement_value if np.isnan(weight) else weight for tup, weight in edge_labels.items()}
      
      nx.set_edge_attributes(G, new_edge_labels, "weight") # inplace!
      return G


def random_weighted_undirected_graph():
      nkG = nk.generators.MocnikGenerator(dim=2, n=100, k=2, weighted=True) .generate() # Density parameter, determining the ratio of edges to nodes
      # print(nk.overview(nkG))
      nxG = nk.nxadapter.nk2nx(nkG).to_undirected()

      # %%
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


def get_jazz_network():
      print('Jazz network as networkit grap.')
      # Jazz network (unweighted edge list)
      jazz_path = '/home/annina/scripts/great_unread_nlp/src/download.tsv.arenas-jazz/arenas-jazz/out.arenas-jazz'
      # col_names = ['source', 'target'] #see readme of dataset: 'weight', 'timestamp' should also exist but don't
      # df = pd.read_csv(jazz_path, sep='\t', header=0, names=col_names)
      # print(df)
      edgeListReader = nk.graphio.EdgeListReader('\t', 1, '%')
      G = edgeListReader.read(jazz_path)
      return G

def nk_complete_weighted_graph(n):
      G = nx.complete_graph(n)
      G = nk.nxadapter.nx2nk(G)
      # Iterate over the edges of G and add weights
      G = nk.graphtools.toWeighted(G) # weights = 1
      for u, v, w in G.iterEdgesWeights():
            w = random.random()
            G.setWeight(u, v, w)
      return G