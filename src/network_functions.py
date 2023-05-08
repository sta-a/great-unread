import networkit as nk
import networkx as nx
import pandas as pd

# Convert pandas adjacency matrix to nk graph
def nk_from_adjacency(mx):
    G = nx.from_pandas_adjacency(mx)
    G.name = 'adjacency-matrix' 

    print('Networkx netwok:-----------------------')
    print(f'Nr. of nodes: {G.number_of_nodes()}, \nNr. of edges: {G.number_of_edges()}, \nis weighted: {nx.is_weighted(G)}, \nis directed: {nx.is_directed(G)}')
    print(f'Nr. of Selfloops: {nx.number_of_selfloops(G)}')
    # for i in G.edges(data=True):
    #     print(i)
    # Convert networkx.Graph to networkit.Graph
    G = nk.nxadapter.nx2nk(G, weightAttr='weight') #weightAttr: The edge attribute which should be treated as the edge weight
    print('Networkit network:------------------------')
    print(nk.overview(G))
    return G