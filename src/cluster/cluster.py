from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import linkage, fcluster, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
import pandas as pd
import itertools
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from copy import deepcopy
from itertools import product
from scipy.stats import f_oneway
from networkx import edge_betweenness_centrality as betweenness
from networkx.algorithms.community.centrality import girvan_newman
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import os
import time
import random
random.seed(9)
import sys
sys.path.append("..")
from utils import DataHandler, TextsByAuthor
from .cluster_utils import MetadataHandler
from .create import SimMx

import logging
logging.basicConfig(level=logging.DEBUG)
# Suppress logger messages by 'matplotlib.ticker
# Set the logging level to suppress debug output
ticker_logger = logging.getLogger('matplotlib.ticker')
ticker_logger.setLevel(logging.WARNING)



class SimmxCluster():
    '''
    Clustering on a similarity matrix
    '''
    # ALGS = {
    #     'hierarchical': {
    #         'nclust': [5, 10],
    #         'method': ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'],
    #     },
    #     'spectral': {
    #         'nclust': [5],
    #     },
        # 'kmedoids': {
        #     'nclust': [5],
        # },
    # }

    ALGS = {
        'hierarchical': {
            'nclust': [2],
            'method': ['single'],
        },
        'spectral': {
            'nclust': [2],
        },
        'kmedoids': {
            'nclust': [2],
        },
    }

    def __init__(self, language=None, mx=None, cluster_alg=None):
        self.language = language
        self.mx = mx
        self.cluster_alg = cluster_alg
        self.attr_params = {'gender': 2, 'author': self.get_nr_authors()}


    def get_param_combinations(self):
        # Get params for current cluster alg
        params = self.ALGS[self.cluster_alg]

        # Update params with attribute-specific params
        # params['nclust'].extend(self.attr_params.values()) #########################3

        # Create a list of dicts with format param_name: param_value.
        param_combs = []
        combinations_product = product(*params.values())
        # Iterate through all combinations and pass them to the function
        for combination in combinations_product:
            param_combs.append(dict(zip(params.keys(), combination)))

        return param_combs


    def get_nr_authors(self):
        return len(TextsByAuthor(self.language).nr_texts_per_author)


    def cluster(self, **kwargs):
        cluster_methods = {
            'spectral': self.spectral,
            'hierarchical': self.hierarchical,
            'kmedoids': self.kmedoids,
        }

        if self.cluster_alg in cluster_methods:
            clusters = cluster_methods[self.cluster_alg](**kwargs)
        else:
            raise NotImplementedError(f"Cluster algorithm '{self.cluster_alg}' not implemented.")

        clusters = self.get_cluster_dict(clusters)
        return clusters        
        
    def get_cluster_dict(self, clusters):
        '''
        If result of clustering algorithm is a list with cluster assingments as ints, create a dict where the file names are the keys and the clusters are the values.
        '''
        assert all(isinstance(item, (int, np.int64)) for item in clusters)
        # Change labels so that the biggest cluster has label 0, the second biggest cluster has label 1, and so on.
        counter = Counter(clusters)
        # Create a mapping of integer to its rank based on frequency
        rank_mapping = {value: rank for rank, (value, _) in enumerate(counter.most_common())}
        # Replace each integer in the list with its rank
        clusters = [rank_mapping[value] for value in clusters]


        assert (self.mx.mx.columns == self.mx.mx.index).all()
        clusters = dict(zip(self.mx.mx.index, clusters))

        # # Turn list of ints into list of lists of file names
        # new_clust = []
        # for nr in set(clusters):
        #     bool_list = [1 if el == nr else 0 for el in clusters]
        #     bool_list = pd.Series(bool_list).astype(bool)
        #     fn_list = self.mx.mx.columns[bool_list].tolist()
        #     new_clust.append(fn_list)
        return clusters
    

    def spectral(self, nclust):
        clusters = SpectralClustering(n_clusters=nclust, eigen_solver='arpack', random_state=11, affinity='precomputed', assign_labels='kmeans').fit_predict(self.mx.mx)
        return clusters


    def hierarchical(self, nclust, method): ######################
        # Distance matrix needed as input
        # Use distance matrix for linkage function
        dmx = deepcopy(self.mx)
        assert dmx.is_sim
        dmx.sim_to_dist()
        dmx.dist_to_condensed()

        linkage_matrix = linkage(dmx.mx, method=method)

        # Create flat clusters form hierarchical tree
        flat_clusters = fcluster(linkage_matrix, t=nclust, criterion='maxclust')
        return flat_clusters.tolist()


    def kmedoids(self, nclust):
        # According to kmedoids documentation, a 'kernel matrix' is required as input when metric is 'precomputed'. 
        # 'Kernel matrix' is not defined.
        # It is the same as a distance matrix (see experiment kmedoids-input-check.py)
        dmx = deepcopy(self.mx)
        assert dmx.is_sim
        dmx.sim_to_dist()
        
        kmedoids = KMedoids(n_clusters=nclust, metric='precomputed', method='pam', init='build', random_state=8)
        clusters = kmedoids.fit_predict(dmx.mx)
        return clusters 


class IntEval():
    '''
    Evaluate cluster quality based on internal criteria
    '''
    def __init__(self, mx, clusters, param_comb):
        self.mx = mx
        self.clusters = clusters
        self.param_comb = param_comb

    def evaluate(self):
        sc = self.silhouette_score()
        evals = {'silhouette_score': sc}
        return pd.DataFrame([evals])

    def silhouette_score(self):
        dmx = deepcopy(self.mx)
        assert dmx.is_sim
        dmx.sim_to_dist()
    
        assert all(dmx.mx.index == list(self.clusters.keys()))
        sc = silhouette_score(X=dmx.mx, labels=list(self.clusters.values()), metric='precomputed')
        return round(sc, 3)


class ExtEval(MetadataHandler):
    '''
    Evaluate cluster quality with an external criterion (the ground truths)
    '''
    def __init__(self, language, mx, clusters, info, inteval, param_comb):
        
        self.info = info
        super().__init__(language=language, attr=self.info.attr)
        self.mx = mx
        self.clusters = clusters
        self.inteval = inteval
        self.param_comb = param_comb

        self.metadf = self.add_clustering_to_metadf()
        self.add_subdir('simmx_eval')
        self.scv = SimmxClusterViz(self.language, self.mx, self.clusters, self.info, self.param_comb, self.metadf)

    def evaluate(self):
        eval_methods = {
            'gender': self.eval_gender,
            'author': self.eval_author,
            'canon': self.eval_continuous,
            'year': self.eval_continuous,
            'features': self.eval_continuous,
        }

        if self.attr in eval_methods:
            eval_methods[self.attr]()
        else:
            raise NotImplementedError(f"Evaluation for attribute '{self.attr}' not implemented.")


    def add_clustering_to_metadf(self):
        metadf = self.get_metadata()

        # Map file names to clusters using the index, which is the file_name column
        metadf['cluster'] = metadf.index.map(self.clusters)

        return metadf

    def write_eval(self, df, file_name):
        path = os.path.join(self.subdir, file_name)

        info = self.info.as_df()
        info = pd.concat([info] * len(df), ignore_index=True)
        inteval = pd.concat([self.inteval] * len(df), ignore_index=True)
        df = pd.concat([info, inteval, df], axis=1)

        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(f"{','.join(df.columns)}\n")

        df.to_csv(path, mode='a', header=False, index=False)


    def write_continuous(self, df):
        self.write_eval(df, file_name='continuous_results.csv')

    
    def write_categorical(self, df):
        self.write_eval(df, file_name='categorical_results.csv')


    def eval_gender(self):
        # Evaluate only if there are 2 clusters
        if self.param_comb['nclust'] != 2:
            return

        df = self.metadf

        confusion_table = pd.crosstab(df['gender'], df['cluster'], margins=True, margins_name='Total')
        print('confusion_table\n\n-----------------------------\n', confusion_table, '\n-----------------------------------------\n\n')

        # Map gender to number
        gender_col = df['gender'].replace({'m': 0, 'f': 1, 'a': 0, 'b': 0})
        self.logger.info(f'Count "b" and "a" gender labels as "m" when evaluating gender.')
        assert len(df['cluster'] == self.nr_texts)

        # Calculate Adjusted Rand Index
        ari_score = adjusted_rand_score(gender_col, df['cluster'])

        # Get nr elements per cluster for plot
        value_counts = df['cluster'].value_counts()
        nelements = ', '.join([f'label{val}-{count}' for val, count in value_counts.items()])
        qual = {'nclust': len(self.clusters), 'nelements': nelements, 'ARI': round(ari_score, 3)}
        qual = pd.DataFrame([qual])
        self.scv.draw_heatmap(qual)
        self.write_categorical(qual[['ARI']])
    
    
    def eval_author(self):
        true_labels = []
        pred_labels = []

        # Set dominant label as the true label of the cluster
        for cluster in self.metadf['cluster'].unique():
            df = self.metadf[self.metadf['cluster'] == cluster]

            # Ignore clusters of length 1 for ARI calculation
            if len(df) == 1:
                continue

            # Find most common author in cluster
            # If a cluster has length 1 but the author has more than one work: ignored
            # Count occurrences of each author
            counter = Counter(df['author'].tolist())
            # Find the most common author
            most_common_count = counter.most_common(1)[0][1]
            # Find all strings with the most common count
            most_common_authors = [item[0] for item in counter.items() if item[1] == most_common_count]


            # If several authors have the same count, assing cluster to the author with the smalles number of works (the author with the smallest class probability)
            if len(most_common_authors) != 1:
                min_nr_works = float('inf')
                min_author = []
                for author in most_common_authors:
                    nr_works = len(self.metadf[self.metadf['author'] == author])
                    if nr_works < min_nr_works:
                        min_nr_works = nr_works
                        min_author = [author]
                    elif nr_works == min_nr_works:
                        min_author.append(author)
                # If several have same class probabilty, chose randomly
                if len(min_author) > 1:
                    most_common_author = random.choice(min_author)
                else:
                    most_common_author = min_author[0]

            else:
                most_common_author = most_common_authors[0]

            for _ in range(0, len(df)):
                true_labels.append(most_common_author)
            pred_labels.extend(df['author'].tolist())

        assert len(true_labels) == len(pred_labels)

        ari_score = adjusted_rand_score(true_labels, pred_labels)
        qual = {'nclust': len(self.clusters), 'ARI': round(ari_score, 3)}
        qual = pd.DataFrame([qual])
        self.scv.draw_heatmap(qual)
        self.write_categorical(qual[['ARI']])

    
    def eval_continuous(self):
        results = []

        for col_name in self.metadf.columns:
            if col_name == 'cluster':
                continue  # Skip the 'cluster' column itself

            # Filter NaN values in 'col_name' with boolean mask
            df = self.metadf[self.metadf[col_name].notna()]

            # Run logreg
            # Extract the 'col_name' values and reshape
            X = df[col_name].values.reshape(-1, 1)
            y = df['cluster'].values.ravel()
            logreg = self.logreg(col_name, X, y)

            # Run ANOVA
            # Create a list of arrays for each unique integer in 'cluster'
            X_cluster = [df[df['cluster'] == cluster][col_name].values.reshape(-1, 1) for cluster in df['cluster'].unique()]

            anova = self.anova(X_cluster)

            results.append((col_name, f'logreg-{self.info.as_string()}-{col_name}.png', round(anova, 3), round(logreg, 3)))

        result = pd.DataFrame(results, columns=['feature', 'plot_name', 'anova-pval', 'logreg-accuracy'])
        self.write_continuous(result)


    def anova(self, X_cluster):
        # Perform ANOVA to evaluate relationship between clustering and continuous variable
        f_statistic, pval = f_oneway(*X_cluster)

        return pval[0]
    
    
    def logreg(self, feature, X, y):
        # Multinomial logistic regression to evaluate relationship between clustering and continuous variable
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        y_pred = model.predict(X)

        self.scv.draw_logreg(feature, model, X, y, y_pred)

        return accuracy_score(y_true=y, y_pred=y_pred)
    

class MxReorder():
    '''Sort row and column indices so that clusters are visible in heatmap.'''

    ORDERS = ['fn', 'olo']

    def __init__(self, language, mx, info, metadf):
        self.language = language
        self.mx = mx
        self.info = info
        self.metadf = metadf
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)


    def order(self):
        order_methods = {
            'fn': self.order_fn,
            'olo': self.order_olo,
        }

        if self.info.order in order_methods:
            ordmx = order_methods[self.info.order]()
            if isinstance(ordmx, SimMx):
                ordmx = ordmx.mx
        else:
            raise ValueError(f"Invalid order value: {self.info.order}")

        assert self.mx.mx.shape == ordmx.shape
        assert self.mx.mx.equals(self.mx.mx.T)
        assert ordmx.index.equals(ordmx.columns), 'Index and columns of ordmx must be equal.'

        return ordmx

    def order_fn(self):
        # Sort rows and columns of each cluster according to file name, which starts with the name of the author
        ordmxs = []

        # Get index labels belonging to the current cluster
        for cluster in self.metadf['cluster'].unique():
            file_names = self.metadf[self.metadf['cluster'] == cluster].index.tolist()

            df = self.mx.mx.loc[:, file_names].sort_index(axis=1)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=1)

        ordmxs = []
        for cluster in self.metadf['cluster'].unique():
            file_names = self.metadf[self.metadf['cluster'] == cluster].index.tolist()
            df = ordmx.loc[file_names, :].sort_index(axis=0)
            ordmxs.append(df)
        ordmx = pd.concat(ordmxs, axis=0)
        return ordmx

    
    def order_olo(self):
        # Use distance matrix for linkage function
        dmx = deepcopy(self.mx)
        assert dmx.is_sim
        dmx.sim_to_dist()

        sorted_fns = []

        # Get unique cluster lables, sorted from rarest to most common label
        unique_clust = self.metadf['cluster'].value_counts().sort_values().index.tolist()

        # Iterate over unique clusters
        for cluster in unique_clust:
            # Extract file names for the current cluster
            file_names = self.metadf[self.metadf['cluster'] == cluster].index.tolist()

            if len(file_names) <=2:
                sorted_fns.append(file_names)

            # Get OLO for cur rent cluster
            else:
                # Subset the similarity matrix for the current cluster
                cmx = dmx.mx.loc[file_names, file_names]
                sq_cmx = squareform(cmx)
                
                cluster_linkage = linkage(sq_cmx, method='average')
                
                order = leaves_list(optimal_leaf_ordering(cluster_linkage, sq_cmx))
                # Map integer indices to string indices
                ordered_fn = dmx.mx.index[order].tolist()
                sorted_fns.append(ordered_fn)
        
        sorted_fns = [fn for clst_fns in sorted_fns for fn in clst_fns]
        ordmx = dmx.mx.loc[sorted_fns, sorted_fns]

        # Convert back to similarity
        ordmx = SimMx(self.language, name='olo', mx=ordmx, normalized=True, is_sim=False, is_directed = self.mx.is_directed, is_condensed=False)
        ordmx.dist_to_sim()

        nr_texts = DataHandler(self.language).nr_texts
        assert (ordmx.mx.shape[0] == nr_texts) and (ordmx.mx.shape[1] == nr_texts) 

        # Reorder the final matrix based on the optimal order of clusters ############################

        self.logger.info(f'OLO matrix reorder.')
        return ordmx




class SimmxClusterViz(MetadataHandler):
    def __init__(self, language, mx, clusters, info, param_comb, metadf):
        self.info = info
        super().__init__(language=language, attr=self.info.attr)

        self.mx = mx
        self.clusters = clusters
        self.param_comb = param_comb
        self.metadf = metadf
        self.add_subdir()


    def draw_heatmap(self, qual):
        ordmx = MxReorder(self.language, self.mx, self.info, self.metadf).order()
 
        # hot_r, viridis, plasma, inferno
        # ordmx = np.triu(ordmx) ####################
        plt.imshow(ordmx, cmap='plasma', interpolation='nearest')

        # Add a color bar to the heatmap for better understanding of the similarity values
        plt.colorbar()

        # Add axis labels and title (optional)
        # plt.xlabel('Data Points')
        # plt.ylabel('Data Points')

        title = ', '.join([f'{col}: {val}' for col, val in qual.iloc[0].items()])

        plt.title(title, fontsize=8)

        self.save_data(data=plt, data_type='png', subdir=True, file_name=f'heatmap-{self.info.as_string()}.png')
        plt.close()


    def draw_logreg(self, feature, model, X, y_true, y_pred):
        # Visualize the decision boundary
        plt.figure(figsize=(10, 6))
        plt.grid(True)

        # Generate a range of values for X for plotting the decision boundary
        X_range = np.linspace(min(X), max(X), 300).reshape(-1, 1)
        # Predict the corresponding y values for the X_range
        y_range = model.predict(X_range)

        # Plot the decision boundary
        plt.plot(X_range, y_range, color='red', linewidth=3, label='Decision Boundary')

        # Scatter plot for the data points
        plt.scatter(X, y_true, c=y_pred, cmap='Set1', edgecolors='k', marker='o', s=100, label='Clusters (logreg)')

        # Set labels and title
        plt.xlabel(f'{feature.capitalize()}')
        plt.ylabel('Clusters (Cluster Alg)')
        plt.title('Logistic Regression')

        plt.yticks(np.unique(y_true))

        # Display the legend
        plt.legend()
        self.save_data(data=plt, data_type='png', subdir=True, file_name=f'logreg-{self.info.as_string()}-{feature}.png')
        plt.close()
    


class NetworkCluster():
    ALGS = ['gn'] #######################, 'louvain']

    def __init__(self, language, graph=None, cluster_alg=None):
        self.language = language
        self.graph = graph
        self.cluster_alg = cluster_alg
        self.attr_params = {'gender': 2, 'author': self.get_nr_authors()}
    
    def find_louvain_resolution(self):
        # Does not work for 2 clusters
        n=1000
        resolution = 10
        while n > 2:
            resolution /= 10
            clusters = nx.community.louvain_communities(self.graph, weight='weight', seed=111, resolution=resolution)
            n = len(clusters)
            print(f'{len(clusters)} clusters after Louvain clustering with resolution {resolution}.')
        return clusters

    def get_gn(self):
        # girvan_newman

        def most_central_edge(graph):
            # Utilize edge weights when choosing an edge with the highest betweenness centrality
            centrality = betweenness(graph, weight="weight")
            return max(centrality, key=centrality.get)
        
        comp = girvan_newman(self.graph, most_valuable_edge=most_central_edge)
        # limited = itertools.takewhile(lambda clusters: len(clusters) == self.attr_params[self.attr], comp)
        # Problem: minimum number of clusters bigger than predefined cluster sizes
        for communities in itertools.islice(comp, 1):
            clusters = tuple(sorted(clusters) for clusters in communities)
        return clusters

    def cluster(self):
        # if self.cluster_alg == 'louvain':
        #     if self.attr == 'gender':
        #         clusters = self.find_louvain_resolution()
        s = time.time()
        if self.cluster_alg == 'louvain':
            clusters = nx.community.louvain_communities(self.graph, weight='weight', seed=111, resolution=1)
        elif self.cluster_alg == 'gn':
            clusters = self.get_gn()
        print(f'{time.time()-s}s to get {self.cluster_alg} clusers.')
        self.clusters = clusters

    def get_nr_authors(self):
        return len(TextsByAuthor(self.language).nr_texts_per_author)




# class SimmxClusterViz(DataHandler):
#     def _relabel_axis(self):
#         labels = self.ax.get_ymajorticklabels()
#         for label in labels:
#             color = self.file_group_mapping.loc[self.file_group_mapping['file_name'] ==label.get_text(), 'group_color']
#             label = label.set_color(str(color.values[0]))


#     def draw_mds(self, clusters):
#         print(f'Drawing MDS.')
#         df = MDS(n_components=2, dissimilarity='precomputed', random_state=6, metric=True).fit_transform(self.mx)
#         df = pd.DataFrame(df, columns=['comp1', 'comp2'], index=self.mx.index)
#         df = df.merge(self.file_group_mapping, how='inner', left_index=True, right_on='file_name', validate='one_to_one')
#         df = df.merge(clusters, how='inner', left_on='file_name', right_index=True, validate='1:1')

#         def _group_cluster_color(row):
#             color = None
#             if row['group_color'] == 'b' and row['cluster'] == 0:
#                 color = 'darkblue'
#             elif row['group_color'] == 'b' and row['cluster'] == 1:
#                 color = 'royalblue'
#             elif row['group_color'] == 'r' and row['cluster'] == 0:
#                 color = 'crimson'
#             #elif row['group_color'] == 'r' and row['cluster'] == 0:
#             else:
#                 color = 'deeppink'
#             return color

#         df['group_cluster_color'] = df.apply(_group_cluster_color, axis=1)


#         fig = plt.figure(figsize=(5,5))
#         ax = fig.add_subplot(1,1,1)
#         plt.scatter(df['comp1'], df['comp2'], color=df['group_cluster_color'], s=2, label="MDS")
#         plt.title = self.plot_name
#         self.save(plt, 'kmedoids-MDS', dpi=500)


