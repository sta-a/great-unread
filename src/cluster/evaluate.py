
import pandas as pd
import os
import itertools
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from networkx.algorithms.community import modularity
from copy import deepcopy
from itertools import product
import random
random.seed(9)

from scipy.stats import f_oneway
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from .visualize import MxViz, NkViz
from .cluster_utils import MetadataHandler, ColorMap
import sys
sys.path.append("..")
from utils import DataHandler
import logging
logging.basicConfig(level=logging.DEBUG)


class MxIntEval():
    '''
    Evaluate cluster quality based on internal criteria
    '''
    def __init__(self, mx, clusters):
        self.mx = mx
        self.clusters = clusters # df with file_name and cluster cols

    def evaluate(self):
        if self.clusters is None:
            sc = np.nan
        else:
            sc = self.silhouette_score()
        evals = {'silhouette_score': sc}
        return pd.DataFrame([evals])

    def silhouette_score(self):
        clusters = self.clusters.df
        assert all(self.mx.dmx.index == clusters['cluster'].index)
        sc = silhouette_score(X=self.mx.dmx, labels=list(clusters['cluster']), metric='precomputed')
        return round(sc, 3)
    

class NkIntEval():
    '''
    Evaluate cluster quality based on internal criteria
    '''
    def __init__(self, network, clusters, cluster_alg, param_comb):
        self.network = network
        self.clusters = clusters # df with file_name and cluster cols
        self.cluster_alg = cluster_alg
        self.param_comb = param_comb

    def evaluate(self):
        mod = np.nan
        if self.clusters is not None:
            if self.cluster_alg == 'louvain': #######################################
                mod = self.modularity()
        evals = {'modularity': mod}
        return pd.DataFrame([evals])

    def modularity(self):
        assert self.clusters.type == 'setlist'
        mod = modularity(self.network.graph, self.clusters.initial_clusts, resolution=self.param_comb['resolution'])
        return mod



class ExtEval(DataHandler):
    '''
    Evaluate cluster quality with an external criterion (the ground truths)
    '''
    def __init__(self, language, mode, viz, clusters, info, param_comb, inteval):
        super().__init__(language, output_dir='similarity')
        self.mode = mode
        self.viz = viz
        self.clusters = clusters
        self.info = info
        self.param_comb = param_comb
        self.inteval = inteval
        self.metadf = None

        self.add_subdir('eval')

        self.eval_clust()


    def eval_clust(self):
        # Visualize clusters with heatmap
        # Get nr elements per cluster for plot
        df = self.clusters.df.copy(deep=True)
        self.viz.set_metadf(df)
        value_counts = df['cluster'].value_counts()
        nelements = ', '.join([f'label{val}-{count}' for val, count in value_counts.items()])
        counts = {'nclust': df['cluster'].nunique(), 'nelements': nelements}
        counts = pd.DataFrame([counts])
        self.viz.visualize(plttitle=counts)


    def add_clust_to_metadf(self):
        # Combine file names, attributes, and cluster assignments
        mh = MetadataHandler(language = self.language, attr=self.info.attr)
        metadf = mh.get_metadata()
        metadf = pd.merge(metadf, self.clusters.df, left_index=True, right_index=True, validate='1:1')
        metadf = mh.add_color(metadf, attr='cluster')
        # self.viz.set_metadf(metadf) ######################################
        return metadf
    

    def set_params(self):
        scale = {'gender': 'cat', 'author': 'cat', 'canon': 'cont', 'year': 'cont', 'features': 'cont'}
        eval_methods = {
            'gender': self.eval_gender,
            'author': self.eval_author,
            'canon': self.eval_continuous,
            'year': self.eval_continuous,
            'features': self.eval_continuous,
            }
        file_name = {
            'cat': 'categorical_results.csv',
            'cont': 'continuous_results.csv',
            }

        self.scale = scale[self.info.attr]
        self.eval_method = eval_methods[self.info.attr]
        self.file_name = file_name[self.info.attr]


    def set_info(self, info):
        self.info = info


    def evaluate(self, info):
        self.set_info(info)
        self.set_params()
        self.metadf = self.add_clust_to_metadf()

        if self.clusters is None:
            if self.scale == 'cat':
                empty_df = pd.DataFrame([{'ARI': np.nan}])
                self.write_eval(empty_df)
            else:
                empty_df = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]], columns=['feature', 'plot_name', 'anova-pval', 'logreg-accuracy'])
                self.write_eval(empty_df)

        else:
            evaldf = self.eval_method()
            self.write_eval(evaldf)


    def write_eval(self, evaldf):
        info = self.info.as_df()
        df = pd.concat([info, self.inteval, evaldf], axis=1)

        # if not os.path.exists(path):
        #     with open(path, 'w') as f:
        #         f.write(f"{','.join(df.columns)}\n")

        self.save_data(data=df, 
                       file_name = self.file_name, 
                       subdir=True, data_type='csv', 
                       pandas_kwargs={'mode': 'a', 'header': False, 'index': False, 'na_rep': 'NA'})


    def eval_gender(self):
        df = self.metadf.copy(deep=True)

        confusion_table = pd.crosstab(df['gender'], df['cluster'], margins=True, margins_name='Total')
        print('confusion_table\n\n-----------------------------\n', confusion_table, '\n-----------------------------------------\n\n')

        # Map gender to number
        gender_col = df['gender'].replace({'m': 0, 'f': 1, 'a': 0, 'b': 0})
        self.logger.info(f'Count "b" and "a" gender labels as "m" when evaluating gender.')
        assert len(df['cluster'] == self.nr_texts)

        # Calculate Adjusted Rand Index
        ari_score = adjusted_rand_score(gender_col, df['cluster'])
        return pd.DataFrame([{'ARI': round(ari_score, 3)}])


    
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
        return pd.DataFrame([{'ARI': round(ari_score, 3)}])

    
    def eval_continuous(self):
        results = []

        # If attr is 'features', iterate through all columns of the extracted features df
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
        return result


    def anova(self, X_cluster):
        # Perform ANOVA to evaluate relationship between clustering and continuous variable
        f_statistic, pval = f_oneway(*X_cluster)

        return pval[0]
    
    
    def logreg(self, feature, X, y_true):
        # Multinomial logistic regression to evaluate relationship between clustering and continuous variable
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y_true)
        
        y_pred = model.predict(X)

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

        return accuracy_score(y_true=y_true, y_pred=y_pred)
    

