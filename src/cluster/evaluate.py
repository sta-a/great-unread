
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

from .cluster_utils import MetadataHandler
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
            mod = self.modularity()
        evals = {'modularity': mod}
        return pd.DataFrame([evals])

    def modularity(self):
        if 'resolution' in self.param_comb:
            res = self.param_comb['resolution']
        else:
            res = 1 ######################3
        mod = modularity(self.network.graph, self.clusters.initial_clusts, resolution=res)
        return round(mod, 3)



class ExtEval(DataHandler):
    '''
    Evaluate cluster quality with an external criterion (the ground truths)
    '''
    def __init__(self, language, mode, viz, clusters, info, inteval, network):
        super().__init__(language, output_dir='similarity')
        self.mode = mode
        self.viz = viz
        self.clusters = clusters
        self.info = info
        self.inteval = inteval
        self.network = network
        self.cat_attrs = ['gender', 'author']

        self.add_subdir(f'{self.mode}eval')
        self.file_paths = self.get_file_paths()

        self.merge_clust_meta_dfs()
        self.eval_clust()


    def get_file_paths(self):
        paths = {}
        for fn in ['cont', 'cat']:
            paths[fn] = self.get_file_path(file_name=f'{fn}_results.csv', subdir=True)
        return paths
        

    def merge_clust_meta_dfs(self):
        if self.clusters is not None:
            # Combine file names, attributes, and cluster assignments
            metadf = pd.merge(self.info.metadf, self.clusters.df, left_index=True, right_index=True, validate='1:1')
            mh = MetadataHandler(self.language)
            # Add color for clusters
            metadf = mh.add_color_to_df(metadf, 'cluster')
            metadf = mh.add_shape_to_df(metadf)

            # Create a new col for categorical attributes that matches a number to every cluster-attribute combination
            # Map the new cols to colors
            for ca in self.cat_attrs:
                colname = f'{ca}_cluster'
                metadf[colname] = metadf.groupby(['gender', 'cluster']).ngroup()
                metadf = mh.add_color_to_df(metadf, colname)

            self.info.metadf = metadf
            self.viz.set_info(self.info)


    def eval_clust(self): 
        if self.clusters is not None:
            # Visualize clusters with heatmap
            # Get nr elements per cluster
            cluster_column = self.info.metadf['cluster']
            value_counts = cluster_column.value_counts()
            nclust = cluster_column.nunique()

            # Count clusters with only one data point
            iso_cluster_count = sum(count == 1 for count in value_counts)
                
            clst_str = ', '.join(f'label{val}-{count}' for val, count in value_counts.items() if count>1)
            if iso_cluster_count > 0:
                clst_str += f', isolated-{iso_cluster_count}'

            self.clst_info = pd.DataFrame({'nclust': nclust,'clst_str': clst_str}, index=[0])
            self.plttitle = f'nclust: {nclust}, {clst_str}'

            _ = self.viz.visualize(pltname='clstviz', plttitle=self.plttitle)

    
    def set_params(self):
        if self.info.attr in self.cat_attrs:
            self.scale = 'cat'

            if self.info.attr == 'gender':
                self.eval_method = self.eval_gender
            elif self.info.attr == 'author':
                self.eval_method = self.eval_author

        else:
            self.scale = 'cont'
            self.eval_method = self.eval_continuous

        self.viz.set_info(self.info)


    def evaluate(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.info, key, value)
        self.set_params()

        if self.clusters is None:
            # self.info.extra = 'one_cluster'
            if self.scale == 'cat':
                empty_df = pd.DataFrame([{'ARI': np.nan}])
                self.write_eval(empty_df)
            else:
                empty_df = pd.DataFrame([[np.nan, np.nan]], columns=['anova-pval', 'logreg-accuracy'])
                self.write_eval(empty_df)

        else:
            evaldf = self.eval_method()

            eetitle = ','.join([f'{col}: {evaldf.iloc[0][col]}' for col in evaldf.columns])
            ietitle = ','.join([f'{col}: {self.inteval.iloc[0][col]}' for col in self.inteval.columns])

            vizdict = self.viz.visualize(pltname='evalviz', plttitle=f'{self.plttitle}\n{eetitle},{ietitle}')
            self.write_eval(evaldf, vizdict)
        
        # Return updated info after finishing evaluation
        return self.info


    def get_existing_fileinfos(self):
        # Return list of all parameter combinations that have already been evaluated
        infos = []
        for path in list(self.file_paths.values()):
            if os.path.exists(path):
                df = pd.read_csv(path, header=0)
                file_info_column = df['file_info'].tolist()
                infos.extend(file_info_column)
        return infos
    

    def write_eval(self, evaldf, vizdict):
        file_info = pd.DataFrame({'file_info': self.info.as_string()}, index=[0])

        all_dfs = [self.info.as_df(), self.clst_info, self.inteval, evaldf]
        if vizdict is not None: 
            vizdf = pd.DataFrame([vizdict])
            all_dfs.append(vizdf)
        all_dfs.append(file_info) # file_info as last column

        df = pd.concat(all_dfs, axis=1)

        # Write header only if file does not exist
        file_path = self.file_paths[self.scale]
        if os.path.exists(file_path):
            mode = 'a'
            header = False
        else:
            mode = 'w'
            header = True
        self.save_data(data=df, 
                       file_path=file_path, 
                       data_type='csv', 
                       pandas_kwargs={'mode': mode, 'header': header, 'index': False, 'na_rep': 'NA'})


    def eval_gender(self):
        df = self.info.metadf.copy(deep=True)

        confusion_table = pd.crosstab(df['gender'], df['cluster'], margins=True, margins_name='Total')
        # print('\n\n-----------------------------\n', confusion_table, '\n-----------------------------------------\n\n')

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
        for cluster in self.info.metadf['cluster'].unique():
            df = self.info.metadf[self.info.metadf['cluster'] == cluster]

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
                    nr_works = len(self.info.metadf[self.info.metadf['author'] == author])
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
        # Filter NaN values in attr column with boolean mask
        df = self.info.metadf[self.info.metadf[self.info.attr].notna()]

        # Run logreg
        # Extract the attr values and reshape
        X = df[self.info.attr].values.reshape(-1, 1)
        y_true = df['cluster'].values.ravel()
        logreg = self.logreg(X, y_true)

        # Run ANOVA
        # Create a list of arrays for each unique integer in 'cluster'
        X_cluster = [df[df['cluster'] == cluster][self.info.attr].values.reshape(-1, 1) for cluster in df['cluster'].unique()]

        anova = self.anova(X_cluster)
        result = [round(anova, 3), round(logreg, 3)]
        result = pd.DataFrame([result], columns=['anova-pval', 'logreg-accuracy'])
        return result


    def anova(self, X_cluster):
        # Perform ANOVA to evaluate relationship between clustering and continuous variable
        f_statistic, pval = f_oneway(*X_cluster)
        return pval[0]
    
    
    def logreg(self, X, y_true):
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
        plt.scatter(X, y_true, c=y_pred, cmap='Set1', edgecolors='k', marker='o', s=100, label='Clusters from LogReg)')

        # Set labels and title
        plt.xlabel(f'{self.info.attr.capitalize()}')
        plt.ylabel('Clusters from Clustering Alg)')
        plt.title('Logistic Regression')

        plt.yticks(np.unique(y_true))

        # Display the legend
        plt.legend()
        self.save_data(data=plt, data_type='png', subdir=True, file_name=f'logreg-{self.info.as_string()}-{self.info.attr}.png')
        plt.close()

        return accuracy_score(y_true=y_true, y_pred=y_pred)
    

