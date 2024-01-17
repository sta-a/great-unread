
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
from itertools import groupby
import random
random.seed(9)

from scipy.stats import f_oneway
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from .cluster_utils import CombinationInfo
from .cluster import ClusterBase
import sys
sys.path.append("..")
from utils import DataHandler
import logging
logging.basicConfig(level=logging.WARNING)


class MxIntEval():
    '''
    Evaluate cluster quality based on internal criteria
    '''
    def __init__(self, combination):
        self.mx, self.clusters, _ = combination
        assert self.clusters is not None

    def evaluate(self):
        sc = self.silhouette_score()
        evals = {'silhouette_score': sc}
        return evals

    def silhouette_score(self):
        clusters = self.clusters.df
        assert all(self.mx.dmx.index == clusters['cluster'].index)
        sc = silhouette_score(X=self.mx.dmx, labels=list(clusters['cluster']), metric='precomputed')
        return sc
    

class NkIntEval():
    '''
    Evaluate cluster quality based on internal criteria
    '''
    def __init__(self, combination):
        self.network, self.clusters, self.info = combination
        assert self.clusters is not None
        self.cluster_alg = self.info.cluster_alg
        self.param_comb = self.info.param_comb

    def evaluate(self):
        mod = np.nan
        mod = self.modularity()
        evals = {'modularity': mod}
        return evals

    def modularity(self):
        if 'resolution' in self.param_comb:
            res = self.param_comb['resolution']
        else:
            res = 1 ######################3
        mod = modularity(self.network.graph, self.clusters.initial_clusts, resolution=res)
        return mod



class ExtEval(DataHandler):
    '''
    Evaluate cluster quality with an external criterion (the ground truths)
    '''
    def __init__(self, language, mode, info, inteval):
        super().__init__(language, output_dir='similarity')
        self.mode = mode
        self.info = info
        self.inteval = inteval
        self.cat_attrs = ['gender', 'author']

        self.add_subdir(f'{self.mode}eval')
        self.file_paths = self.get_file_paths()


    def get_file_paths(self):
        paths = {}
        for fn in ['cont', 'cat']:
            paths[fn] = self.get_file_path(file_name=f'{fn}_results.csv', subdir=True)
        return paths


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


    def evaluate(self, attr, info):
        self.info = info
        assert not hasattr(self, 'special_clst_info')
        if attr == 'cluster':
            self.eval_clst()
        else:
            self.eval_attr()


    def eval_clst(self, df=None):

        def get_clst_counts(df):

            # Convert 7,6,6,5,5,5 to 7, 2x6, 3x5
            def replace_repeated_numbers(numbers):
                grouped_numbers = [(key, len(list(group))) for key, group in groupby(numbers)]
                result = [f"{count}x{key}" if count > 1 else str(key) for key, count in grouped_numbers]
                return ', '.join(result)
            
            # Get cluster counts     
            nclust = df['cluster'].nunique()
            clst_sizes = df['cluster'].value_counts().tolist()
            clst_sizes = replace_repeated_numbers(clst_sizes)

            return {'nclust': nclust, 'clst_sizes': clst_sizes}
        
        if df is None:
            # Initial cluster evaluation for whole metadf
            self.default_clst_info = get_clst_counts(self.info.metadf)
        else:
            # Special evaluation if rows have been dropped from metadf due to missing values in the attr column
            self.special_clst_info = get_clst_counts(df)


    def eval_attr(self):
        self.set_params()
        evalscores = self.eval_method()
        self.write_eval(evalscores)
    

    def write_eval(self, evalscores):
        df = self.info.as_dict()
        df.update(evalscores)
        df.update(self.inteval)
        if hasattr(self, 'special_clst_info'):
            df.update(self.special_clst_info)
            del self.special_clst_info
        else:
            df.update(self.default_clst_info)
        df['file_info'] = self.info.as_string()

        df = pd.DataFrame(df, index=[0])
        df = df.round(3)

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
    

    def get_purity(self):
        purities = []
        mdf = self.info.metadf.copy(deep=True)

        # Find the most common label in the cluster
        for cluster in mdf['cluster'].unique():
            df = mdf[mdf['cluster'] == cluster]
            nelements = len(df)

            # Ignore clusters of length 1 for ARI calculation
            if nelements == 1:
                continue

            # Count occurrences of each true label in the cluster
            label_counts = Counter(df[self.info.attr])
            
            # Find the most frequent label in the cluster
            most_frequent_label = max(label_counts, key=label_counts.get)
            
            # Calculate purity for the current cluster
            cluster_purity = label_counts[most_frequent_label] / nelements
            purities.append(cluster_purity)

        return np.mean(purities)
    
    
    def get_categorical_scores(self, attrcol):
        df = self.info.metadf.copy(deep=True)
        assert len(attrcol) == self.nr_texts
        assert len(df['cluster'] == self.nr_texts)

        # confusion_table = pd.crosstab(df[self.info.attr], df['cluster'], margins=True, margins_name='Total')
        # print('\n\n-----------------------------\n', confusion_table, '\n-----------------------------------------\n\n')

        purity = self.get_purity()

        ari_score = adjusted_rand_score(attrcol, df['cluster'] )
        nmi_score = normalized_mutual_info_score(attrcol, df['cluster'] )
        fmi_score = fowlkes_mallows_score(attrcol, df['cluster'] )
        df = {'ARI': ari_score, 'nmi': nmi_score, 'fmi': fmi_score, 'mean_purity': purity}
        return df


    def eval_author(self):
        df = self.info.metadf.copy(deep=True)
        # Create a mapping dictionary
        unique_authors = df['author'].unique()
        author_mapping = {author: i for i, author in enumerate(unique_authors)}

        # Replace strings with numbers using the mapping
        author_col = df['author'].replace(author_mapping)
        scores = self.get_categorical_scores(author_col)
        return scores
    

    def eval_gender(self):
        df = self.info.metadf.copy(deep=True)
        gender_col = df['gender'].replace({'m': 0, 'f': 1, 'a': 0, 'b': 0})
        self.logger.debug(f'Count "b" and "a" gender labels as "m" when evaluating gender.')
        scores = self.get_categorical_scores(gender_col)
        return scores
    

    def filter_attr(self):
        # Filter NaN values in attr column with boolean mask
        assert len(self.info.metadf) == self.nr_texts
        df = self.info.metadf[self.info.metadf[self.info.attr].notna()]

        # Check that there is more than one cluster after filtering
        cb = ClusterBase(language=self.language, mode=self.mode, cluster_alg=None)
        valid = cb.evaluate_clusters(df, self.info, source='eval')

        # Re-evaluate clustering if rows were dropped because of nan in attr column
        if valid:
            if len(self.info.metadf) != self.nr_texts:
                self.eval_clst(df)

        return df, valid
    

    def eval_continuous(self):
        df, valid = self.filter_attr()

        # Run logreg
        # Extract the attr values and reshape
        X = df[self.info.attr].values.reshape(-1, 1)
        y_true = df['cluster'].values.ravel()

        if valid:
            logreg = self.logreg(X, y_true)

            # Run ANOVA
            # Create a list of arrays for each unique integer in 'cluster'
            X_cluster = [df[df['cluster'] == cluster][self.info.attr].values.reshape(-1, 1) for cluster in df['cluster'].unique()]

            anova = self.anova(X_cluster)
            cont_scores = {'anova-pval': anova, 'logreg-accuracy': logreg, 'nr_attr_nan': df[self.info.attr].isna().sum()}
        else:
            self.logger.info(f'Invalid clustering after filtering attr col for nan: {self.info.as_string()}')
            cont_scores = {'anova-pval': 'invalid', 'logreg-accuracy': 'invalid', 'nr_attr_nan': df[self.info.attr].isna().sum()}
        return cont_scores


    def anova(self, X_cluster):
        # Perform ANOVA to evaluate relationship between clustering and continuous variable
        f_statistic, pval = f_oneway(*X_cluster)
        return pval[0]
    
    
    def logreg(self, X, y_true):
        # Multinomial logistic regression to evaluate relationship between clustering and continuous variable
        model = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
        model.fit(X, y_true)
        
        y_pred = model.predict(X)

        plot = False
        if plot: ################################################
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
            plt.scatter(X, y_true, c=y_pred, cmap='Set1', edgecolors='k', marker='o', s=100, label='Clusters from LogReg')

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
    

