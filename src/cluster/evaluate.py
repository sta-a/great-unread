
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
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from .cluster_utils import CombinationInfo
import sys
sys.path.append("..")
from utils import DataHandler
import logging
logging.basicConfig(level=logging.WARNING)


class MxIntEval():
    '''
    Evaluate cluster quality based on internal criteria
    '''
    def __init__(self, mx, clusters):
        self.mx = mx
        self.clusters = clusters # df with file_name and cluster cols
        assert self.clusters is not None

    def evaluate(self):
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
        assert self.clusters is not None
        self.cluster_alg = cluster_alg
        self.param_comb = param_comb

    def evaluate(self):
        mod = np.nan
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
        if attr == 'cluster':
            self.eval_clst()
        else:
            self.eval_attr()


    def eval_clst(self):         
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

        # Store information to display as plot titles
        self.plttitle = CombinationInfo(clstinfo=f'nclust: {nclust}, {clst_str}')


    def eval_attr(self):
        self.set_params()
        evaldf = self.eval_method()

        self.plttitle.add('exteval', ','.join([f'{col}: {evaldf.iloc[0][col]}' for col in evaldf.columns]))
        self.plttitle.add('inteval', ','.join([f'{col}: {self.inteval.iloc[0][col]}' for col in self.inteval.columns]))
    
        self.write_eval(evaldf)
    

    def write_eval(self, evaldf):
        file_info = pd.DataFrame({'file_info': self.info.as_string()}, index=[0])
        plttitle = pd.DataFrame({'plttitle': self.plttitle.as_string(sep='\n')}, index=[0])
        df = pd.concat([self.info.as_df(), self.clst_info, self.inteval, evaldf, file_info, plttitle], axis=1)

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
        df = pd.DataFrame([{'ARI': round(ari_score, 3), 'nmi': nmi_score, 'fmi': fmi_score, 'mean_purity': purity}])
        df = df.round(3)
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
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
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
    

