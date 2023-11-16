
import pandas as pd
import os
import itertools
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
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
import logging
logging.basicConfig(level=logging.DEBUG)


class MxIntEval():
    '''
    Evaluate cluster quality based on internal criteria
    '''
    def __init__(self, mx, clusters, param_comb):
        print(clusters)
        self.mx = mx
        self.clusters = clusters # cluster df with file_name and cluster cols
        self.param_comb = param_comb

        print('clust', self.clusters)

    def evaluate(self):
        if self.clusters is None:
            sc = np.nan
        else:
            sc = self.silhouette_score()
        evals = {'silhouette_score': sc}
        return pd.DataFrame([evals])

    def silhouette_score(self):  
        assert all(self.mx.dmx.index == list(self.clusters['cluster'].index))
        sc = silhouette_score(X=self.mx.dmx, labels=list(self.clusters['cluster']), metric='precomputed')
        return round(sc, 3)



class Eval(MetadataHandler):
    def __init__(self, language, clusters, info, param_comb):
        self.info = info
        super().__init__(language=language, attr=self.info.attr)
        self.clusters = clusters
        self.param_comb = param_comb
        self.test = True


    def add_clustering_to_metadf(self):
        # Combine file names, attributes, and cluster assignments
        metadf = self.get_metadata()
        metadf = pd.merge(metadf, self.clusters, left_index=True, right_index=True, validate='1:1')
        if not self.test:
            assert len(metadf) == self.nr_texts
        return metadf


class MxExtEval(Eval):
    '''
    Evaluate cluster quality with an external criterion (the ground truths)
    '''
    def __init__(self, language, mx, clusters, info, param_comb, inteval):
        super().__init__(language, clusters, info, param_comb)
        self.mx = mx
        self.inteval = inteval
        self.add_subdir('mxeval')

    def evaluate(self):
        if self.clusters is None:
            cat = ['gender', 'author']
            cont = ['canon', 'year', 'features']

            if self.attr in cat:
                empty_df = pd.DataFrame([{'ARI': np.nan}])
                self.write_categorical(empty_df)
            else:
                empty_df = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]], columns=['feature', 'plot_name', 'anova-pval', 'logreg-accuracy'])
                self.write_continuous(empty_df)

        else:
            self.metadf = self.add_clustering_to_metadf()
            self.scv = MxViz(self.language, self.mx, self.clusters, self.info, self.param_comb, self.metadf)
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


    def write_eval(self, df, file_name):
        path = os.path.join(self.subdir, file_name)

        info = self.info.as_df()
        info = pd.concat([info] * len(df), ignore_index=True)
        inteval = pd.concat([self.inteval] * len(df), ignore_index=True)
        df = pd.concat([info, inteval, df], axis=1)

        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(f"{','.join(df.columns)}\n")

        df.to_csv(path, mode='a', header=False, index=False, na_rep='NA')


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
    

    
class NkExtEval(Eval):
    def __init__(self, language, network, clusters, info, param_comb):
        super().__init__(language, clusters, info, param_comb)
        self.network = network
        self.add_subdir('nkeval')
    
    def evaluate(self):
        self.metadf = self.add_clustering_to_metadf()
        nv = NkViz(self.language, self.network, self.info, self.metadf)
        nv.draw_nx()