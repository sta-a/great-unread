
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

from scipy.stats import f_oneway, kruskal

from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score, homogeneity_completeness_v_measure, adjusted_rand_score, accuracy_score, balanced_accuracy_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.linear_model import LogisticRegression

from .cluster_utils import CombinationInfo, MetadataHandler
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
    def __init__(self, combination, eval_only=False):
        self.mx, self.clusters, _ = combination
        self.eval_only = eval_only
        assert self.clusters is not None

    def evaluate(self):
        sc = self.silhouette_score()
        evals = {'silhouette_score': sc}
        return evals

    def silhouette_score(self):
        try:
            if not self.eval_only:
                clusters = self.clusters.df
            else:
                clusters = self.clusters
            sc = silhouette_score(X=self.mx.dmx, labels=list(clusters['cluster']), metric='precomputed')
            return sc
        except Exception as e:
            print("Error occurred during silhouette score calculation:")
            print("Error type:", type(e).__name__)
            print("Error message:", str(e))
            
            # Print additional relevant information if available
            if hasattr(self, 'clusters') and hasattr(self.clusters, 'df'):
                print("Clusters DataFrame shape:", self.clusters.df.shape)
            if hasattr(self, 'mx') and hasattr(self.mx, 'dmx'):
                print("Matrix shape:", self.mx.dmx.shape)
                print("Matrix name:", self.mx.name)

            labels=list(clusters['cluster'])
            unique_labels = np.unique(labels)
            print("Unique Cluster Labels:", unique_labels)

            return np.nan
    

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
        mod = self.modularity()
        evals = {'modularity': mod}
        return evals

    def modularity(self):
        if 'resolution' in self.param_comb:
            res = self.param_comb['resolution']
        else:
            res = 1
        mod = modularity(self.network.graph, self.clusters.initial_clusts, resolution=res)
        return mod



class ExtEval(DataHandler):
    '''
    Evaluate cluster quality with an external criterion (the ground truths)
    '''
    def __init__(self, language, cmode, info, inteval, output_dir='similarity', by_author=False):
        super().__init__(language, output_dir=output_dir, by_author=by_author)
        self.cmode = cmode
        self.info = info
        self.inteval = inteval
        self.cat_attrs = ['gender', 'author', 'canon-ascat', 'year-ascat']

        self.add_subdir(f'{self.cmode}eval')
        self.file_paths = self.get_file_paths()

        mh = MetadataHandler(language=self.language, by_author=self.by_author)
        self.full_metadf = mh.get_metadata() # Used for adding isolated nodes if they have been removed


    def get_full_metadf(self):
        df = self.info.metadf.copy(deep=True)
        df_with_iso = pd.concat([df, self.full_metadf[~self.full_metadf.index.isin(df.index)]], axis=0)

        df['gender'] = df['gender'].replace({'m': 0, 'f': 1, 'a': 0, 'b': 0})
        df_with_iso['gender'] = df_with_iso['gender'].replace({'m': 0, 'f': 1, 'a': 0, 'b': 0})
        self.logger.debug(f'Count "b" and "a" gender labels as "m" when evaluating gender.')

        # Create a mapping dictionary, replace strings with numbers using the mapping
        unique_authors = df['author'].unique()
        author_mapping = {author: i for i, author in enumerate(unique_authors)}
        df['author'] = df['author'].replace(author_mapping)

        unique_authors = df_with_iso['author'].unique()
        author_mapping = {author: i for i, author in enumerate(unique_authors)}
        df_with_iso['author'] = df_with_iso['author'].replace(author_mapping)

        # Ensure the placeholder value 9999 is not already in the 'cluster' column
        if 9999 in df_with_iso['cluster'].values:
            placeholder_value = df_with_iso['cluster'].max() + 1  # Find a new unique value
            self.logger.debug(f'9999 is already present in the "cluster" column, using {placeholder_value} instead.')
        else:
            placeholder_value = 9999

        # Fill missing 'cluster' values with the chosen placeholder
        df_with_iso['cluster'] = df_with_iso['cluster'].fillna(placeholder_value)

        return df, df_with_iso
    

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
            elif self.info.attr == 'canon-ascat':
                self.eval_method = self.eval_canon_ascat
            elif self.info.attr == 'year-ascat':
                self.eval_method = self.eval_year_ascat

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
        # Get number of clusters and cluster sizes

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
        weights = []
        mdf, df_with_iso = self.get_full_metadf()

        # Find the most common label in the cluster
        for cluster in mdf['cluster'].unique():
            df = mdf[mdf['cluster'] == cluster]
            nelements = len(df)

            # Count occurrences of each true label in the cluster
            label_counts = Counter(df[self.info.attr])
            
            # Find the most frequent label in the cluster
            most_frequent_label = max(label_counts, key=label_counts.get)
            
            # Calculate purity for the current cluster
            cluster_purity = label_counts[most_frequent_label] / nelements
            purities.append(cluster_purity)
            weights.append(nelements)

        # Calculate weighted purity
        weighted_purity = np.average(purities, weights=weights)

        return weighted_purity
    
    
    # def get_categorical_scores(self, attrcol):
    #     df, df_with_iso = self.get_full_metadf()

    #     # assert len(attrcol) == self.nr_texts
    #     # assert len(df['cluster'] == self.nr_texts)

    #     purity = self.get_purity()
    #     ari_score = adjusted_rand_score(attrcol, df['cluster'])
    #     nmi_score = normalized_mutual_info_score(attrcol, df['cluster'])
    #     ad_nmi_score = adjusted_mutual_info_score(attrcol, df['cluster'])
    #     fmi_score = fowlkes_mallows_score(attrcol, df['cluster'])
    #     homogeneity, completeness, vmeasure = homogeneity_completeness_v_measure(labels_true=attrcol, labels_pred=df['cluster'])
    #     df = {'ARI': ari_score, 'nmi': nmi_score, 'ad_nmi': ad_nmi_score, 'fmi': fmi_score, 'mean_purity': purity, 'homogeneity': homogeneity, 'completeness': completeness, 'vmeasure': vmeasure}
    #     return df

    def get_categorical_scores(self, attrcol):
        df, df_with_iso = self.get_full_metadf()

        # Perform calculations for the first dataframe (df)
        purity_df = self.get_purity()
        ari_score_df = adjusted_rand_score(df[attrcol], df['cluster'])
        nmi_score_df = normalized_mutual_info_score(df[attrcol], df['cluster'])
        ad_nmi_score_df = adjusted_mutual_info_score(df[attrcol], df['cluster'])
        fmi_score_df = fowlkes_mallows_score(df[attrcol], df['cluster'])
        homogeneity_df, completeness_df, vmeasure_df = homogeneity_completeness_v_measure(labels_true=df[attrcol], labels_pred=df['cluster'])

        # Perform calculations for the second dataframe (df_with_iso)
        purity_with_iso = self.get_purity()
        ari_score_with_iso = adjusted_rand_score(df_with_iso[attrcol], df_with_iso['cluster'])
        nmi_score_with_iso = normalized_mutual_info_score(df_with_iso[attrcol], df_with_iso['cluster'])
        ad_nmi_score_with_iso = adjusted_mutual_info_score(df_with_iso[attrcol], df_with_iso['cluster'])
        fmi_score_with_iso = fowlkes_mallows_score(df_with_iso[attrcol], df_with_iso['cluster'])
        homogeneity_with_iso, completeness_with_iso, vmeasure_with_iso = homogeneity_completeness_v_measure(labels_true=df_with_iso[attrcol], labels_pred=df_with_iso['cluster'])

        # Combine results into a non-nested dictionary
        results = {
            'ARI': ari_score_df,
            'nmi': nmi_score_df,
            'ad_nmi': ad_nmi_score_df,
            'fmi': fmi_score_df,
            'mean_purity': purity_df,
            'homogeneity': homogeneity_df,
            'completeness': completeness_df,
            'vmeasure': vmeasure_df,
            'ARI_with_iso': ari_score_with_iso,
            'nmi_with_iso': nmi_score_with_iso,
            'ad_nmi_with_iso': ad_nmi_score_with_iso,
            'fmi_with_iso': fmi_score_with_iso,
            'mean_purity_with_iso': purity_with_iso,
            'homogeneity_with_iso': homogeneity_with_iso,
            'completeness_with_iso': completeness_with_iso,
            'vmeasure_with_iso': vmeasure_with_iso
        }
        
        return results




    def eval_canon_ascat(self):
        df, df_with_iso = self.get_full_metadf()
        scores = self.get_categorical_scores(attrcol='canon-ascat')
        return scores
    
    def eval_year_ascat(self):
        df, df_with_iso = self.get_full_metadf()
        scores = self.get_categorical_scores(attrcol='year-ascat')
        return scores
    
    
    def eval_author(self):
        df, df_with_iso = self.get_full_metadf()
        scores = self.get_categorical_scores(attrcol='author')
        return scores
    

    def eval_gender(self):
        df, df_with_iso = self.get_full_metadf()
        scores = self.get_categorical_scores(attrcol='gender')
        return scores
    

    def filter_attr(self):
        # Filter NaN values in attr column with boolean mask
        # attr columns might contains NaN for stepwise distance and related measures if text is only one chunk long and feature cannot be calculated
        # assert len(self.info.metadf) == self.nr_texts
        df, df_with_iso = self.get_full_metadf()
        df = df[df[self.info.attr].notna()]
        df_with_iso = df_with_iso[df_with_iso[self.info.attr].notna()]
        if self.info.attr == 'canon' or self.info.attr == 'year':
            assert len(df_with_iso) == self.nr_texts

        # Check that there is more than one cluster after filtering
        cb = ClusterBase(language=self.language, cmode=self.cmode, cluster_alg=None, output_dir=self.output_dir, by_author=self.by_author)
        valid = cb.evaluate_clusters(df, self.info, source='eval')

        # Re-evaluate clustering if rows were dropped because of nan in attr column
        if valid:
            if len(self.info.metadf) != self.nr_texts:
                self.eval_clst(df)

        return df, df_with_iso, valid
    

    def eval_continuous(self):
        df, df_with_iso, valid = self.filter_attr()

        if valid:
            # Extract the attr values and reshape
            X = df[self.info.attr].values.reshape(-1, 1)
            y_true = df['cluster'].values.ravel()
            logreg_acc, logrec_acc_balanced = self.logreg(X, y_true)

            sc = self.get_ext_attr_silhouette(X, y_true)
            dbs = self.get_ext_davies_bouldin_score(X, y_true)
            chs = self.get_ext_calinski_harabasz_score(X, y_true)
            avg_variance, weighted_avg_variance, smallest_variance, avg_variance_with_iso, weighted_avg_variance_with_iso, smallest_variance_with_iso = self.get_ext_attr_variance(df, df_with_iso)
            wcss = self.get_wcss(df)

            # Create a list of arrays for each unique integer in 'cluster'
            cluster_groups = self.get_cluster_groups(df)
            anova = self.anova(cluster_groups)
            kw_statistic, kw_pval = self.kruskal(cluster_groups)
            cont_scores = {
                'ext_silhouette': sc,
                'ext_davies_bouldin': dbs,
                'ext_calinski_harabasz': chs,
                'avg_variance': avg_variance,
                'avg_variance_with_iso': avg_variance_with_iso,
                'weighted_avg_variance': weighted_avg_variance,
                'weighted_avg_variance_with_iso': weighted_avg_variance_with_iso,
                'smallest_variance': smallest_variance,
                'smallest_variance_with_iso': smallest_variance_with_iso,
                'ext_wcss': wcss,
                'anova_pval': anova, 
                'logreg_acc': logreg_acc, 
                'logreg_acc_balanced': logrec_acc_balanced, 
                'nr_attr_nan': df[self.info.attr].isna().sum(), 
                'kruskal_statistic': kw_statistic, 
                'kruskal_pval': kw_pval,
                'valid_clsts': True}
        else:
            self.logger.info(f'Invalid clustering after filtering attr col for nan: {self.info.as_string()}')
            i = np.nan
            cont_scores = {
                'ext_silhouette': i,
                'ext_davies_bouldin': i,
                'ext_calinski_harabasz': i,
                'avg_variance': i,
                'avg_variance_with_iso': i,
                'weighted_avg_variance': i,
                'weighted_avg_variance_with_iso': i,
                'smallest_variance': i,
                'smallest_variance_with_iso': i,
                'ext_wcss': i,
                'anova_pval': i, 
                'logreg_acc': i, 
                'logreg_acc_balanced': i, 
                'nr_attr_nan': df[self.info.attr].isna().sum(), 
                'kruskal_statistic': i, 
                'kruskal_pval': i,
                'valid_clsts': False}
        return cont_scores
    

    def get_cluster_groups(self, df):
        # Returns list of arrays, where each array contains the values of the attr for a particular cluster
        cluster_groups = [group[self.info.attr].values for _, group in df.groupby('cluster')]
        return cluster_groups
    

    def kruskal(self, cluster_groups):
        kw_statistic, kw_pval = kruskal(*cluster_groups)
        return kw_statistic, kw_pval


    def anova(self, cluster_groups):
        cluster_groups = [x.reshape(-1, 1) for x in cluster_groups]
        # Perform ANOVA to evaluate relationship between clustering and continuous variable
        f_statistic, pval = f_oneway(*cluster_groups)
        return pval[0]
    
    
    def logreg(self, X, y_true, draw=False, path=None):
        # Multinomial logistic regression to evaluate relationship between clustering and continuous variable
        model = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=1)
        model.fit(X, y_true)
        
        y_pred = model.predict(X)

        if draw:
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
            plt.ylabel('Clusters from Clustering Algorithm')
            plt.title('Logistic Regression')

            plt.yticks(np.unique(y_true))

            acc = round(accuracy_score(y_true=y_true, y_pred=y_pred), 2)
            bal = round(balanced_accuracy_score(y_true=y_true, y_pred=y_pred), 2)

            # Display the legend
            # plt.legend()
            # Create legend entries with label counts
            unique_labels, label_counts = np.unique(y_pred, return_counts=True)
            legend_entries = [f'{label} (count: {count})' for label, count in zip(unique_labels, label_counts)]

            # Display the legend with label counts
            plt.legend(legend_entries)



            self.save_data(data=plt, data_type='png', subdir=True, file_path=os.path.join(path, f'logreg-{self.info.as_string()}_acc{acc}_bal{bal}.png'))
            plt.close()

        return accuracy_score(y_true=y_true, y_pred=y_pred), balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    

    def get_ext_attr_silhouette(self, X, y_true):
        sc = silhouette_score(X=X, labels=y_true)
        return sc
    
    def get_ext_davies_bouldin_score(self, X, y_true):
        return davies_bouldin_score(X, y_true)
    
    def get_ext_calinski_harabasz_score(self, X, y_true):
        return calinski_harabasz_score(X, y_true)


    def get_ext_attr_variance(self, mdf, df_with_iso):
        print(df_with_iso['cluster'])
        min_points = 2  # consider only clusters that have more than 2 elements

        def calculate_variances(df):
            # Get the size of each cluster
            cluster_sizes = df.groupby('cluster').size()

            # Initialize lists to store variance and number of points for valid clusters
            valid_variances = []
            valid_num_points = []

            # Iterate through clusters
            for cluster_label, size in cluster_sizes.items():
                if size > min_points:
                    # Filter dataframe for current cluster
                    cluster_df = df[df['cluster'] == cluster_label]

                    # Calculate variance for current cluster
                    variance = cluster_df[self.info.attr].var()
                    valid_variances.append(variance)
                    valid_num_points.append(size)

            # Convert lists to Series for further analysis
            variances_series = pd.Series(valid_variances, index=cluster_sizes.index[cluster_sizes > min_points])
            num_points_series = pd.Series(valid_num_points, index=cluster_sizes.index[cluster_sizes > min_points])

            # Calculate weighted average of variances
            weighted_avg_variance = (variances_series * num_points_series).sum() / num_points_series.sum()

            # Non-weighted average
            avg_variance = variances_series.mean()

            # Find the smallest variance
            smallest_variance = variances_series.min()

            return avg_variance, weighted_avg_variance, smallest_variance

        # Calculate variances for both mdf and df_with_iso
        avg_variance, weighted_avg_variance, smallest_variance = calculate_variances(mdf)
        avg_variance_with_iso, weighted_avg_variance_with_iso, smallest_variance_with_iso = calculate_variances(df_with_iso)

        # Return results with both normal and _with_iso appended
        return avg_variance, weighted_avg_variance, smallest_variance, avg_variance_with_iso, weighted_avg_variance_with_iso, smallest_variance_with_iso



    def get_wcss(self, mdf):
        # Calculate cluster means
        cluster_means = mdf.groupby('cluster')[self.info.attr].mean()
        
        wcss_value = 0
        
        # Iterate over each cluster
        for cluster_id, cluster_mean in cluster_means.items():
            # Select attr values for the current cluster
            cluster_values = mdf[mdf['cluster'] == cluster_id][self.info.attr]
            # Compute squared differences for the current cluster
            squared_diffs = ((cluster_values - cluster_mean) ** 2).sum()
            
            # Add to WCSS
            wcss_value += squared_diffs
        
        return wcss_value
