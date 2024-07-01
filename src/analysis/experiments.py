import sys
sys.path.append("..")
from copy import deepcopy
import pandas as pd
import time
import os
import pickle
import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable



from utils import DataHandler
from .mxviz import MxAttrGridViz, MxSingleViz, MxKeyAttrViz, S2vKeyAttrViz
from .nkviz import NkKeyAttrViz, NkAttrGridViz, NkNetworkGridkViz, SparsGridkViz, NkSingleViz
from .viz_utils import ClusterAuthorGrid
from .embedding_eval import EmbMxCombinations
from .topeval import TopEval
from cluster.network import NXNetwork
from cluster.combinations import MxCombinations

# import logging
# logging.basicConfig(level=logging.DEBUG)


class Experiment(DataHandler):


    def __init__(self, language, cmode, by_author=False, output_dir='analysis'):
        assert output_dir in ['analysis', 'analysis_s2v']
        super().__init__(language, output_dir=output_dir)
        self.cmode = cmode
        self.by_author = by_author

        # Use embeddings
        if 's2v' in self.output_dir:
            self.mc = EmbMxCombinations(self.language, by_author=self.by_author)
        # Use distances
        else:
            self.mc = MxCombinations(self.language, by_author=self.by_author)



    def get_experiments(self, select_exp=None):
        # Default values
        maxsize = 0.9
        embmxs = ['both', 'full'] # embedding-based distance matrices
        if self.cmode == 'mx':
            int_evalcol = 'silhouette_score'
        else:
            int_evalcol = 'modularity'
        cont_attrs = ['canon', 'year']
        cat_attrs = ['gender', 'author', 'canon-ascat', 'year-ascat']
        by_author_attrs = ['canon-max', 'canon-min']
        if self.by_author:
            cont_attrs = cont_attrs + by_author_attrs
            cat_attrs.remove('author')



        '''
        Top: Find combinations with highest evaluation scores
        '''
        # Different evaluation metrics for external continuous attribute
        cont_ext_evalcols = [
            'ext_silhouette',
            'ext_davies_bouldin',
            'ext_calinski_harabasz',
            'avg_variance',
            'weighted_avg_variance',
            'smallest_variance',
            'ext_wcss']
            # 'anova_pval',
            # 'kruskal_statistic',
            # 'kruskal_pval'] # logreg

        # Different evaluation metrics for external categorical attribute
        cat_ext_evalcols = [
            'ARI',
            'nmi',
            'fmi',
            'mean_purity',
            'homogeneity',
            'completeness',
            'vmeasure']

        def add_to_top(d):
            d['maxsize'] = maxsize
            d['intthresh'] = 0.3 #############
            d['intcol'] = int_evalcol
            d['viztype'] = 'keyattr'
            return d
        
        def add_evalcols(evalcols_list, basic_dict):
            newlist = []
            for evalcol in evalcols_list:
                d = deepcopy(basic_dict)
                d['evalcol'] = evalcol
                d['name'] = f"{d['name']}_{evalcol}"
                newlist.append(d)
            return newlist

        # Highest scores over all attributes
        topcont_dict = {'name': 'topcont', 'dfs': ['cont']}
        topcont_dict = add_to_top(topcont_dict)
        topcont = add_evalcols(cont_ext_evalcols, topcont_dict)

        topcat_dict = {'name': 'topcat', 'dfs': ['cat']}
        topcat_dict = add_to_top(topcat_dict)
        topcat = add_evalcols(cat_ext_evalcols, topcat_dict)



        # Find combinations with highest evaluation scores for interesting attributes
        attrcont = []
        for attr in cont_attrs:
            for cdict in topcont:
                d = deepcopy(cdict)
                d['name'] = d['name'].replace('cont', attr)
                d['attr'] = attr
                attrcont.append(d)
        attrcat = []
        for attr in cat_attrs:
            for cdict in topcat:
                d = deepcopy(cdict)
                d['name'] = d['name'].replace('cat', attr)
                d['attr'] = attr
                attrcat.append(d)


        # Get best performance of embedding distances
        topcont_emb = []
        for cdict in topcont:
            d = deepcopy(cdict)
            d['name'] = d['name'] + '_emb'
            d['mxname'] = embmxs
            topcont_emb.append(d)
        topcat_emb = []
        for cdict in topcat:
            d = deepcopy(cdict)
            d['name'] = d['name'] + '_emb'
            d['mxname'] = embmxs
            topcat_emb.append(d)
        
        all_top = topcont + topcat + attrcont + attrcat + topcont_emb + topcat_emb


        # The same as all_top, but set the minimum and maximum number of clusters
        # Only for s2v, ignore embmxs
        def modify_top_nclust(d): #################################
            d['name'] = f"{d['name']}_nclust"
            d['min_nclust'] = 4
            # d['max_clust'] = 
            # del d['maxsize']
            # del d['intthresh']
            # del d['intcol']
            return d
        all_top_nclust = deepcopy(topcont + topcat + attrcont + attrcat)
        all_top_nclust = [modify_top_nclust(x) for x in all_top_nclust]
        

        '''
        Many networks on the same figure.
        '''
        # One network, all attributes
        # cat: it doesn't matter which df is chosen
        # In Topeval, a single combination for each sparsified matrix is chosen, attributes don't matter
        attrgrid = [{'name': 'attrgrid', 'dfs': ['cat'], 'mxname': ['burrows'] + embmxs, 'viztype': 'attrgrid'}]
        attrgrid_int = [{'name': 'attrgrid_int', 'dfs': ['cat'], 'intcol': int_evalcol, 'viztype': 'attrgrid'}]

        all_attrgrid = attrgrid + attrgrid_int

        # One attribute, all networks
        nkgridcont = []
        for x in attrcont:
            d = deepcopy(x)
            d['name'] = d['name'].replace('top', 'nkgrid')
            d['viztype'] = 'nkgrid'
            del d['evalcol']
            del d['dfs']
            del d['maxsize']
            del d['intthresh']
            del d['intcol']
            nkgridcont.append(d)
        nkgridcat = []
        for x in attrcat:
            d = deepcopy(x)
            d['name'] = d['name'].replace('top', 'nkgrid')
            d['viztype'] = 'nkgrid'
            del d['evalcol']
            del d['dfs']
            del d['maxsize']
            del d['intthresh']
            del d['intcol']
            nkgridcat.append(d)

        
        all_nkgrid = nkgridcat + nkgridcont

        sparsgrid = [{'name': 'sparsgrid', 'viztype': 'sparsgrid', 'attr': 'canon'}]
        singleimage = [{'name': 'singleimage', 'viztype': 'singleimage', 'attr': 'canon'}]

        # Make single images where clusters are highlighted
        # cat df, author attr don't matter, one combination is chosen
        #  'intthresh': 0.3: don't filter, create all images first
        top_cluster = [{'name': 'top_cluster', 'viztype': 'top_cluster',  'dfs': ['cat'], 'attr': 'author', 'maxsize': maxsize, 'evalcol': int_evalcol}] # sort vy evalcol


        '''
        Consistent clusters and centralities
        '''
        clustconst = []
        central = []
        if not self.by_author:
            clustconst = [{'name': 'clustconst', 'maxsize': maxsize, 'dfs': ['cat'], 'attr': ['author']}]
            central = deepcopy(clustconst)
            central[0]['name'] = 'central'
            central[0]['mxname'] = ['burrows'] + embmxs
            

        # exps = top_cluster + all_top + singleimage
        exps = all_top + singleimage # sparsgrid + all_nkgrid + all_attrgrid + clustconst + central ######################
        if select_exp is not None:
            [print(x['name']) for x in exps]
            exps = [x for x in exps if x['name'] == select_exp]

        for e in exps:
            print(e['name'])
        return exps


    def run_experiments(self, select_exp=None, ntop=30):
        exps = self.get_experiments(select_exp)
        for exp in exps:
            print(f"------------------{exp['name']}-------------------\n")
            if 'ntop' not in exp:
                exp['ntop'] = ntop

            self.add_subdir(f"{self.cmode}_{exp['name']}")

            if (exp['viztype'] == 'nkgrid') or (exp['viztype'] == 'sparsgrid') or (exp['viztype'] == 'singleimage'):
                te = None
            else:
                te = TopEval(self.language, output_dir=self.mc.output_dir, cmode=self.cmode, exp=exp, expdir=self.subdir, by_author=self.by_author)

            if exp['name'] == 'clustconst':
                self.run_clustconst(exp, te)
            elif exp['name'] == 'central':
                if self.cmode == 'nk':
                    self.run_central(exp, te)
                    
            # Visualization is independent of cmode
            elif exp['name'] == 'top_cluster':
                viz = ClusterAuthorGrid(self.language, cmode=self.cmode, exp=exp, te=te, by_author=self.by_author, output_dir=self.output_dir, subdir=self.subdir)
                viz.visualize()
            else:
                self.visualize(exp, te)


    def run_central(self, exp, te):
        df = Central(self.language, self.cmode, exp, te).run()
        centralities = df.columns
        for centrality in centralities:
            exp['evalcol'] = centrality
            te = TopEval(self.language, self.mc.output_dir, self.cmode, exp, expdir=self.subdir, df=df, by_author=self.by_author)
            self.visualize_nk(exp, te, vizname=centrality)


    def run_clustconst(self, exp, te):
        ClusterComparison(self.language, self.cmode, exp, te).run()


    def visualize(self, exp, te):
        if self.cmode == 'mx':
            self.visualize_mx(exp, te)
        else:
            self.visualize_nk(exp, te)


    def visualize_mx(self, exp, te, vizname='viz'):
        # viztypes: 'attrgrid', 'nkgrid' 'keyattr'
        if exp['viztype'] == 'nkgrid':
            pass
        elif exp['viztype'] == 'sparsgrid':
            pass
        elif exp['viztype'] == 'singleimage':
            viz = MxSingleViz(self.language, self.output_dir, exp, self.by_author, self.mc)
            viz.visualize()
        else:
            for topk in te.get_top_combinations():
                info, plttitle = topk
                
                # Get matrix
                mx = self.mc.load_single_mx(mxname=info.mxname)
                info.add('order', 'olo')
                if exp['viztype'] == 'attrgrid':
                    viz = MxAttrGridViz(self.language, self.output_dir, mx, info, plttitle=plttitle, exp=exp)
                else:
                    if 's2v' in self.output_dir:
                        viz = S2vKeyAttrViz(self.language, mx, info, plttitle, exp, by_author=self.by_author, subdir=self.subdir)
                    else:
                        viz = MxKeyAttrViz(self.language, self.output_dir, mx, info, plttitle=plttitle, exp=exp, by_author=self.by_author)
                viz.visualize(vizname)


    def visualize_nk(self, exp, te):
        # viztypes: 'attrgrid', 'nkgrid' 'keyattr'
        if exp['viztype'] == 'nkgrid':
            viz = NkNetworkGridkViz(self.language, self.output_dir, exp, self.by_author)
            viz.visualize()
        elif exp['viztype'] == 'sparsgrid':
            viz = SparsGridkViz(self.language, self.output_dir, exp, self.by_author)
            viz.visualize()
        elif exp['viztype'] == 'singleimage':
            viz = NkSingleViz(self.language, self.output_dir, exp, self.by_author)
            viz.visualize()
        else:
            for topk in te.get_top_combinations():
                info, plttitle = topk
                # name of 'threshold-0%9' has changed to 'threshold-0%90' after combinations were run
                if '0%9.' in info.spmx_path:
                    info.spmx_path = info.spmx_path.replace('0%9.', '0%90.') ##################
                cluster_path_string = '/cluster/scratch/stahla/data'
                if cluster_path_string in info.spmx_path:
                    info.spmx_path = info.spmx_path.replace(cluster_path_string, '/home/annina/scripts/great_unread_nlp/data')
                if exp['viztype'] == 'attrgrid':
                    # In Topeval, a single combination for each sparsified matrix is chosen, attributes don't matter
                    viz = NkAttrGridViz(self.language, self.output_dir, info, plttitle=plttitle, exp=exp, by_author=self.by_author)                      
                else:
                    viz = NkKeyAttrViz(self.language, self.output_dir, info, plttitle=plttitle, exp=exp, by_author=self.by_author)
                viz.visualize()




class ClusterComparison(DataHandler):
    def __init__(self, language, cmode, exp, te):
        super().__init__(language, output_dir='analysis')
        self.cmode = cmode
        self.exp = exp
        self.te = te
        self.add_subdir(f"{self.cmode}_{self.exp['name']}")


    def run(self):
        dfdict= self.collect_clusters()
        for name, df in dfdict.items():
            self.create_heatmap(name, df)


    def count_matching_columns(self, df):
        '''
        Count the number of columns where the values of two rows are the same for each pair of rows in the df.
        '''
        def matching_columns(row1, row2):
            return (row1 == row2).sum()

        distances = pairwise_distances(df.values, metric=matching_columns)
        np.fill_diagonal(distances, 0)
        df = pd.DataFrame(distances, index=df.index, columns=df.index)
        assert df.equals(df.T)
        return df


    def collect_clusters(self):
        # Collect cluster assingments for each distance (+ sparsification + params) + cluster alg + params combination
        path = self.get_file_path(file_name='allclst.csv', subdir=True)
        if not os.path.exists(path):
            cluster_cols = {}
            for topk in self.te.get_top_combinations():
                info, plttitle = topk
                cluster_cols[info.as_string()] = info.metadf['cluster']

            dfa = None
            for key, col in cluster_cols.items():
                col = col.to_frame().rename(columns={col.name: key})
                if dfa is None:
                    dfa = col
                else:
                    dfa = dfa.merge(col, left_index=True, right_index=True, validate='1:1')
            self.save_data(data=dfa, data_type='csv', file_path=path, pandas_kwargs={'index': True})
        else:
            dfa = pd.read_csv(path, index_col='file_name')
            dfa = dfa.astype(int)


        # Compare shared clusters between texts
        path = self.get_file_path(file_name='sharedclst.csv', subdir=True)
        if not os.path.exists(path):
            dfs = self.count_matching_columns(dfa)
            self.save_data(data=dfs, data_type='csv', file_path=path, pandas_kwargs={'index': True})
        else:
            dfs = pd.read_csv(path, index_col='file_name')
            dfs = dfs.astype(int)
        
        # Keep only top 10 % of values
        assert dfs.equals(dfs.T)

        f = np.array(dfs.values.flatten())
        cutoff = np.percentile(f, 90)

        dfcut = dfs.copy()
        assert dfcut.equals(dfcut.T)
        dfcut[dfcut < cutoff] = 0 
        assert dfcut.equals(dfcut.T)
        return {'allclst': dfa, 'sharedclst': dfs, 'shared_ordered_clst': deepcopy(dfs), 'dfcut': dfcut, 'dfcut_ordered': deepcopy(dfcut)}


    def order_mx(self, mx):
        # Order the rows and columns of a symmetric matrix according to optimal leaf ordering.
        # Compute hierarchical clustering for rows
        sqmx = squareform(mx)
        lk = linkage(sqmx, method='average')

        # Compute optimal leaf ordering for rows
        order = leaves_list(optimal_leaf_ordering(lk, sqmx))
        ordered_fns = mx.index[order].tolist()
        ordmx = mx.loc[ordered_fns, ordered_fns]

        return ordmx


    # def map_categorical(self, col):
    #     # Count the occurrences of each unique value in the specified column
    #     value_counts = col.value_counts()

    #     # Create an iterator that cycles through colors
    #     colors = iter(cc.glasbey_bw_minc_20) # color palette with no greys

    #     # If an element occurs only once, set it to dark grey
    #     dark_grey =  [0.7, 0.7, 0.7] # RGB for dark gray
    #     color_mapping = {
    #         element: next(colors) if count > 1 else dark_grey
    #         for element, count in value_counts.items()
    #     }
    #     return col.map(color_mapping)


    def map_categorical(self, df, name):

        def iso_to_none(col):
            value_counts = col.value_counts()

            # Find values that occur only once
            single_occurrences = value_counts.index[value_counts == 1]

            # Set values that occur only once to NaN
            col.loc[col.isin(single_occurrences)] = np.nan
            return col
        
        # For visualizing all cluster assingments per text, set the clusters with count 1
        if name == 'allclst':
            df = df.apply(iso_to_none, axis=0)
        norm = Normalize(vmin=0, vmax=df.max().max())
        cmap = ScalarMappable(norm=norm, cmap='tab20b')

       # Map integer values to colors using the colormap
        nan_color=(0.7, 0.7, 0.7, 1)
        nan_color = (1, 1, 1, 1) ########################

        def map_to_color(col):
            # Nan values (isolated clusters) are grey
            color_mapping = {
                x: nan_color if pd.isna(x) else (1, 1, 1, 1) if x == 0 else cmap.to_rgba(x)
                for x in col.unique()
            }
            return col.map(color_mapping)
        
        df = df.apply(map_to_color)
        return df, cmap
            

    def create_heatmap(self, name, df):
        if name == 'shared_ordered_clst' or name=='dfcut_ordered':
            df = self.order_mx(df)

        df_color, cmap = self.map_categorical(df, name)
        array = np.array(df_color.values.tolist())

        # Plot the colors
        height = 10
        width = (df.shape[1]/df.shape[0]) * height
        fig, ax = plt.subplots(figsize=(width, height))
        ax.imshow(array, aspect='equal')

        # ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)
        cbar = fig.colorbar(cmap, ax=ax)

        self.save_data(data=plt, subdir=True, data_type='svg', file_name=f'{name}.svg')
        # plt.show()




class Central(DataHandler):
    def __init__(self, language, cmode, exp, te):
        super().__init__(language, output_dir='analysis', data_type='pkl')
        self.cmode = cmode
        self.exp = exp
        self.te = te
        self.add_subdir(f"{self.cmode}_{self.exp['name']}")
        self.pkl_path = self.get_file_path(subdir=True, file_name=f'centralities.pkl')
        self.csv_path = self.get_file_path(subdir=True, file_name=f'centralities.csv', data_type='csv')
    

    def run(self):
        df = self.load_correlations()
        df = df.merge(self.te.df, how='inner', left_index=True, right_on='file_info', validate='1:1')
        return df


    def create_correlations(self):
        cents = self.load_centralities()
        canoncol = self.te.metadf['canon']
        corrs = {}
        for centrality, df in cents.items():
            correlations = df.apply(lambda col: col.corr(canoncol))
            corrs[centrality] = correlations.round(2)

        df = pd.DataFrame(corrs)
        self.save_data(data=df, file_path=self.csv_path, data_type='csv', subdir=True, pandas_kwargs={'index': True})


    def load_correlations(self):
        if not os.path.exists(self.csv_path):
            self.create_correlations()
        with open(self.csv_path, 'rb') as f:
            df = pd.read_csv(self.csv_path, header=0, index_col=0)
        return df


    def load_centralities(self):
        if not os.path.exists(self.pkl_path):
            self.create_centralities()
        with open(self.pkl_path, 'rb') as f:
            cents = pickle.load(f)
        return cents


    def create_centralities(self):
        deg = {}
        between= {}
        close = {}
        eigen = {}

        for topk in self.te.get_top_combinations():
            info, plttitle = topk
            graph = NXNetwork(self.language, path=info.spmx_path).graph
            
            deg[info.as_string()] = nx.degree_centrality(graph)
            between[info.as_string()] = nx.betweenness_centrality(graph)
            close[info.as_string()] = nx.closeness_centrality(graph)
            eigen[info.as_string()] = None #nx.eigenvector_centrality_numpy(graph, max_iter=1000, tol=1e-04)

        cents = {
            'deg': deg,
            'between': between,
            'close': close,
            # 'eigen': eigen
        }

        for k, v in cents.items():
            df = pd.DataFrame(v)
            cents[k] = df.round(5)

        with open(self.pkl_path, 'wb') as f:
            pickle.dump(cents, f)

        
