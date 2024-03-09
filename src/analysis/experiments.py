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
from .mxviz import MxViz, MxVizAttr
from .nkviz import NkViz, NkVizAttr
from .analysis_utils import GridImage
from .topeval import TopEval
from cluster.network import NXNetwork
from cluster.combinations import CombinationsBase

# import logging
# logging.basicConfig(level=logging.DEBUG)


class Experiment(DataHandler):
    def __init__(self, language, cmode, by_author=False):
        super().__init__(language, output_dir='analysis')
        self.cmode = cmode
        self.by_author = by_author
        # self.mh = MetadataHandler(self.language)
        # self.metadf = self.mh.get_metadata(add_color=True)


    def get_experiments(self):
        # Default values
        maxsize = 0.9
        embmxs = ['both', 'full']
        cat_evalcol = 'ARI'
        cont_evalcol = 'logreg_acc'
        if self.cmode == 'mx':
            int_evalcol = 'silhouette_score'
        else:
            int_evalcol = 'modularity'


        def add_to_top(d):
            d['maxsize'] = maxsize
            d['special'] = True
            d['intthresh'] = 0.3
            d['intcol'] = int_evalcol
            return d


        # Overall best performance
        topcont = [
            {'name': 'topcont', 'evalcol': cont_evalcol, 'dfs': ['cont']},
            {'name': 'topcont_bal', 'evalcol': 'logreg_acc_balanced', 'dfs': ['cont']},
        ]
        topcont = [add_to_top(d) for d in topcont]
        topcat = [{'name': 'topcat', 'evalcol': cat_evalcol, 'dfs': ['cat']}]
        topcat = [add_to_top(d) for d in topcat]


        attrcont = []
        for attr in ['canon', 'year']:
            dlist = deepcopy(topcont)
            for d in dlist:
                d['name'] = d['name'].replace('cont', attr)
                d['attr'] = [attr]
                d['special'] = False
                attrcont.append(d)

        attrcat = []
        for attr in ['gender', 'author']:
            dlist = deepcopy(topcat)
            for d in dlist:
                d['name'] = d['name'].replace('cat', attr)
                d['attr'] = [attr]
                d['special'] = False
                attrcat.append(d)

        attrcat_nointernal = []
        for cdict in attrcat:
            d = deepcopy(cdict)
            d['name'] = d['name'] + '_nointernal'
            del d['intcol']
            del d['intthresh']
            attrcat_nointernal.append(cdict)


        # Get best performance of embedding distances
        topcont_emb = []
        for cdict in topcont:
            d = deepcopy(cdict)
            d['name'] = d['name'] + '_emb'
            del d['intthresh']
            del d['intcol']
            d['mxname'] = embmxs
            topcont_emb.append(d)


        # Get best performance of embedding distances
        topcat_emb = []
        for cdict in topcat:
            d = deepcopy(cdict)
            d['name'] = d['name'] + '_emb'
            del d['intthresh']
            del d['intcol']
            d['mxname'] = embmxs
            topcat_emb.append(d)


        # Visualize attributes, ignore clustering
        attrviz = [{'name': 'attrviz', 'dfs': ['cat'], 'mxname': ['burrows'] + embmxs}]

        attrviz_int = [{'name': 'attrviz_int', 'dfs': ['cat']}]

        clustconst = [{'name': 'clustconst', 'maxsize': maxsize, 'dfs': ['cat'], 'attr': ['author']}]

        central = deepcopy(clustconst)
        central[0]['name'] = 'central'
        central[0]['mxname'] = ['burrows'] + embmxs
        central[0]['special'] = True

        exps = topcont + topcat + attrcont + attrcat + attrcat_nointernal + topcont_emb + topcat_emb + attrviz_int + clustconst + attrviz
        exps = attrviz
        return exps


    def run_experiments(self, ntop=10):
        exps = self.get_experiments()
        for exp in exps:
            expname = exp['name']
            print(f'------------------{expname}-------------------\n')
            if 'ntop' not in exp:
                exp['ntop'] = ntop

            self.add_subdir(f'{self.cmode}_{expname}')
            te = TopEval(self.language, self.cmode, exp, expdir=self.subdir, by_author=self.by_author)

            if expname == 'clustconst':
                self.run_clustconst(exp, te)
            elif expname == 'central':
                if self.cmode == 'nk':
                    self.run_central(exp, te)
            else:
                self.visualize(exp, te)


    def run_central(self, exp, te):
        df = Central(self.language, self.cmode, exp, te).run()
        centralities = df.columns
        for centrality in centralities:
            exp['evalcol'] = centrality
            te = TopEval(self.language, self.cmode, exp, expdir=self.subdir, df=df, by_author=self.by_author)
            self.visualize_nk(exp, te, vizname=centrality)


    def run_clustconst(self, exp, te):
        ClusterComparison(self.language, self.cmode, exp, te).run()


    def visualize(self, exp, te):
        if self.cmode == 'mx':
            self.visualize_mx(exp, te)
        else:
            self.visualize_nk(exp, te)


    def visualize_mx(self, exp, te, vizname='viz'):
        expname = exp['name']

        for topk in te.get_top_combinations():
            info, plttitle = topk
            print(info.as_string())
            if 'special' in exp and exp['special']:
                info.add('special', 'canon')

            # Get matrix
            cb = CombinationsBase(self.language, add_color=False, cmode='mx')
            mx = [mx for mx in cb.mxs if mx.name == info.mxname]
            assert len(mx) == 1
            mx = mx[0]

            info.add('order', 'olo')
            if expname == 'attrviz' or expname == 'attrviz_int':
                viz = MxVizAttr(self.language, mx, info, plttitle=plttitle, expname=expname)
            else:
                viz = MxViz(self.language, mx, info, plttitle=plttitle, expname=expname)
            viz.visualize(vizname)


    def visualize_nk(self, exp, te, vizname='viz'):
        expname = exp['name']
        print(expname)

        for topk in te.get_top_combinations():
            info, plttitle = topk
            print(info.as_string())
            if 'special' in exp and exp['special']:
                info.add('special', 'canon')
            network = NXNetwork(self.language, path=info.spmx_path)
            if exp['name'] == 'attrviz' or expname == 'attrviz_int':
                viz = NkVizAttr(self.language, network, info, plttitle=plttitle, expname=expname)
                viz.visualize(vizname)
                # gi = GridImage(self.language, self.cmode, exp)
                # gi.run()      
            else:
                viz = NkViz(self.language, network, info, plttitle=plttitle, expname=expname) 
                viz.visualize(vizname)




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

        
