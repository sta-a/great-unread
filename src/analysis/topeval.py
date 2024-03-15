import pandas as pd
import os
import sys
sys.path.append("..")
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from cluster.combinations import InfoHandler
from cluster.evaluate import ExtEval
import time
import regex as re
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)


class TopEval(InfoHandler):
    '''
    Filter evaluation files for different experiments.
    -df: a filtered evaluation file. Must contain column 'file_info'
        Df can be precomputed and passed to TopEval to use methods, or is calculated by TopEval
    '''
    def __init__(self, language, cmode, exp, expdir, df=None, by_author=False):
        super().__init__(language=language, add_color=True, cmode=cmode, by_author=by_author)
        self.cmode = cmode
        self.exp = exp
        self.expdir = expdir
        self.ntop = self.exp['ntop']
        self.df = df
        if self.df is None:
            self.df = self.load_data()


    def extend_sizes_col(self, df):
        '''
        If a size occurs multiple times, it is written as "ntimes x size". If a size occurs only once, it is written as the size only.
        Example: One cluster of size 300 and two clusters of size 100
        'clst_sizes' columns contains string in format: "300, 2x100"
        This function converts it to "300, 100, 100"
        '''
        def compressed_to_list(x):
            sizes = []
            x = ''.join(x.split())
            x = x.split(',')
            for size in x:
                if 'x' in size:
                    n, nr = size.split('x')
                    l = [int(nr)]*int(n)
                    sizes.extend(l)
                else:
                    sizes.append(int(size))
            return sizes
        
        df['clst_sizes_ext'] = df['clst_sizes'].apply(compressed_to_list)
        return df
    

    def filter_clst_sizes(self, df):
        '''
        Find the size of the biggest cluster.
        Ignore clusters of size 1.
        Filter for rows where size of biggest cluster is below threshold.
        '''
        df['niso'] = df['clst_sizes_ext'].apply(lambda x: x.count(1))
        df['nsamples'] = df['clst_sizes_ext'].apply(sum)
        df['nsamples_noniso'] = df['nsamples'] - df['niso']
        df['threshold'] = round(self.exp['maxsize'] * df['nsamples_noniso'])
        df['biggest_clst'] = df['clst_sizes_ext'].apply(lambda x: x[0])
        df = df[df['biggest_clst'] <= df['threshold'] ]

        # niso = df['clst_sizes_ext'].apply(lambda x: x.count(1))
        # nsamples = df['clst_sizes_ext'].apply(sum)
        # nsamples_noniso = nsamples - niso
        # threshold = round(self.exp['maxsize'] * nsamples_noniso)

        # df['biggest_clst'] = df['clst_sizes_ext'].apply(lambda x: x[0])
        # df = df[df['biggest_clst'] <= threshold]
        return df
    

    def add_rank(self, df):
        df['rank'] = df[self.exp['evalcol']].rank(method='dense', ascending=False).astype(int)
        return df
    
                                
    def make_plttitle(self, df):
        '''
        Combine relevant evaluation measures into a string to display on the plots.
        '''
        general = ['nclust', 'clst_sizes']
        
        if 'evalcol' in self.exp:
            df = self.add_rank(df)
            general = ['rank'] + general

        if self.scale == 'cat':
            cols = ['ARI', 'nmi', 'fmi', 'mean_purity', 'silhouette_score']
        else:
            cols = ['logreg_acc', 'logreg_acc_balanced', 'anova_pval', 'kruskal_pval', 'silhouette_score']

        # Internal evaluation metric is different for mx and nk clustering
        if self.cmode == 'nk':
            cols = [s.replace('silhouette_score', 'modularity') for s in cols]

        cols = general + cols

        # String of interesting columns to be displayed on the plot
        df['plttitle'] = df[cols].apply(lambda row: ', '.join(f"{col}: {row[col]}" for col in cols), axis=1)
        return df


    def drop_duplicated_rows(self, df):
        '''
        Identify duplicated rows based on the "file_info" column.
        Duplicated rows can occur when evaluation is cancelled and restarted.
        Combinations that were evaluated but not saved to picked in the first call are reevaluated.
        '''
        duplicated_rows = df[df.duplicated(subset=['file_info'], keep=False)] ###############################
        duplicated_rows.to_csv(f'{self.cmode}-duplicated-rows.csv', header=True, index=True)

        # Keep only the first occurrence of each duplicated content in 'file_info'
        df = df.drop_duplicates(subset=['file_info'], keep='first')
        return df
    

    def drop_na_rows(self, df):
        nrow = len(df)
        dfna = df[df.isna().any(axis=1)]
        df = df.dropna()
        assert nrow == len(dfna) + len(df)
        dfna.to_csv(f'{self.cmode}-na-rows.csv', header=True, index=True)
        return df
    

    def filter_top_rows(self, df, nrow=None):
        '''
        Find rows with the best evaluation scores.
        '''
        if nrow is None:
            nrow = self.ntop

        evalcol = self.exp['evalcol']
        # Filter out rows that contain string values ('invalid')
        mask = pd.to_numeric(df[evalcol], errors='coerce').notnull()
        df = df.loc[mask]

        df[evalcol] = pd.to_numeric(df[evalcol], errors='raise')
        df = df.nlargest(n=nrow, columns=evalcol, keep='all')
        return df 
        

    # def filter_attribute(self, df):
    #     '''
    #     Filter the df and keep only the rows with a certain attribute.
    #     '''
    #     # Create a regex pattern by joining the list elements with '|'
    #     pattern = '|'.join(map(re.escape, self.exp['attr']))

    #     # Apply the filter
    #     filtered_df = df[df['attr'].str.contains(pattern, na=False)]
        
    #     return filtered_df


    def filter_columns_substring(self, df, colname):     
        '''
        Filter the df based on whether the values in column 'colname' contain
        substrings that are in the list.
        '''
        # Create a regex pattern by joining substrings_list with '|'
        pattern = '|'.join(self.exp[colname])
        df = df[df[colname].str.contains(pattern, na=False)]
        return df
    

    def filter_inteval(self, df):
        return df[df[self.exp['intcol']] >= self.exp['intthresh']]
    

    def filter_attrviz(self, df):
        # df = df[['mxname', 'clst_alg_params', 'file_info']]
        subset = ['mxname']
        if 'sparsmode' in df.columns:
            subset.append('sparsmode')
        return df.drop_duplicates(subset=subset)
    
    
    def create_data(self):
        self.dfs = {}
        for scale in self.exp['dfs']:
            self.scale = scale
            evaldir = os.path.join(self.output_dir, f'{self.cmode}eval')
            df = pd.read_csv(os.path.join(evaldir, f'{self.scale}_results.csv'), header=0, na_values=['NA'])

            df = self.drop_na_rows(df)
            df = self.drop_duplicated_rows(df)
            df = self.extend_sizes_col(df)
            if 'maxsize' in self.exp:
                df = self.filter_clst_sizes(df)
            if 'attr' in self.exp:
                # df = self.filter_attribute(df)
                print('ext attr', self.exp['attr'])
                df = df[df['attr'] == self.exp['attr']]
            if 'mxname' in self.exp:
                df = self.filter_columns_substring(df, 'mxname')
            if 'clst_alg_params' in self.exp:
                df = self.filter_columns_substring(df, 'clst_alg_params')
            if 'intthresh' in self.exp:
                df = self.filter_inteval(df)
            if (self.exp['name'] == 'attrviz') or (self.exp['name'] == 'attrviz_int'):
                df = self.filter_attrviz(df)
            self.dfs[self.scale] = df
    

        if len(self.exp['dfs']) == 2:
            df = pd.concat(self.dfs, ignore_index=False)
        else:
            df = self.dfs[self.scale]

        if 'evalcol' in self.exp:
            df = df.sort_values(by=self.exp['evalcol'], ascending=False)

        # if 'evalcol' in self.exp and 'intcol' in self.exp: ###################################
        #     self.plot_cols(df)
        
        df = self.make_plttitle(df)
        self.save_data(data=df)


    def plot_cols(self, df):
        '''
        Plot the values from columns 'intcol' and 'evalcol' with lines connecting the dots.
        '''
        df = self.filter_top_rows(df, nrow=500)
        print(df.shape)
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        x_values = np.arange(len(df))
        plt.plot(x_values, df[self.exp['intcol']], marker='o', label='intcol', color='blue', markersize=3, linewidth=1)
        plt.plot(x_values, df[self.exp['evalcol']], marker='o', label='evalcol', color='red', markersize=3, linewidth=1)
        plt.xlabel('Rows')  # Label for the x-axis
        plt.ylabel('Values')  # Label for the y-axis

        correlation = df[self.exp['intcol']].corr(df[self.exp['evalcol']])
        plt.title(f'Plot of intcol and evalcol (Correlation: {correlation:.2f})')


        plt.legend()
        plt.grid(True)
        plt.show()
    

    def run_logreg(self, info, metadf):
        if ('logreg_acc' in self.exp.values()) or ('logreg_acc_balanced' in self.exp.values()):

            # Filter nan
            df = metadf.dropna(subset=[info.attr])
            X = df[info.attr].values.reshape(-1, 1)
            y_true = df['cluster'].values.ravel()

            ee = ExtEval(self.language, self.cmode, info, inteval=None)
            logreg_acc, logrec_acc_balanced = ee.logreg(X, y_true, draw=True, path=self.expdir)


    def get_top_combinations(self, ncomb=float('inf')):
        '''
        Get combinations with the best evaluation scores, load their info from file.
        - df: Can be passed as a parameter so that method can be used independently.
            Otherwise, the class's df is used.
        '''
        df = self.df
        if 'evalcol' in self.exp: # keep only rows with best evaluation metric
            df = self.filter_top_rows(df)

        topdict = dict(zip(df['file_info'], df['plttitle'])) 

        for i, (tinfo, plttitle) in tqdm(enumerate(topdict.items())):
            if i >= ncomb:
                break
            comb_info, attr = tinfo.rsplit('_', 1)
            info = self.load_info(comb_info)
            
            info.add('attr', attr)
            metadf = self.merge_dfs(self.metadf, info.clusterdf)
            metadf = self.mh.add_cluster_color_and_shape(metadf)
            info.add('metadf', metadf)
            self.run_logreg(info, metadf)
            yield info, plttitle


    def get_file_path(self, file_name):
        # file_name for compatibility
        return os.path.join(self.expdir, f'df.{self.data_type}')
    
