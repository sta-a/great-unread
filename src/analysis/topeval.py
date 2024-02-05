import pandas as pd
import os
import sys
sys.path.append("..")
import itertools
from cluster.combinations import InfoHandler
import logging
logging.basicConfig(level=logging.DEBUG)


class TopEval(InfoHandler):
    '''
    Filter evaluation files.
    '''
    def __init__(self, language, cmode, expname, expd):
        super().__init__(language=language, add_color=True, cmode=cmode)
        self.cmode = cmode
        self.expname = expname
        self.expd = expd
        self.ntop = 2
        self.prepare_dfs()


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
        Filter for rows where size of biggest cluster is below threshold.
        This avoids combinations where most data points are put into one cluster.
        '''
        df['biggest_clst'] = df['clst_sizes_ext'].apply(lambda x: x[0])
        df = df[df['biggest_clst'] <= self.expd['maxsize']]
        return df
    
                                
    def make_plttitle(self, df):
        '''
        Combine relevant evaluation measures into a string to display on the plots.
        '''
        if self.cmode == 'mx':
            if self.scale == 'cat':
                cols = ['ARI', 'nmi', 'fmi', 'mean_purity', 'silhouette_score', 'nclust', 'clst_sizes']
            else:
                cols = ['logreg_acc', 'logreg_acc_balanced', 'anova_pval', 'silhouette_score', 'nclust', 'clst_sizes']

        elif self.cmode == 'nk':
            if self.scale == 'cat':
                cols = ['ARI', 'nmi', 'fmi', 'mean_purity', 'modularity', 'nclust', 'clst_sizes']
            else:
                cols = ['logreg_acc', 'logreg_acc_balanced', 'anova_pval', 'modularity', 'nclust', 'clst_sizes']

        # Create the plttitle column
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
        nrows = len(df)
        dfna = df[df.isna().any(axis=1)]
        df = df.dropna()
        assert nrows == len(dfna) + len(df)
        dfna.to_csv(f'{self.cmode}-na-rows.csv', header=True, index=True)
        return df
    

    def filter_top_rows(self, df):
        '''
        Find rows with the best evaluation scores.
        '''
        def find_next_divisible(b, s):
            # Check if b is divisible by s
            if b % s == 0:
                return b
            
            # Find the next bigger number that is divisible by s
            next_divisible = (b // s + 1) * s
            return next_divisible
        

        if len(self.expd['dfs']) == 1:
            nrows = self.ntop
        else:
            # The same clustering has the same internal evaluation value, but multiple rows in the df to to the different attrs
            # Find the next multiple of the nr attrs being considered that is bigger than ntop
            nrows = find_next_divisible(self.ntop, len(self.expd['attr']))

        evalcol = self.expd['evalcol']
        # Filter out rows that contain string values ('invalid')
        mask = pd.to_numeric(df[evalcol], errors='coerce').notnull()
        df = df.loc[mask]

        df[evalcol] = pd.to_numeric(df[evalcol], errors='raise')
        df = df.nlargest(n=nrows, columns=evalcol, keep='first') ##################################  'all'
        print('nlargest', df.shape)
        return df 
        

    def filter_attr(self, df):
        df = df[df['attr'].isin(self.expd['attr'])]
        return df
    
    
    def prepare_dfs(self):
        self.dfs = {}
        for scale in self.expd['dfs']:
            self.scale = scale
            evaldir = os.path.join(self.output_dir, f'{self.cmode}eval')
            df = pd.read_csv(os.path.join(evaldir, f'{self.scale}_results.csv'), header=0, na_values=['NA'])

            df = self.drop_na_rows(df)
            df = self.drop_duplicated_rows(df)
            df = self.extend_sizes_col(df)
            if 'maxsize' in self.expd:
                df = self.filter_clst_sizes(df)
            if 'attr' in self.expd:
                df = self.filter_attr(df)
            df = self.filter_top_rows(df)
            df = self.make_plttitle(df)
            self.dfs[self.scale] = df
    

    def get_topdict(self):
        '''
        Find combinations with the best evaluation scores for both categorical and continuous attributes.
        '''
        topdict = {}

        if len(self.expd['dfs']) == 1:
            # For external evaluation, ntop rows are kept for each cat and cont
            # Different evaluation metrics are used and they are not compareable
            for name, df in self.dfs.items():
                d = dict(zip(df['file_info'], df['plttitle'])) 
                topdict.update(d)
        else:
            dfs = [df[['file_info', 'plttitle', self.expd['evalcol']]] for df in self.dfs.values()]
            df = pd.concat(dfs)
            # For inteval, only ntop rows are kept, because cat and cont can be compared with the same inteval measure
            df = self.filter_top_rows(df)
            self.dfs = {'df': df}
        return topdict
    

    def get_top_combinations(self):
        '''
        Get combinations with the best evaluation scores, load their info from file.
        '''
        topdict = self.get_topdict()

        for tinfo, plttitle in topdict.items():
            comb_info, attr = tinfo.rsplit('_', 1)
            info = self.load_info(comb_info)
            
            info.add('attr', attr)
            metadf = self.merge_dfs(self.metadf, info.clusterdf)
            metadf = self.mh.add_cluster_color_and_shape(metadf)
            info.add('metadf', metadf)
            yield info, plttitle


    def save_dfs(self, path):
        for name, df in self.dfs.items():
            self.save_data(data=df, file_path=os.path.join(path, f'{name}.{self.data_type}'))
