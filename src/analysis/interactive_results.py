
# %%
'''
These classes are for further analyzing and visualizing the interactive selections made using the classes in interactive_viz.py
These manal analyses were not important for the final results in the thesis.
'''


import pandas as pd
from copy import deepcopy
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import DataHandler
from cluster.combinations import InfoHandler
from analysis.nkviz import NkSingleViz


class NkSingleVizHighlight(NkSingleViz):
    def __init__(self, language, output_dir, exp, by_author, metadf, attr):
        self.attr = attr
        super().__init__(language, output_dir, exp, by_author)
        print('nksingleviz', language, output_dir, exp, by_author)
        self.metadf = metadf
        self.key_attrs = [self.attr]

    def add_custom_subdir(self):
        self.add_subdir(f'NKAnalysis_highlight_{self.attr}')

    def get_metadf(self):
        self.df = deepcopy(self.metadf)
    

class NKHighlights(DataHandler):
    # Highlight interesting nodes
    def __init__(self, language, output_dir='analysis_s2v', by_author=False):
        super().__init__(language, output_dir=output_dir, by_author=by_author)
        self.by_author = by_author
        self.ih = InfoHandler(language=self.language, add_color=False, cmode=None, by_author=self.by_author)


    def mds_eng_tips(self):
        # Experiment: eng, MDS, by_author
        # highlight red tips that stand out
        attr = 'redtip'
        maplist = [
            'Dickens_Charles_all_1848',
            'Eliot_George_all_1862',
            'Hardy_Thomas_all_1884',
            'Bronte_Charlotte_all_1850',
            'Edgeworth_Maria_all_1809',
            'MacDonald_George_all_1870']
        metadf = self.map_colors_to_list(attr, maplist)
        viz = NkSingleVizHighlight(language=self.language, output_dir=self.output_dir, exp=None, by_author=self.by_author, metadf=metadf, attr=attr)
        viz.visualize()

    def nk_eng_centralwhite(self):
        # Experiment: eng, network, by_author
        # highlight blue group with central white node
        attr = 'centralwhite' # all women
        maplist = ['Burney_Frances_all_1792', 
            'Smith_Charlotte_all_1791', 
            'Inchbald_Elizabeth_all_1794', 
            'Opie_Amelia_all_1802', 
            'Roche_Regina-Maria_all_1796', 
            'Brunton_Mary_all_1812', 
            'Haywood_Eliza_all_1735', 
            'Fielding_Sarah_all_1752', 
            'Scott_Sarah_all_1762']
        metadf = self.map_colors_to_list(attr, maplist)
        viz = NkSingleVizHighlight(language=self.language, output_dir=self.output_dir, exp=None, by_author=self.by_author, metadf=metadf, attr=attr)
        viz.visualize()


    def map_colors_to_list(self, colorname, maplist):
        df = deepcopy(self.ih.metadf)
        colors = plt.get_cmap('tab10').colors
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(maplist)}
        def get_color(label):
            return color_map.get(label, 'lightgray')  # Return 'lightgray' if label is not found
        df[f'{colorname}_color'] = df.index.to_series().apply(get_color)
        return df
    
high = NKHighlights('eng', by_author=True)
# high.mds_eng_tips()
high.nk_eng_centralwhite()


# %%


class InteractiveResults(DataHandler):
    '''
    Analyze results of selections with interactive plots
    '''
    def __init__(self, language, output_dir='analysis_s2v', subdir=None, by_author=False):
        super().__init__(language, output_dir=output_dir, by_author=by_author)
        self.by_author = by_author
        self.add_subdir(subdir)
        ih = InfoHandler(language=self.language, add_color=False, cmode=None, by_author=self.by_author)
        self.df_nocolor = deepcopy(ih.metadf) # Df with all nodes, self.df has only nodes with positions (no iso nodes)
        self.prepare_headers()

        if self.by_author:
            if self.language == 'eng':
                if 'MxNkAnalysis' in self.subdir:
                    self.nresults = 404 # 58 dists * 7 spars, 2 mxs are too small
                else:
                    self.nresults = 406 # 58 dists * 7 spars
        else:
            self.nresults = 522 # 58 dists * 9 spars
        self.txt_files = [f for f in os.listdir(self.subdir) if f.endswith('.txt')]
        print(len(self.txt_files), self.nresults)
        assert len(self.txt_files) == self.nresults

        self.df = self.load_comments_df()

        # self.idcols = ['mxname', 'curr_attr', 'dim']
        # self.df = self.load_data()


    def load_df_from_file(self, filename):
        path = os.path.join(self.subdir, filename)
        if os.path.getsize(path) == 0: # check that df is not empty
            return None
        else:
            ncol = self.count_ncolumns(path)
            if ncol == 105:
                header = self.header_short
            elif ncol == 106:
                header = self.header_long
            else:
                print(ncol)
            df = pd.read_csv(path, header=None)
            df.columns = header

            if ncol == 105:
                # Insert 'class_counter' column filled with NaN values
                dim_index = df.columns.get_loc('dim')
                df.insert(dim_index + 1, 'class_counter', np.nan)
            return df


    def load_comments_df(self):
        rows_with_comments = []
        for filename in self.txt_files:
                df = self.load_df_from_file(filename=filename)
                if df is not None:
                    non_nan_row = df[df['comment'].notna()]
                    assert len(non_nan_row) == 1, "There should be exactly one non-NaN comment per file."
                    rows_with_comments.append(non_nan_row.iloc[0])

        df = pd.DataFrame(rows_with_comments)
        df = df[['mxname', 'comment']]
        return df
    
    
    def load_selected_dfs(self, filtered_df):
        dfs = []
        filenames = filtered_df['mxname'].tolist()
        for filename in self.txt_files:
            f = filename.split('.')[0] # remove extension and attr
            f = f.split('_')
            f = '_'.join(f[:-2]) 
            if f in filenames:
                df = self.load_df_from_file(filename=filename)
                dfs.append(df)
        result_df = pd.concat(dfs, axis=0)
        return result_df


    def filter_string(self, df, string):
        df_with_string = df[df['comment'].str.contains(string, case=False, na=False)]
        df_without_string = df[~df['comment'].str.contains(string, case=False, na=False)]
        return df_with_string, df_without_string
    

    def filter_df(self):
        df_notint, df = self.filter_string(df=self.df, string='not int')

        # df_very_int, _ = self.filter_string(df=df, string='very int')
        # df_very_int.to_csv('dfveryint')

        if self.language == 'eng':
            if 'MxNkAnalysis' in self.subdir:
                df_tip, _ = self.filter_string(df=df, string='tip')
                full_df_tip = self.load_selected_dfs(df_tip)
                label_counts = full_df_tip['label'].value_counts()
                label_counts = label_counts.reset_index()
                label_counts.columns = ['value', 'count']
                print(label_counts)
                label_counts.to_csv('label_counts_redtip.csv')

            else:
                df_tip, _ = self.filter_string(df=df, string='central white')
                full_df_tip = self.load_selected_dfs(df_tip)
                label_counts = full_df_tip['label'].value_counts()
                label_counts = label_counts.reset_index()
                label_counts.columns = ['value', 'count']
                print(label_counts)
                label_counts.to_csv('label_counts_centralwhite.csv')



        # kw = ['very int']
        # kw_mds = ['tip', 'tail', 'central', 'peripheral', 'most connected', 'least connected']
        # kw_quality = ['int']



    def count_ncolumns(self, path):
        with open(path, 'r') as file:
            first_line = file.readline()
            columns = first_line.strip().split(',')
            return len(columns)


    def prepare_headers(self):
        # The results files that were created do not yet contain the column 'class_counter'
        header_short = f"mxname,comment,curr_attr,dim,label,{','.join(self.df_nocolor.columns)}"
        header_long = f"mxname,comment,curr_attr,dim,class_counter,label,{','.join(self.df_nocolor.columns)}"
        self.header_short = header_short.split(',')
        self.header_long = header_long.split(',')



    def load_data(self):
        df = pd.read_csv(self.results_path, header=0, index_col=None, sep=',')
        print('df original', df.shape)

        # Split the df into a df with the labels and one with the remaining mxnames and comments
        df_comment = df[df['label'].isna()]
        print("DataFrame with NaN in 'label' column:")
        print(df_comment.shape)
        df_comment = df_comment[['mxname', 'comment']] # keep only one column
        df_comment = df_comment.dropna(subset=['comment']) # Drop rows where comment column is empty
        assert not df_comment['mxname'].duplicated().any() # Assert that there is only one comment per mxname

        df = df[~df['label'].isna()]
        print("\nDataFrame without NaN in 'label' column:")
        print(df.shape)
        # Drop duplicated rows, which means that a data point was clicked multiple times
        df = df.drop_duplicates(subset=self.idcols + ['label'])
        assert df['comment'].isna().all()
        df = df.drop(columns=['comment']) # drop empty column

        # Merge comment df and label df so that each row with a label has the comment
        nrows_before_merge = df.shape[0]
        df = df.merge(df_comment, on='mxname', validate='m:1', how='outer')
        assert df.shape[0] == nrows_before_merge

        # df = df.merge(self.metadf, left_on='mxname', right_index=True, validate='1:1')
        # assert df.shape[0] == nrows_before_merge
        return df


    # def get_networks(self):
    #     grouped = self.df.groupby(self.idcols)
    #     for group_key, group_df in grouped:
    #         # Extract the list of values in the 'label' column for the current group
    #         labels_list = group_df['label'].tolist()
    #         parts = group_df.loc[0, 'mxname'].split('_', maxsplit=2)
    #         mxname, spars = parts[:2]
    #         spmx_path = os.path.join(self.output_dir, 'similarity', self.language, 'sparsification', f'sparsmx-{mxname}_{spars}.pkl')
    #         print(mxname, spars, spmx_path)
    #         info = CombinationInfo(mxname=mxname, sparsmode=spars, spmx_path=spmx_path)

##################################################################################################

# import argparse
# import os

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--language', type=str)
#     parser.add_argument('--by_author', action='store_true')  # Boolean argument, if flag is used, by_author is set to True

#     args = parser.parse_args()

#     language = args.language
#     by_author = args.by_author

#     print(f"Selected language: {language}")
#     print(f"Is by_author: {by_author}")


# subdirs = ['MxNkAnalysis', 'NkAnalysis']
# for language in ['eng']:
#     for subdir in ['MxNkAnalysis']:
#         ir = InteractiveResults(language=language, subdir=subdir, by_author=True)
#         ir.filter_df()

