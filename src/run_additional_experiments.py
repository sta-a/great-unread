# %%

import sys
sys.path.append("..")
from utils import TextsByAuthor, DataHandler
import pandas as pd
import os
import numpy as np
import pickle
from analysis.experiments import Experiment
from analysis.topeval import TopEval
from cluster.create import D2vDist, Delta
from cluster.cluster_utils import MetadataHandler


# Some additional experiments, many of which are mentioned in the text
# Helps evaluate the clustering results
# Check combinations with the highest nr of clusters for recovering author
# Analysis, mx

def extend_sizes_col(df):
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


def filter_clst_sizes(df):
    '''
    Find the size of the biggest cluster.
    Ignore clusters of size 1.
    Filter for rows where size of biggest cluster is below threshold.
    '''
    df['niso'] = df['clst_sizes_ext'].apply(lambda x: x.count(1))
    df['nsamples'] = df['clst_sizes_ext'].apply(sum)
    df['nsamples_noniso'] = df['nsamples'] - df['niso']
    df['threshold'] = round(0.9 * df['nsamples_noniso'])
    df['biggest_clst'] = df['clst_sizes_ext'].apply(lambda x: x[0])
    df = df[df['biggest_clst'] <= df['threshold'] ]
    return df
    

class AnalysisMxAuthor(DataHandler):
    def __init__(self, language=None, output_dir='extraexp', by_author=False):
        super().__init__(language=language, output_dir=output_dir, data_type='csv', by_author=by_author)

        self.add_subdir(f'mx_{self.__class__.__name__}')
        self.path = f'/media/annina/elements/back-to-computer-240615/data/similarity/{self.language}/mxeval/cat_results.csv'
        print(f'Path: {self.path}')
        self.df = pd.read_csv(self.path)

    def find_max_nr_clusts_dbscan(self):
        # Find the max number of clusters that were produced using DBSCAN
        # Filter rows where 'clst_alg_params' contains 'dbscan'
        dbscan_df = self.df[self.df['clst_alg_params'].str.contains('dbscan', case=False, na=False)]

        # Find the row with the maximum value in 'nclust'
        max_row = dbscan_df.loc[dbscan_df['nclust'].idxmax()]

        print(max_row)

    def analyze_clusters_with_lower_silhouette(self):
        # Filter rows where 'nclust' is 50 or bigger
        df = self.df[self.df['nclust'] >= 50]
        df = df[df['attr'] == 'author']
        df = df[df['silhouette_score'] >=0.2] # lower silhouette score finds good results for English
        df = df[df['ARI'] > 0]

        # df = df.sort_values(by='silhouette_score', ascending=False)
        df = df.sort_values(by='vmeasure', ascending=False)
        print(df)
        self.save_data(data=df, file_name=os.path.basename(self.path), subdir=True)

        df['plttitle'] = 'empty-plttitle'
        # exp = {'name': self.__class__.__name__, 'ntop': 30, 'maxsize': 90, 'intthresh': 0.2, 'viztype': 'keyattr'} # visualize all attrs
        exp = {'name': self.__class__.__name__, 'ntop': 30, 'maxsize': 90, 'intthresh': 0.2, 'viztype': 'singleimage_extra_experimtents'} # make single image with author or clusters highlighted
        te = TopEval(language=self.language, output_dir='similarity', cmode='mx', exp=exp, expdir=None, df=df, by_author=self.by_author)
        experiment = Experiment(language=self.language, cmode='mx', by_author=self.by_author, output_dir=self.output_dir)
        experiment.visualize_mx(exp=exp, te=te)

    def analyze_clusters_with_vmeasure(self):
        # Filter rows where 'nclust' is 50 or bigger
        df = self.df[self.df['attr'] == 'author']
        df = df[df['silhouette_score'] >=0.2] # lower silhouette score finds good results for English

        # df = df.sort_values(by='silhouette_score', ascending=False)
        df = df.sort_values(by='vmeasure', ascending=False)
        df = extend_sizes_col(df)
        df = filter_clst_sizes(df)
        print(df)
        # print(df)
        # self.save_data(data=df, file_name=os.path.basename(self.path), subdir=True)

        # df['plttitle'] = 'empty-plttitle'
        # # exp = {'name': self.__class__.__name__, 'ntop': 30, 'maxsize': 90, 'intthresh': 0.2, 'viztype': 'keyattr'} # visualize all attrs
        # exp = {'name': self.__class__.__name__, 'ntop': 30, 'maxsize': 90, 'intthresh': 0.2, 'viztype': 'singleimage_extra_experimtents'} # make single image with author or clusters highlighted
        # te = TopEval(language=self.language, output_dir='similarity', cmode='mx', exp=exp, expdir=None, df=df, by_author=self.by_author)
        # experiment = Experiment(language=self.language, cmode='mx', by_author=self.by_author, output_dir=self.output_dir)
        # experiment.visualize_mx(exp=exp, te=te)




class AnalysisMxGender(DataHandler):
    def __init__(self, language=None, output_dir='extraexp', by_author=False):
        super().__init__(language=language, output_dir=output_dir, data_type='csv', by_author=by_author)

        self.add_subdir(f'mx_{self.__class__.__name__}')
        self.path = f'/media/annina/elements/back-to-computer-240615/data/similarity/{self.language}/mxeval/cat_results.csv'
        print(f'Path: {self.path}')
        self.df = pd.read_csv(self.path)


    def analyze_clusters(self):
        # Filter rows where 'nclust' is 50 or bigger
        df = self.df[self.df['attr'] == 'gender']
        # df = df[df['silhouette_score'] >=0.3] # lower silhouette score finds good results for English
        # df = df[df['nclust'] >= 3]

        # # df = df.sort_values(by='silhouette_score', ascending=False)
        # df = df.sort_values(by='weighted_avg_variance', ascending=True)

        df = df[df['silhouette_score'] >=0.2] # lower silhouette score finds good results for English
        print(df)
        df = df.sort_values(by='clst_sizes', ascending=False)
        # df = df.sort_values(by='nclust', ascending=False) # both_kmedoids-nclust-50_year

        # print(df)
        self.save_data(data=df, file_name=os.path.basename(self.path), subdir=True)
        df['plttitle'] = 'empty-plttitle'
        exp = {'name': self.__class__.__name__, 'ntop': 1, 'maxsize': 90, 'intthresh': 0.2, 'viztype': 'singleimage_extra_experimtents'}
        te = TopEval(language=self.language, output_dir='similarity', cmode='mx', exp=exp, expdir=None, df=df, by_author=self.by_author)
        experiment = Experiment(language=self.language, cmode='mx', by_author=self.by_author, output_dir=self.output_dir)
        experiment.visualize_mx(exp=exp, te=te)



class AnalysisNkGender(DataHandler):
    def __init__(self, language=None, output_dir='extraexp', by_author=False):
        super().__init__(language=language, output_dir=output_dir, data_type='csv', by_author=by_author)

        self.add_subdir(f'nk_{self.__class__.__name__}')
        self.path = f'/media/annina/elements/back-to-computer-240615/data/similarity/{self.language}/nkeval/cat_results.csv'
        print(f'Path: {self.path}')
        self.df = pd.read_csv(self.path)


    def analyze_homogeneity(self):
        # homogeneity_path = '/media/annina/elements/back-to-computer-240615/data/analysis/eng/nk_topgender_homogeneity/df.csv' # take all valid combinations
        # self.df = pd.read_csv(homogeneity_path)
        # df = self.df.sort_values(by='nclust', ascending=True)
        # self.save_data(data=df, file_name=os.path.basename(self.path), subdir=True)
        # df['plttitle'] = 'empty-plttitle'
        exp = {'name': self.__class__.__name__, 'ntop': 1, 'nclust_max': 50, 'nclust_min': 50,  'maxsize': 90, 'intthresh': 0.3, 'intcol': 'modularity', 'viztype': 'keyattr', 'attr': 'gender', 'evalcol': 'homogeneity', 'dfs': ['cat']}
        te = TopEval(language=self.language, output_dir='similarity', cmode='nk', exp=exp, expdir=self.subdir, by_author=self.by_author)
        df = te.create_data()
        print(df)
        experiment = Experiment(language=self.language, cmode='mx', by_author=self.by_author, output_dir=self.output_dir)
        experiment.visualize_nk(exp=exp, te=te, subdir=self.subdir)



class AnalysisMxYear(DataHandler):
    def __init__(self, language=None, output_dir='extraexp', by_author=False):
        super().__init__(language=language, output_dir=output_dir, data_type='csv', by_author=by_author)

        self.add_subdir(f'mx_{self.__class__.__name__}')
        self.path = f'/media/annina/elements/back-to-computer-240615/data/similarity/{self.language}/mxeval/cont_results.csv'
        print(f'Path: {self.path}')
        self.df = pd.read_csv(self.path)


    def analyze_clusters(self):
        # Filter rows where 'nclust' is 50 or bigger
        df = self.df[self.df['attr'] == 'year']
        # df = df[df['silhouette_score'] >=0.3] # lower silhouette score finds good results for English
        # df = df[df['nclust'] >= 3]

        # # df = df.sort_values(by='silhouette_score', ascending=False)
        # df = df.sort_values(by='weighted_avg_variance', ascending=True)

        df = df[df['silhouette_score'] >=0.2] # lower silhouette score finds good results for English
        # df = df.sort_values(by='silhouette_score', ascending=False)
        df = df.sort_values(by='nclust', ascending=False) # both_kmedoids-nclust-50_year

        print(df)
        self.save_data(data=df, file_name=os.path.basename(self.path), subdir=True)
        df['plttitle'] = 'empty-plttitle'
        exp = {'name': self.__class__.__name__, 'ntop': 30, 'maxsize': 90, 'intthresh': 0.2, 'viztype': 'singleimage_extra_experimtents'}
        te = TopEval(language=self.language, output_dir='similarity', cmode='mx', exp=exp, expdir=None, df=df, by_author=self.by_author)
        experiment = Experiment(language=self.language, cmode='mx', by_author=self.by_author, output_dir=self.output_dir)
        experiment.visualize_mx(exp=exp, te=te)


class AnalysisNkCanonByAuthor(DataHandler):
    def __init__(self, language=None, output_dir='extraexp', by_author=True):
        super().__init__(language=language, output_dir=output_dir, data_type='csv', by_author=by_author)
        self.cmode = 'nk'
        self.add_subdir(f'{self.cmode}_{self.__class__.__name__}')

    def analyze_clusters(self):
        exp = {'dfs': ['cont'], 'name': f'{self.cmode}_{self.__class__.__name__}', 'maxsize': 0.9, 'attr': 'canon', 'intthresh': 0.3, 'intcol': 'modularity', 'ntop': 30, 'viztype': 'keyattr'} 
        te = TopEval(language=self.language, output_dir='similarity', cmode='nk', exp=exp, expdir=self.subdir, df=None, by_author=True)
        df = te.create_data()
        print(df)



# for language in ['eng', 'ger']:
for language in ['ger']:
    ama = AnalysisMxAuthor(language=language)
    # ama.analyze_clusters_with_vmeasure()
    ama.analyze_clusters_with_lower_silhouette()

    # ama = AnalysisNkGender(language=language)
    # ama.analyze_homogeneity()

    # ama = AnalysisMxYear(language=language)
    # ama.analyze_clusters()

    # ama = AnalysisNkCanonByAuthor(language=language, by_author=True)
    # ama.analyze_clusters()



# %%
import os

# Get the current directory
current_dir = '/media/annina/elements/back-to-computer-240615/data/analysis/eng' # mx_topauthor_ARI_nclust-50-50
# List to store directories and their file counts
dir_file_counts = []

# Iterate through each item in the current directory
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    
    # Check if the item is a directory
    if os.path.isdir(item_path) and 'nk' in item_path and 'gender' in item_path and 'purity' in item_path:
        # Count the number of files in the directory
        num_files = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
        
        # Add the directory and file count to the list
        dir_file_counts.append((item, num_files))

# Sort the list by the number of files (second element of the tuple)
dir_file_counts.sort(key=lambda x: x[1], reverse=True)

for idx, (dir_name, file_count) in enumerate(dir_file_counts, start=1):
    print(f"{idx}. Directory: {dir_name}, Number of files: {file_count}", os.path.join(current_dir, dir_name))




# year: /media/annina/elements/back-to-computer-240615/data/analysis/eng/mx_topyear_ext_calinski_harabasz interesting





# %%

import pandas as pd
import os
# Display the table in a format that can be copied into latex

def make_latex_table(df):
    print('\n\n\\begin{table}')
    print('\\centering')
    print('\\caption{}')
    print(df.to_latex(index=False))
    print('\\label{}')
    print('\\end{table}')


def extract_distance_and_sparsification(mxname):
    parts = mxname.split('_')
    # Extract the first part for Distance and the second part for Sparsification
    return parts[0], parts[1]
    return None, None




def prepare_df(path, eval_metric, int_eval_metric, topn=None):
    if not 'csv' in path:
        path = os.path.join(path, 'df.csv')
    df = pd.read_csv(path)
    # Filter out invalid linkage functions
    mask = ~df['clst_alg_params'].str.contains('centroid|median|ward', case=False, na=False)
    df = df[mask]
    if 'sparsmode' in df.columns:
        df = df[['mxname', 'sparsmode', 'clst_alg_params', eval_metric, int_eval_metric]]
    else:
        df = df[['mxname', 'clst_alg_params', eval_metric, int_eval_metric]]

    if df['mxname'].str.contains('dimensions').any():
        df[['Distance', 'sparsmode']] = df['mxname'].apply(lambda x: pd.Series(extract_distance_and_sparsification(x)))
        df = df[['Distance', 'sparsmode', 'clst_alg_params', eval_metric, int_eval_metric]]

    df = df.sort_values(by=eval_metric, ascending=False)
    if topn is None:
        topn = len(df)
    df = df.head(topn)
    df['clst_alg_params'] = df['clst_alg_params'].str.replace('%', '.')


    df = df.rename({
        'mxname': 'Distance', 
        'sparsmode': 'Sparsification', 
        'clst_alg_params': 'Clst. Alg. + Params',
        'modularity': 'Modularity',
        'silhouette_score': 'Silhouette Score'}, inplace=False, axis='columns')
    return df







# dirpath = '/media/annina/elements/back-to-computer-240615/data/analysis/eng/nk_topauthor_ARI_nclust-50-100' 
# dirpath = '/media/annina/elements/back-to-computer-240615/data/analysis/ger/nk_topauthor_ARI_nclust-50-100' 
# topn = 20
# mode = 'nk'


# dirpath = '/media/annina/elements/back-to-computer-240615/data/analysis_s2v/ger/mx_topauthor_ARI_nclust-50-50' # check 50-100  - the same?
# topn = 3
# mode = 'mx

dirpath = '/media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/ger/mx_topgender_ARI_nclust-2-2'
dirpath = '/media/annina/elements/back-to-computer-240615/data_author/analysis/eng/mx_topgender_ARI_nclust-2-2'
topn = 3
mode = 'mx'

dirpath = '/media/annina/elements/back-to-computer-240615/data/extraexp/eng/mx_AnalysisMxAuthor/cat_results.csv'
mode = 'mx'
topn = None

eval_metric = 'ARI'
if mode == 'nk':
    int_eval_metric = 'modularity'
else:
    int_eval_metric = 'silhouette_score'



df = prepare_df(dirpath, eval_metric, int_eval_metric, topn)
make_latex_table(df)





# %%
# Create copying commands
from copy import deepcopy
import shutil

copy = True
source = '/media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/ger/nk_singleimage_s2v/full_simmel-5-10_dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True_dbscan-eps-0%3-minsamples-30_cluster.png'
source = '/media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/ger/nk_singleimage_s2v/full_simmel-5-10_dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True_canon.png'
source = '/media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/ger/mx_singleimage_s2v/s2v-full_simmel-5-10_dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True_canon.png'

target = deepcopy(source)

target = target.replace('/media/annina/elements/back-to-computer-240615', '/home/annina/Documents/thesis')
if '/data/' in target:
    target = target.replace('/data/', '/data_latex/')
elif '/data_author/' in target:
    target = target.replace('/data_author/', '/data_author_latex/')



print(target)

if copy:
    target_dir = os.path.dirname(target)
    
    # Ensure the parent directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f'Created directories for: {target}')
    else:
        print(f'Directories already exist for: {target}')
    shutil.copy(source, target)
    print(f'Copied file from {source} to {target}')

    if '%' in target:
        # Replace '%' with '.'
        new_target_file = target.replace('%', '.')
        shutil.copy(source, new_target_file)

    with open('copy-files-for-thesis.sh', 'a') as f:
        f.write(f'{source} -> {target}\n')

# %%
