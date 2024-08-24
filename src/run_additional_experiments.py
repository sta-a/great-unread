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



# %%
# Create a color bar for cmap 'seismic'
import matplotlib.pyplot as plt
import matplotlib as mpl
# Create a figure and axis with a slim vertical size
fig, ax = plt.subplots(figsize=(0.3, 7))
# Remove the axis frame
ax.set_frame_on(False)
# Create a colorbar with the 'seismic' colormap and set it to vertical
cbar = mpl.colorbar.ColorbarBase(ax, cmap='seismic', orientation='vertical')
# Increase the size of the tick labels
cbar.ax.tick_params(labelsize=20)  # Set font size for tick labels
imgpath = '/home/annina/Documents/thesis/data_latex/extraexp/colorbar-seismic.png'
plt.savefig(imgpath, bbox_inches='tight', pad_inches=0.1)
plt.close()


# %%



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
    if os.path.isdir(item_path) and 'nk' in item_path and 'gender' in item_path and 'vmeasure' in item_path:
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
# Create copying commands
from copy import deepcopy
import shutil

copy = True
source = '/media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/ger/nk_singleimage_s2v/full_simmel-5-10_dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True_dbscan-eps-0%3-minsamples-30_cluster.png'
source = '/media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/ger/nk_singleimage_s2v/full_simmel-5-10_dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True_canon.png'
source = '/media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/ger/mx_singleimage_s2v/s2v-full_simmel-5-10_dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True_canon.png'
source = '/media/annina/elements/back-to-computer-240615/data/extraexp/eng/MxSingleVizCluster/both_hierarchical-nclust-50-method-complete_author.png'
source = '/media/annina/elements/back-to-computer-240615/data/extraexp/eng/MxSingleVizCluster/both_hierarchical-nclust-50-method-complete_cluster.png'
source = '/media/annina/elements/back-to-computer-240615/data_author/analysis/eng/nk_singleimage/braycurtis-2000_simmel-3-10_year.png'
source = '/media/annina/elements/back-to-computer-240615/data_author/analysis/ger/nk_singleimage/manhattan-1000_simmel-3-10_year.png'




def copy_imgs_from_harddrive(source, copy=True):
    target = deepcopy(source)

    target = target.replace('/media/annina/elements/back-to-computer-240615', '/home/annina/Documents/thesis')
    if '/data/' in target:
        target = target.replace('/data/', '/data_latex/')
    elif '/data_author/' in target:
        target = target.replace('/data_author/', '/data_author_latex/')

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

        new_target_file = None
        if '%' in target:
            # Replace '%' with '.'
            new_target_file = target.replace('%', '.')
            shutil.copy(source, new_target_file)

        with open('copy-files-for-thesis.sh', 'a') as copy_paths_file:
            copy_paths_file.write(f'{source} -> {target}\n')

        if new_target_file is not None:
            target = new_target_file
        return target
    
# copy_imgs_from_harddrive(source)




import pandas as pd
import os
import re

import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", category=FutureWarning, message="In future versions `DataFrame.to_latex`")
# Display the table in a format that can be copied into latex

def make_latex_table(df, caption, label, outf):
    if 'Spars' in df.columns:
        column_format = 'lllccccc'
    else:
        column_format = 'llccccc'

    print('\n\n\\begin{table}')
    print('\\centering')
    print('\\scriptsize')
    print(f'\\caption{{{caption}}}')
    print(df.to_latex(index=False, column_format=column_format))
    print(f'\\label{{{label}}}')
    print('\\end{table}')

    outf.write('\\begin{table}\n')
    outf.write('\\centering\n')
    outf.write('\\scriptsize\n')
    outf.write(f'\\caption{{{caption}}}\n')
    outf.write(f'\\label{{{label}}}\n')
    outf.write(df.to_latex(index=False, column_format=column_format))
    outf.write('\\end{table}\n')


def extract_distance_and_sparsification(mxname):
    parts = mxname.split('_')
    # Extract the first part for Distance and the second part for Sparsification
    return parts[0], parts[1]
    return None, None

# Function to update the values
def update_clst_alg_param(value):
    value = re.sub(r'-nclust-\d+', '', value)
    parts = value.split('-')  # Split the string by '-'
    if 'hierarchical' in value:
        value = f'{parts[0]}, {parts[-1]}'  # Join the first and last parts
    elif 'louvain' in value:
        value = f'{parts[0]}, res {parts[-1]}'
    elif 'alpa' in value:
        value = 'lp'
    elif 'dbscan' in value:
        value = f'{parts[0]}, {parts[1]} {parts[2]}, {parts[3]} {parts[4]}'
    return value  # If not in the right format, return the value unchanged

# Apply the function to the column



def prepare_df(path, eval_metric, int_eval_metric, topn=None):
    if not 'csv' in path:
        path = os.path.join(path, 'df.csv')
    df = pd.read_csv(path)
    # Filter out invalid linkage functions
    mask = ~df['clst_alg_params'].str.contains('centroid|median|ward', case=False, na=False)
    df = df[mask]

    if eval_metric == 'vmeasure':
        collist = ['mxname', 'clst_alg_params', 'nclust', 'homogeneity', 'completeness', eval_metric, int_eval_metric]
    else:
        collist = ['mxname', 'clst_alg_params', 'nclust',  eval_metric, int_eval_metric]
    if 'sparsmode' in df.columns:
        collist.insert(1, 'sparsmode')
    
    df = df[collist]

    if df['mxname'].str.contains('dimensions').any():
        df[['Distance', 'sparsmode']] = df['mxname'].apply(lambda x: pd.Series(extract_distance_and_sparsification(x)))
        df = df[['Distance', 'sparsmode', 'clst_alg_params', eval_metric, int_eval_metric]]

    df = df.sort_values(by=eval_metric, ascending=False)
    if topn is None:
        topn = len(df)
    df = df.head(topn)
    df['clst_alg_params'] = df['clst_alg_params'].str.replace('%', '.')
    df['clst_alg_params'] = df['clst_alg_params'].apply(update_clst_alg_param)


    df = df.rename({
        'mxname': 'Dist', 
        'sparsmode': 'Spars', 
        'clst_alg_params': 'Alg + Params',
        'modularity': 'Modularity',
        'nclust': 'NrClst',
        'vmeasure': 'V-measure',
        'homogeneity': 'Homogeneity',
        'completeness': 'Completeness',
        'silhouette_score': 'Silhouette'}, inplace=False, axis='columns')
    return df


def build_caption(attr, level, datadir, language, eval_metric, highest_eval_value):
    if level == 'mx':
        mx_or_nk = 'unsparsified matrices'
    else:
        mx_or_nk = 'networks'
    if datadir == 'data':
        by_author = 'text-based'
        is_by_author = False
    else:
        by_author = 'author-based'
        is_by_author = True 
    if attr == 'eng':
        lang_string = 'English'
    else:
        lang_string = 'German'
    tab_caption = f'Top combinations for \\inquotes{{{attr}}} on {mx_or_nk}, {by_author}, {lang_string}'
    tab_label = f'tab:{level}-{attr}-{eval_metric}-{language}-isbyauthor-{is_by_author}'
    return tab_caption, tab_label


def write_latex_figure(outf, clst_path_eng, attr_path_eng, clst_path_ger, attr_path_ger, caption_eng, caption_ger, caption_fullfig, label, attr):
    
    outf.write(f"""\n\n\\begin{{figure}}[h!]
    \\centering
    \\captionsetup[subfigure]{{labelformat=empty}} % Suppress automatic subfigure labeling for this figure only
    % Top row
    \\begin{{subfigure}}[t]{{\\textwidth}}
        \\centering
        \\begin{{subfigure}}[t]{{0.45\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{clst_path_eng}}}
            \\caption{{Clusters}}
        \\end{{subfigure}}
        \\hfill
        \\begin{{subfigure}}[t]{{0.45\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{attr_path_eng}}}
            \\caption{{{attr.capitalize()}}}
        \\end{{subfigure}}
        \\caption{{{caption_eng}}}
    \\end{{subfigure}}
    
    \\vspace{{1em}} % Adjust the space between top and bottom rows
    
    % Bottom row
    \\begin{{subfigure}}[t]{{\\textwidth}}
        \\centering
        \\begin{{subfigure}}[t]{{0.45\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{clst_path_ger}}}
            \\caption{{Clusters}}
        \\end{{subfigure}}
        \\hfill
        \\begin{{subfigure}}[t]{{0.45\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{attr_path_ger}}}
            \\caption{{{attr.capitalize()}}}
        \\end{{subfigure}}
        \\caption{{{caption_ger}}}
    \\end{{subfigure}}
    
    % Overall caption for the figure
    \\caption{{{caption_fullfig}}}
    \\label{{{label}}}
\\end{{figure}}
""")



def prepare_latex_figure(outf, attr, datadir, eval_metric, first_rows, first_rows_latex, level): 
    if datadir == 'data':
        by_author = 'text-based'
        is_by_author = False
    else:
        by_author = 'author-based'
        is_by_author = True 

    paths = {}
    paths['clst_eng'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis/eng/nk_singleimage/{first_rows['eng']['mxname']}_{first_rows['eng']['sparsmode']}_{first_rows['eng']['clst_alg_params']}_cluster.png"
    paths['attr_eng'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis/eng/nk_singleimage/{first_rows['eng']['mxname']}_{first_rows['eng']['sparsmode']}_{attr}.png"
    paths['clst_ger'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis/ger/nk_singleimage/{first_rows['ger']['mxname']}_{first_rows['ger']['sparsmode']}_{first_rows['ger']['clst_alg_params']}_cluster.png"
    paths['attr_ger'] = f"/media/annina/elements/back-to-computer-240615/{datadir}/analysis/ger/nk_singleimage/{first_rows['ger']['mxname']}_{first_rows['ger']['sparsmode']}_{attr}.png"
    caption_eng = f"Distance \\dist{{{first_rows['eng']['mxname']}}}, sparsified with \\sparsname{{{first_rows['eng']['sparsmode']}}}, {first_rows_latex['eng']['Alg + Params']}, {eval_metric} = {first_rows['eng'][eval_metric]}, English"
    caption_ger = f"Distance \\dist{{{first_rows['ger']['mxname']}}}, sparsified with \\sparsname{{{first_rows['ger']['sparsmode']}}}, {first_rows_latex['ger']['Alg + Params']}, {eval_metric} = {first_rows['ger'][eval_metric]}, German"
    caption_fullfig = f'Best combination for \\inquotes{{{attr}}}  on networks. Groups of size 1 are colored gray.'
    fig_label = f'fig:{level}-{attr}-{eval_metric}-isbyauthor-{is_by_author}'

    for pathname, source in paths.items():
        target = copy_imgs_from_harddrive(source, copy=True)
        paths[pathname] = target


    if 'louvain' in caption_eng:
        caption_eng = caption_eng.replace('louvain, res', 'Louvain with resolution = ')
    if 'louvain' in caption_ger:
        caption_ger = caption_ger.replace('louvain, res', 'Louvain with resolution = ')


    write_latex_figure(outf, paths['clst_eng'], paths['attr_eng'], paths['clst_ger'], paths['attr_ger'], caption_eng, caption_ger, caption_fullfig, fig_label, attr)
    return fig_label



import os
import pandas as pd
with open('latex/tables_for_latex.txt', 'w') as outf:
    # for attr in ['author', 'gender']:
    for attr in ['gender']:
        # for level in ['mx', 'nk']:
        for level in ['nk']:
            for datadir in ['data', 'data_author']:
                first_rows = {}
                first_rows_latex = {}
                tab_labels = {}
                outf.write('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
                outf.write('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
                for language in ['eng', 'ger']:
                    make_latex_figure = False
                    if attr == 'author' and datadir == 'data_author':
                        continue
                    elif attr == 'gender' and datadir == 'data':
                        continue
                    else:
                        make_latex_figure = True


                    path = f'/media/annina/elements/back-to-computer-240615/{datadir}/analysis/{language}/{level}_top{attr}_vmeasure_nclust-2-10'
                    df = pd.read_csv(os.path.join(path, 'df.csv'))
                    if level == 'mx':
                        int_eval_metric = 'silhouette_score'
                    else:
                        int_eval_metric = 'modularity'
                    eval_metric = 'vmeasure'
                    highest_eval_value = df.loc[0, eval_metric]
                    if highest_eval_value >= 0.3:
                        topn = 20
                    else:
                        topn = 3

                    print('------------------------------------------------------------------------')
                    print(attr, level, datadir, language, eval_metric)
                    print(path)
                    print('------------------------------------------------------------------------')

                    outf.write('%------------------------------------------------------------------------\n%')
                    outf.write(f'{attr} {level} {datadir} {language} {eval_metric}\n%')
                    outf.write(f'{path}\n%')
                    outf.write('------------------------------------------------------------------------\n')

                    first_rows[language] = df.iloc[0].to_dict()

                    tab_caption, tab_label = build_caption(attr, level, datadir, language, eval_metric, highest_eval_value)
                    tab_labels[language] = tab_label
                    df = prepare_df(path=path, eval_metric=eval_metric, int_eval_metric=int_eval_metric, topn=topn)
                    first_rows_latex[language] = df.iloc[0].to_dict()
                    outf.write('%------------------------------------------------------------------------\n')

                    make_latex_table(df,tab_caption, tab_label, outf)
                if level == 'nk' and make_latex_figure:
                    fig_label = prepare_latex_figure(outf, attr, datadir, eval_metric, first_rows, first_rows_latex, level)
                    outf.write(f"\n\n The top combinations are shown in Tables \\ref{{{tab_labels['eng']}}} and \\ref{{{tab_labels['ger']}}}, and Figure \\ref{{{fig_label}}} shows the single best combination for each language.\n\n")
# %%

# for language in ['eng', 'ger']:
#     with open(f'latex/nk_topauthor_ARI_nclust-50-100_{language}.txt', 'w') as outf:
#         dirpath = f'/media/annina/elements/back-to-computer-240615/data/analysis/{language}/nk_topauthor_ARI_nclust-50-100' 
#         topn = 20
#         mode = 'nk'


        # dirpath = '/media/annina/elements/back-to-computer-240615/data/analysis_s2v/ger/mx_topauthor_ARI_nclust-50-50' # check 50-100  - the same?
        # topn = 3
        # mode = 'mx

        # dirpath = '/media/annina/elements/back-to-computer-240615/data_author/analysis_s2v/ger/mx_topgender_ARI_nclust-2-2'
        # dirpath = '/media/annina/elements/back-to-computer-240615/data_author/analysis/eng/mx_topgender_ARI_nclust-2-2'
        # topn = 3
        # mode = 'mx'

        # dirpath = '/media/annina/elements/back-to-computer-240615/data/extraexp/ger/mx_AnalysisMxAuthor/cat_results.csv'
        # mode = 'mx'
        # topn = None

        # eval_metric = 'ARI'
        # if mode == 'nk':
        #     int_eval_metric = 'modularity'
        # else:
        #     int_eval_metric = 'silhouette_score'

        # df = prepare_df(dirpath, eval_metric, int_eval_metric, topn)
        # make_latex_table(df, 'caption', 'label', outf)


# %%
# Make network with labels
import pickle
import networkx as nx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

path = '/home/annina/scripts/great_unread_nlp/data_author/similarity/eng/sparsification/sparsmx-braycurtis-2000_simmel-3-10.pkl'
pos = '/media/annina/elements/back-to-computer-240615/data_author/analysis/eng/nkpos/braycurtis-2000_simmel-3-10.pkl'
with open(path, 'rb') as file:
        data = pickle.load(file)
with open(pos, 'rb') as file:
        pos = pickle.load(file)
similarity_df = data.mx
# Create a NetworkX graph from the similarity DataFrame
G = nx.Graph()

# Add nodes
for node in similarity_df.index:
    G.add_node(node)

# Add edges with weights
for i in range(similarity_df.shape[0]):
    for j in range(i + 1, similarity_df.shape[1]):
        node_i = similarity_df.index[i]
        node_j = similarity_df.columns[j]
        weight = similarity_df.iloc[i, j]
        if weight > 0:
            G.add_edge(node_i, node_j, weight=weight)

# Generate positions using graphviz_layout
# pos = graphviz_layout(G, prog='neato')

# Draw the graph
plt.figure(figsize=(15, 15))
nx.draw_networkx(G, pos, with_labels=True, node_size=50, node_color='lightblue', edge_color='gray', font_size=5, font_weight='bold')
plt.title('Network from Similarity Matrix')
# plt.show()
plt.savefig('braycurtis-2000_simmel-3-10.png')
# %%
