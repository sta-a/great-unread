# %%

import sys
sys.path.append("..")
from utils import TextsByAuthor, DataHandler
import pandas as pd
import os
import numpy as np
import shutil
from copy import deepcopy
import pickle
from analysis.experiments import Experiment
from analysis.topeval import TopEval
from cluster.create import D2vDist, Delta
from cluster.cluster_utils import MetadataHandler
from cluster.cluster_utils import ColorBar
from utils import copy_imgs_from_harddrive

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
        self.path = f'/media/annina/elements/back-to-computer-240615/data_author/similarity/{self.language}/mxeval/cont_results.csv'
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
        # self.save_data(data=df, file_name=os.path.basename(self.path), subdir=True)
        # df['plttitle'] = 'empty-plttitle'
        # exp = {'name': self.__class__.__name__, 'ntop': 30, 'maxsize': 90, 'intthresh': 0.2, 'viztype': 'singleimage_extra_experimtents'}
        # te = TopEval(language=self.language, output_dir='similarity', cmode='mx', exp=exp, expdir=None, df=df, by_author=self.by_author)
        # experiment = Experiment(language=self.language, cmode='mx', by_author=self.by_author, output_dir=self.output_dir)
        # experiment.visualize_mx(exp=exp, te=te)


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


class YearClusterQuantiles(DataHandler):
    def __init__(self, language=None, output_dir='extraexp', by_author=True):
        super().__init__(language=language, output_dir=output_dir, data_type='csv', by_author=by_author)

        self.add_subdir(f'nk_{self.__class__.__name__}')
        self.path = '/media/annina/elements/back-to-computer-240615/data_author/similarity/eng/nkeval/cont_results.csv'
        self.df = pd.read_csv(self.path)
        self.attrs = ['canon', 'year']
        self.evalcol = 'avg_variance'


    def get_quantiles(self):
        df = self.df[self.df['attr'].str.contains('|'.join(self.attrs), case=False, na=False)]
        colslist = ['mxname', 'sparsmode', 'clst_alg_params', self.evalcol, 'nclust', 'clst_sizes', 'modularity', 'attr', 'file_info']
        df = df[colslist]

        # Filter with quality criteria
        df = df[df['modularity'] >= 0.3] 

        # Compute the quartiles
        canon_threshold = df[df['attr'] == 'canon'][self.evalcol].quantile(0.2)
        year_threshold = df[df['attr'] == 'year'][self.evalcol].quantile(0.8)

        # Step 2: Filter rows
        canon_quantile_df = df[(df['attr'] == 'canon') & (df[self.evalcol] <= canon_threshold)]
        year_quartile_df = df[(df['attr'] == 'year') & (df[self.evalcol] >= year_threshold)]


        # Find pairs
        df = pd.merge(canon_quantile_df, year_quartile_df, on=['mxname', 'sparsmode', 'clst_alg_params'], suffixes=('_canon', '_year'))
        assert (df['nclust_year'] == df['nclust_canon']).all()
        df = df.drop(columns=['nclust_canon'])
        df = df.rename(columns={'nclust_year': 'nclust'})
        assert (df['clst_sizes_year'] == df['clst_sizes_canon']).all()
        df = df.drop(columns=['clst_sizes_canon'])
        df = df.rename(columns={'clst_sizes_year': 'clst_sizes'})
        df = df.rename(columns={'file_info_canon': 'file_info'}) # file info, CombinationInfo that is loaded is for canon

        print(year_quartile_df.shape, canon_quantile_df.shape, df.shape)
        # 'pairs_df' contains the pairs where 'canon' is in the uppermost quartile and 'year' is in the lowest quartile.
        df = extend_sizes_col(df)
        df = filter_clst_sizes(df)

        self.save_data(data=df, file_name='quantiles_df.csv', subdir=True)
        
        df['plttitle'] = 'empty-plttitle'
        exp = {'name': self.__class__.__name__, 'ntop': 1, 'maxsize': 90, 'intthresh': 0.3, 'viztype': 'keyattr'}
        te = TopEval(language=self.language, output_dir='similarity', cmode='nk', exp=exp, expdir=None, df=df, by_author=self.by_author)
        experiment = Experiment(language=self.language, cmode='mx', by_author=self.by_author, output_dir=self.output_dir)
        experiment.visualize_nk(exp=exp, te=te, subdir=self.subdir)



# for language in ['eng', 'ger']:
for language in ['eng']:
    # ama = AnalysisMxAuthor(language=language)
    # ama.analyze_clusters_with_vmeasure()
    # ama.analyze_clusters_with_lower_silhouette()

    # ama = AnalysisNkGender(language=language)
    # ama.analyze_homogeneity()

    # ama = AnalysisMxYear(language=language)
    # ama.analyze_clusters()

    # ama = AnalysisNkCanonByAuthor(language=language, by_author=True)
    # ama.analyze_clusters()

    ycq = YearClusterQuantiles(language)
    ycq.get_quantiles()




# %%
import os

# Get the current directory
current_dir = '/media/annina/elements/back-to-computer-240615/data_author/analysis/ger'
# List to store directories and their file counts
dir_file_counts = []

# Iterate through each item in the current directory
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    
    # Check if the item is a directory
    if os.path.isdir(item_path) and 'mx' in item_path and 'canon' in item_path:
        # Count the number of files in the directory
        num_files = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
        
        # Add the directory and file count to the list
        dir_file_counts.append((item, num_files))

# Sort the list by the number of files (second element of the tuple)
dir_file_counts.sort(key=lambda x: x[1], reverse=True)

for idx, (dir_name, file_count) in enumerate(dir_file_counts, start=1):
    print(f"{idx}. Directory: {dir_name}, Number of files: {file_count}", os.path.join(current_dir, dir_name))



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
for language in ['eng', 'ger']:
    for byauthor in [False, True]:
        cb = ColorBar(language, byauthor)
        # cb.make_bars()
        cb.add_both_labels()

# %%


# copy_imgs_from_harddrive(source)

# for language in ['eng', 'ger']:
#     for datadir in ['data', 'data_author']:
#         source = f'/media/annina/elements/back-to-computer-240615/{datadir}/colorbar/{language}/colorbar-seismic_year_and_canon.png'
#         copy_imgs_from_harddrive(source)

# %%
from utils import copy_imgs_from_harddrive

# Function to generate LaTeX code for a figure with subfigures and write it to file
def write_complex_latex_figure_authorbased(outf, paths, distance, label, basedstr, langstr):
    # LaTeX code template with paths filled using positional indexing from the list
    latex_code = outf.write(f'''
\n\n\\subsection{{{distance.upper(), langstr, basedstr}}}\n
\\begin{{figure}}[H]
    \\centering
    \\captionsetup[subfigure]{{labelformat=empty}} % Suppress automatic subfigure labeling for this figure only
    % Top row
    \\begin{{subfigure}}[t]{{\\textwidth}}
        \\centering
        \\begin{{subfigure}}[t]{{0.31\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{paths[0]}}}
            \\caption{{Year}}
        \\end{{subfigure}}
        \\begin{{subfigure}}[t]{{0.31\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{paths[2]}}}
            \\caption{{Canon}}
        \\end{{subfigure}}
        \\begin{{subfigure}}[t]{{0.31\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{paths[4]}}}
            \\caption{{Gender}}
        \\end{{subfigure}}
    \\end{{subfigure}}

    % Bottom row
    \\begin{{subfigure}}[t]{{\\textwidth}}
        \\centering
        \\begin{{subfigure}}[t]{{0.31\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{paths[1]}}}
            \\caption{{Year}}
        \\end{{subfigure}}
        \\begin{{subfigure}}[t]{{0.31\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{paths[3]}}}
            \\caption{{Canon}}
        \\end{{subfigure}}
        \\begin{{subfigure}}[t]{{0.31\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{paths[5]}}}
            \\caption{{Gender}}
        \\end{{subfigure}}
    \\end{{subfigure}}
    \\caption{{Distance \\dist{{{distance}}}, sparsified with \\sparsname{{simmel-5-10}}}}
    \\label{{{label}}}
\\end{{figure}}
''')



# Write the generated LaTeX code to the specified file path
with open('latextest', 'w') as outf:
    for data_dir in ['data_latex', 'data_author_latex']:
        if data_dir == 'data_latex':
            ba = False
            basedstr = 'text-based'
        else:
            ba = True
            basedstr = 'author-based'
        for distance in ['full', 'both']:
            for language in ['eng', 'ger']:
                if language == 'eng':
                    langst = 'English'
                else:
                    langst = 'German'
                label = f'fig:embimgs-{distance}-{language}-byauthor-{ba}'

                paths = []
                for attr in ['year', 'canon', 'gender', 'author']:
                    if data_dir == 'data_author_latex' and attr == 'author':
                        continue
                    paths.append(f'/home/annina/Documents/thesis/{data_dir}/analysis/{language}/nk_singleimage_appendix/{distance}_simmel-5-10_{attr}.png',)
                    paths.append(f'/home/annina/Documents/thesis/{data_dir}/analysis_s2v/{language}/mx_singleimage_s2v/s2v-{distance}_simmel-5-10_dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True_{attr}.png',)

                for path in paths:
                    copy_imgs_from_harddrive(path)
                if data_dir == 'data_author_latex':
                    write_complex_latex_figure_authorbased(outf, paths, distance, label, basedstr, langst)
                else:


# %%
# from utils import copy_imgs_from_harddrive
# source = '/home/annina/Documents/thesis/data_author_latex/analysis_s2v/ger/mx_singleimage_s2v/s2v-chebyshev-2000_threshold-0.90_dimensions-16_walklength-30_numwalks-200_windowsize-15_untillayer-5_OPT1-True_OPT2-True_OPT3-True_canon.png'
# copy_imgs_from_harddrive(source=source)
# %%



# %%
# Combine all networks per spars into one plot
from analysis.viz_utils import ImageGrid
import os

class NkImgGrid(ImageGrid):
    def __init__(self, language, by_author=False, output_dir='extraexp', imgdir='nk_singleimage_appendix', imgs_as_paths=True):
        super().__init__(language=language, by_author=by_author, output_dir=output_dir, imgdir=imgdir, imgs_as_paths=imgs_as_paths)
        self.nrow = 9
        self.ncol = 7
        self.img_width = 1.2
        self.img_height = 1.4

    def get_title(self, imgname):
        name = os.path.basename(imgname)
        name = name.split('.')[0]
        return name.split('_')[0]


ndist = 58
if os.path.exists('/media/annina/elements/back-to-computer-240615/data/extraexp/eng/nkimggrid/viz.png'):
    os.remove('/media/annina/elements/back-to-computer-240615/data/extraexp/eng/nkimggrid/viz.png')
for language in ['eng', 'ger']:
    for datadir in ['data', 'data_author']:
        if datadir == 'data':
            nspars = 9
            by_author = False
        else:
            nspars = 7
            by_author = True
        cdir = f'/media/annina/elements/back-to-computer-240615/{datadir}/analysis/{language}/nk_singleimage_appendix'
        all_imgs = [x for x in os.listdir(cdir) if '_canon.png' in x]

        for spars in ['threshold-0%8', 'threshold-0%90', 'threshold-0%95', 'authormax', 'authormin','simmel-5-10', 'simmel-3-10', 'simmel-4-6', 'simmel-7-10']:
            print(language, datadir, spars)
            imgs = [os.path.join(cdir, x) for x in all_imgs if spars in x]
            nig = NkImgGrid(language, by_author=by_author, output_dir='extraexp', imgdir=None, imgs_as_paths=True)
            nig.visualize(imgs=imgs, vizname=spars)

# %%
# Create latex figures with plots from above
def write_latex_with_image(image_path, caption):
    latex_content = f"""

        \\begin{{figure}}[!ht]
            \\centering
            \\includegraphics[width=\\textwidth, height=\\textheight, keepaspectratio]{{{image_path}}}
            \\caption{{{caption}}}
            \\label{{fig:image1}}
        \\end{{figure}}

        """
    return latex_content        

import os
ndist = 58

with open('allnetworks.tex', 'w') as outf:
    for language in ['eng', 'ger']:
        if language == 'eng':
            langst = 'English'
        else:
            langst = 'German'
        for datadir in ['data', 'data_author']:
            if datadir == 'data':
                iba = False
                level = 'text-based'
            else:
                iba = True
                level = 'author-based'
            print(cdir)
            all_imgs = [x for x in os.listdir(cdir) if '_canon.png' in x]
            for spars in ['threshold-0%8', 'threshold-0%90', 'threshold-0%95', 'authormax', 'authormin','simmel-5-10', 'simmel-3-10', 'simmel-4-6', 'simmel-7-10']:
                if datadir == 'data_author':
                    if spars == 'authormin' or spars == 'authormax':
                        continue
                path = f'/media/annina/elements/back-to-computer-240615/{datadir}/extraexp/{language}/nkimggrid/{spars}.png'
                caption = f'All networks for {spars}, {level}, {langst}'
                latex_content = write_latex_with_image(path, caption)
                outf.write(latex_content)


# %%
