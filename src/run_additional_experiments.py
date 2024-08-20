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

    def analyze_clusters(self):
        # Filter rows where 'nclust' is 50 or bigger
        df = self.df[self.df['nclust'] >= 50]
        df = df[df['attr'] == 'author']
        df = df[df['silhouette_score'] >=0.2] # lower silhouette score finds good results for English
        df = df[df['ARI'] > 0]

        # df = df.sort_values(by='silhouette_score', ascending=False)
        df = df.sort_values(by='ARI', ascending=False)
        print(df)
        self.save_data(data=df, file_name=os.path.basename(self.path), subdir=True)

        df['plttitle'] = 'empty-plttitle'
        # exp = {'name': self.__class__.__name__, 'ntop': 30, 'maxsize': 90, 'intthresh': 0.2, 'viztype': 'keyattr'}
        exp = {'name': self.__class__.__name__, 'ntop': 30, 'maxsize': 90, 'intthresh': 0.2, 'viztype': 'singleimage_extra_experimtents'}
        te = TopEval(language=self.language, output_dir='similarity', cmode='mx', exp=exp, expdir=None, df=df, by_author=self.by_author)
        experiment = Experiment(language=self.language, cmode='mx', by_author=self.by_author, output_dir=self.output_dir)
        experiment.visualize_mx(exp=exp, te=te)





class AuthorshipAttribution(DataHandler):
    '''
    Check the most similar texts for all texts where the author has more than on text in the corpus.
    '''
    def __init__(self, language, output_dir='extraexp', by_author=False, mxmode='unsparsified'):
        super().__init__(language, output_dir=output_dir, by_author=by_author, data_type='csv')
        self.mh = MetadataHandler(self.language, by_author=self.by_author)
        self.metadf = self.mh.get_metadata(add_color=False)
        self.mxmode = mxmode
        self.add_subdir(f'{self.__class__.__name__}')

    def load_mxs(self):
        # Copied from CombinationsBase
        # Delta distance mxs
        delta = Delta(self.language, by_author=self.by_author)
        all_delta = delta.load_all_data(use_kwargs_for_fn='mode', subdir=True)

        # D2v distance mxs
        d2v = D2vDist(language=self.language, by_author=self.by_author)
        all_d2v = d2v.load_all_data(use_kwargs_for_fn='mode', file_string=d2v.file_string, subdir=True)

        mxs = {**all_delta, **all_d2v}

        for name, mx in mxs.items():
            if 'mirror' in name:
                print('"mirror" was contained in the wrong subdir')
            mx.name = name
            yield mx

    def load_spars_mxs(self):
        sparsdir = os.path.join(self.data_dir, 'similarity', self.language, 'sparsification')
        pickle_files = [f for f in os.listdir(sparsdir) if f.endswith('.pkl')]
        for file_name in pickle_files:
            file_path = os.path.join(sparsdir, file_name)

            mxname = file_name.split('.')[0]
            mxname = mxname.replace('sparsmx-', '')
            
            # Load the matrix from the pickle file
            with open(file_path, 'rb') as file:
                mx = pickle.load(file)
            mx.name = mxname            
            yield mx


    def analyze_authors(self):
        # Step 1: Precompute frequently used values
        author_counts = self.metadf['author'].value_counts()
        authors_with_multiple_entries = author_counts[author_counts > 1].index
        
        # Filter the DataFrame to keep only rows with these authors
        df_reduced = self.metadf[self.metadf['author'].isin(authors_with_multiple_entries)]
        file_to_author = self.metadf['author'].to_dict()  # Convert Series to a dictionary for faster lookups

        # Use the reduced DataFrame to get texts per author
        texts_per_author = df_reduced['author'].value_counts().to_dict()  # Convert to dict for fast lookups

        all_mxs_result = {}
        if self.mxmode == 'unsparsified':
            mxs_generator = self.load_mxs()
        else:
            mxs_generator = self.load_spars_mxs()
        for mxobj in mxs_generator:
            print(mxobj.name)
            mx = mxobj.mx

            # Step 2: Set the diagonal of mx to 0
            np.fill_diagonal(mx.values, 0)

            # Step 3: Filter rows of mx to include only those with authors in df_reduced
            filtered_mx = mx.loc[df_reduced.index.intersection(mx.index)]  # Intersection ensures we're only keeping valid rows
            nr_text_with_nonunique_author = len(filtered_mx)
            assert nr_text_with_nonunique_author == len(df_reduced)

            # Step 4: Initialize dictionaries to store the counts
            # top_n_counts = {}  # Store the n-highest counts
            # top_counts = {}  # Store the maximum value counts
            top_n_counts = []
            top_counts = []
            ranks = []

            # Step 5: Vectorized operations to make the process faster
            for file_name in filtered_mx.index:
                # Step 5.1: Get the author of the current file
                author = file_to_author[file_name]
                
                # Step 5.2: Find how many times the author occurs in df['author']
                n = texts_per_author.get(author, 0) - 1  # Number of neighbors is x-1

                if n <= 0:
                    continue  # Skip if there's no valid neighbor count

                # Step 5.3: Get the current row of the matrix
                current_row = filtered_mx.loc[file_name]

                # Step 5.4: Find the n largest values in the row
                n_highest_indices = np.argpartition(current_row.values, -n)[-n:]  # Fast partial sort to get top n indices
                n_highest_authors = [file_to_author[filtered_mx.columns[idx]] for idx in n_highest_indices]

                # Step 5.5: Count how many of the top n match the author
                count_matching_authors = sum(1 for match_author in n_highest_authors if match_author == author)
                
                # Step 5.6: Store the count for top_n_matches
                # top_n_counts.setdefault(file_name, []).append(count_matching_authors/n)
                top_n_counts.append(count_matching_authors/n)
                
                # Step 5.7: Find the index of the maximum value in the current row
                max_idx = current_row.argmax()  # Fast method to find the index of the maximum value

                # Step 5.8: Compare the author of the max value text with the current author
                max_author = file_to_author[filtered_mx.columns[max_idx]]
                max_match = int(max_author == author)

                # Step 5.9: Store the result for the maximum match
                # top_counts.setdefault(file_name, []).append(max_match)
                top_counts.append(max_match)

                # Step 5.10: Rank-based evaluation
                sorted_indices = np.argsort(-current_row.values)  # Sort indices by similarity in descending order
                rank = np.where([file_to_author[filtered_mx.columns[idx]] == author for idx in sorted_indices])[0]
                if len(rank) > 0:
                    ranks.append(rank[0] + 1)  # +1 to convert from zero-based to one-based rank



            avg_top_n_counts = sum(top_n_counts) / nr_text_with_nonunique_author
            avg_top_counts = sum(top_counts) / nr_text_with_nonunique_author
            avg_rank = np.mean(ranks) if ranks else float('inf')  # Handle the case where ranks list might be empty
            
            
            # Store the results in the dictionary
            all_mxs_result[mxobj.name] = [round(avg_top_counts, 3), round(avg_top_n_counts, 3), round(avg_rank, 3)]

        # Convert the final result dictionary to a DataFrame
        results_df = pd.DataFrame.from_dict(all_mxs_result, orient='index', columns=['avg_top_counts', 'avg_top_n_counts', 'avg_rank'])
        results_df = results_df.sort_values(by='avg_top_counts', ascending=False)
        print(results_df)
        results_path = os.path.join(self.subdir, f'authorship-{self.mxmode}.csv')
        print(results_path)
        results_df.to_csv(results_path, header=True, index=True)


for language in ['eng', 'ger']:
    ama = AnalysisMxAuthor(language=language)
    ama.analyze_clusters()
    # ama.find_max_nr_clusts_dbscan()

    # for mxmode in ['unsparsified', 'sparsified']:
    #     aa = AuthorshipAttribution(language=language, mxmode=mxmode)

    #     aa.analyze_authors()

    # Reorder columns
    # for mxmode in ['unsparsified', 'sparsified']:
    #     path = f'/home/annina/scripts/great_unread_nlp/data/extraexp/{language}/AuthorshipAttribution/authorship-{mxmode}.csv'

    #     # Load the DataFrame
    #     df = pd.read_csv(path, index_col=0)

    #     # Reorder columns
    #     df = df[['avg_top_n_counts', 'avg_top_counts', 'avg_rank']]

    #     # Sort descendingly by 'avg_top_n_counts'
    #     df = df.sort_values(by='avg_top_n_counts', ascending=False)

    #     # Save the modified DataFrame if needed
    #     df.to_csv(path)

# %%

# %%
import os

# Get the current directory
current_dir = '/media/annina/MyBook/back-to-computer-240615/data/analysis/ger'
# List to store directories and their file counts
dir_file_counts = []

# Iterate through each item in the current directory
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    
    # Check if the item is a directory
    if os.path.isdir(item_path) and 'mx' in item_path and 'nclust-50' in item_path and 'author' in item_path:
        # Count the number of files in the directory
        num_files = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
        
        # Add the directory and file count to the list
        dir_file_counts.append((item, num_files))

# Sort the list by the number of files (second element of the tuple)
dir_file_counts.sort(key=lambda x: x[1], reverse=True)

for idx, (dir_name, file_count) in enumerate(dir_file_counts, start=1):
    print(f"{idx}. Directory: {dir_name}, Number of files: {file_count}")
# %%
