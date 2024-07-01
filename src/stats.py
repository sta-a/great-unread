# %%
# %load_ext autoreload
# %autoreload 2

'''
Functions and classes for calculating and plotting essential metrics of the corpus.
'''

import numpy as np
from scipy.stats import linregress

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
from collections import Counter
from scipy.stats import skew
import os
from utils import DataHandler, get_filename_from_path, get_files_in_dir, DataLoader, TextsByAuthor
from cluster.cluster_utils import MetadataHandler
#from hpo_functions import get_data, ColumnTransformer

class MetadataStats(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='svg')


    def get_stats(self):
        print(f'-------Language: {self.language}----------------------------\n')
        # df = DataLoader(self.language).prepare_features(scale=True)
        mh = MetadataHandler(self.language)
        df = mh.get_metadata(add_color=False)
        # Get the counts of gender
        gender_counts = df['gender'].value_counts()
        print("Gender counts:")
        print(gender_counts)

        # Get the number of different authors 
        num_authors = df['author'].nunique()
        print(f"\nNumber of different authors: {num_authors}")


        # Group the data by gender and count the distinct authors
        author_counts = df.groupby('gender')['author'].nunique()
        # Print the result
        print(author_counts)


        max_year = df['year'].max()
        print(f"The highest year is: {max_year}")

        # Find the minimum (lowest) value in the 'year' column
        min_year = df['year'].min()
        print(f"The lowest year is: {min_year}")



class TextStatistics(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='svg')
        self.text_tokenized_dir = self.text_raw_dir.replace('/text_raw', '/text_tokenized') 

    def get_longest_shortest_text(self):
        text_lengths= {}
        # tokenized_paths = get_files_in_dir(self.text_tokenized_dir)
        for tokenized_path in self.doc_paths: ##############################3
            with open(tokenized_path, 'r') as f:
                text = f.read().split()
                text_lengths[get_filename_from_path(tokenized_path)] = len(text)
                

        text_lengths= dict(sorted(text_lengths.items(), key=lambda item: item[1]))

        key_max = max(text_lengths.keys(), key=(lambda k: text_lengths[k]))
        key_min = min(text_lengths.keys(), key=(lambda k: text_lengths[k]))

        print('Maximum Value: ', key_max, text_lengths[key_max])
        print('Minimum Value: ', key_min, text_lengths[key_min])

        # Print the shortest 10 texts and their lengths
        print('Shortest 10 Texts:')
        shortest_texts = list(text_lengths.keys())[:10]  # Get the list of shortest text names
        print(shortest_texts)
        for key in shortest_texts:
            print(key, text_lengths[key])


        title = (
            f'Maximum Value: {key_max}, {text_lengths[key_max]}\n'
            f'Minimum Value: {key_min}, {text_lengths[key_min]}\n'
        )
        self.plot_words_per_text(text_lengths, title)

    def plot_words_per_text(self, text_lengths, title):
        plt.figure(figsize=(10, 6))
        # x_values = range(len(text_lengths)) 
        plt.bar(text_lengths.keys(), text_lengths.values())
        plt.xlabel('Document')
        plt.ylabel('Number of Words')
        plt.title('Number of Words per Text')
        plt.xticks(rotation=45)
        plt.xticks([])  # Remove x-axis labels
        plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        plt.tight_layout()
        plt.text(0.9, 0.95, title, transform=plt.gca().transAxes, va='top',
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
        # plt.show()
        self.save_data(data=plt, file_name='words-per-text')
        plt.close()



class PlotCanonscoresBarChart(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='svg')
        self.df = DataLoader(self.language).prepare_metadata(type='canon')

    
    def plot(self):
        df = self.df.sort_values(by='canon')

        min_score = df['canon'].min()
        max_score = df['canon'].max()

        print(f"Minimum Score: {min_score}")
        print(f"Maximum Score: {max_score}")

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df)), df['canon'])
        plt.title('Canon Scores Distribution')
        plt.xlabel('Texts')
        plt.ylabel('Scores')
        
        print(df)
        x_ticks = range(0, len(df), 100)
        # plt.xticks(x_ticks, df.iloc[x_ticks]['file_name'], rotation=45, ha='right')
        plt.xticks(x_ticks, [str(i) for i in x_ticks], rotation=45, ha='right')

        self.save_data(data=plt, data_type='svg', file_name=f'canon-scores')
        plt.close()


class PlotFeatureDist(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='svg')
        self.cat_attrs = ['author', 'gender']


    def plot(self):
        # df = DataLoader(self.language).prepare_features(scale=True)
        mh = MetadataHandler(self.language)
        df = mh.get_metadata(add_color=False)


        nrows = 14 # 13 if only text features (without author, gender, year, canon) are plotted
        ncols = 7
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 40))

        assert len(df.columns) <= (nrows * ncols)

        # Flatten the 2D array of axes for easier indexing
        axes = axes.flatten()

        for i, column in enumerate(df.columns):
            df[column].hist(ax=axes[i], bins=20)
            if column not in self.cat_attrs:
                skewness = skew(df[column])
                axes[i].set_title(f'{column},{round(skewness, 2)}')
            else:
                axes[i].set_title(f'{column}')

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Save the figure
        self.save_data(data=plt, data_type='svg', file_name=f'feature-dist')
        plt.close()



class PlotCanonScoresPerAuthor(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='png', data_dir='/home/annina/scripts/great_unread_nlp/data')


    def make_plot(self):
        self.combine_author_work_metadata()
        self.get_x_position()
        self.plot()


    def combine_author_work_metadata(self):
        # df = DataLoader(self.language).prepare_features(scale=True)
        mh = MetadataHandler(self.language)
        df = mh.get_metadata(add_color=True)# [['canon', 'gender', 'year']]
        # author_filename_mapping = TextsByAuthor(self.language).author_filename_mapping
        # author = pd.DataFrame([(author, work) for author, works in author_filename_mapping.items() for work in works],
        #                 columns=['author', 'file_name'])
        # self.df = df.merge(author, left_index=True, right_on='file_name', validate='1:1')
        self.df = df


    def get_sorted_authors(self):
        mean_canon = self.df.groupby('author')['canon'].mean()
        self.sorted_authors = mean_canon.sort_values().index


    def get_x_position(self):
        self.get_sorted_authors()
        author_positions = {author: i for i, author in enumerate(self.sorted_authors)}
        self.df['author_position'] = self.df['author'].map(author_positions)

    
    def plot(self):
        fig, ax = plt.subplots(figsize=(25, 8))
        for author, group in self.df.groupby('author'):
            x_position = group['author_position'].iloc[0]  # Take the x-position from the first row of the group
            ax.scatter([x_position] * len(group), group['canon'], c='b', s=5)
            ax.vlines(x_position, ymin=0, ymax=group['canon'].max(), linestyle='dotted', color='gray', linewidth=1) # ymin=group['canon'].min()

        ax.set_xlabel('Author')
        ax.set_ylabel('Canon Score')
        # ax.set_title('Canon Scores by Author')


        ax.set_xticks(range(len(self.sorted_authors)))
        ax.set_xticklabels(self.sorted_authors, rotation=45, ha='right', fontsize=7)

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        self.save_data(data=plt, file_name='canonscores_per_author')



class PlotCanonScoresPerAuthorByYearAndGender(PlotCanonScoresPerAuthor):
    '''
    Plot earliest publication year.
    '''
    def __init__(self, language):
        super().__init__(language)
        self.fontsize = 18

    def get_sorted_authors(self):
        self.df['min_year'] = self.df.groupby('author')['year'].transform('min')
        self.df['author_min_year'] = self.df['author'] + ' (' + self.df['min_year'].astype(str) + ')'
        self.df = self.df.sort_values(by='min_year', ascending=True)
        self.sorted_authors = self.df.index


    def get_x_position(self):
        self.get_sorted_authors()
        self.df['author_position'] = self.df['min_year']

        # Slightly adjust position if several authors have same min_year
        df_unique_author = self.df.drop_duplicates(subset=['author'])
        duplicated_min_year = df_unique_author['min_year'].duplicated().any()
        space_increment = 0.1
        if duplicated_min_year:
            duplicated_min_year_rows = df_unique_author[df_unique_author.duplicated(subset=['min_year'], keep=False)]
            for min_year, group in duplicated_min_year_rows.groupby('min_year'):
                assert len(group) < 10 # not more than 10 entries per group because space increment is 0.1
                increment = 0
                for _, row in group.iterrows():
                    # print('rowname', row.name)
                    self.df.loc[row.name, 'author_position'] += increment
                    increment += space_increment

        self.df['author_position'].to_csv('authorpos')


    def plot(self):
        fig, ax = plt.subplots(figsize=(18, 6))
        for author, group in self.df.groupby('author'):
            x_position = group['author_position'].iloc[0]  # Take the x-position from the first row of the group
            color = group['gender_color']
            ax.scatter([x_position] * len(group), group['canon'], c=color, s=6)
            ax.vlines(x_position, ymin=0, ymax=group['canon'].max(), linestyle='dotted', color=color, linewidth=1) # ymin=group['canon'].min()

            # Position the label at the topmost marker
            # max_canon = group['canon'].max()  # Find the maximum canon score in the group
            # ax.text(x_position, max_canon, author, ha='center', va='bottom', fontsize=3, rotation=45)


        ax.set_xlabel('Year of first publication by author', fontsize=self.fontsize)
        ax.set_ylabel('Canon Score', fontsize=self.fontsize)
        # ax.set_title('Canon Scores by Author, Year and Gender')

        # ax.set_xticks(self.df['author_position'])
        # ax.set_xticklabels(self.df['author_min_year'], rotation=45, ha='right', fontsize=1) # Position labels at the bottom


        # Generate x-ticks every 50 years
        min_year = int(self.df['min_year'].min())
        max_year = int(self.df['min_year'].max())
        ax.set_xticks(range(min_year, max_year + 1, 50))

        # Change the font size of the x and y tick labels
        ax.tick_params(axis='both', which='major', labelsize=self.fontsize)  # Adjust the value of labelsize as desired


        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        self.save_data(data=plt, file_name='canonscores_per_author_and_year')



class PlotYearAndCanon(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='png')
        self.by_author = False
        self.yeartup = ('year', 'Publication Year')
        self.canontup = ('canon', 'Canonization Score')
        self.canonmin_tup = ('canon-min', 'Canonization Score')
        self.canonmax_tup = ('canon-max', 'Canonization Score')

    def plot_single_var(self):
        if not self.by_author:
            plotlist =  [self.yeartup, self.canontup]
        else:
            plotlist =  [self.yeartup, self.canontup, self.canonmin_tup , self.canonmax_tup]

        for attr, yaxis_title in plotlist:
            mh = MetadataHandler(self.language, by_author=self.by_author)
            metadf = mh.get_metadata(add_color=False)
            print(metadf.shape)
            df = metadf.sort_values(by=attr, ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df.index, df[attr], s=3)

            ax.set_xlabel('Number of Texts')
            ax.set_ylabel(yaxis_title)
            ax.grid(True, linestyle='--', alpha=0.5)
            # ax.set_title('Publication Years')

            # Remove x-axis labels
            # ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            max_text_index = len(df)
            text_ticks = list(range(0, max_text_index + 1, 100))
            ax.set_xticks(text_ticks)
            ax.set_xticklabels([str(t) for t in text_ticks])


            self.save_data(data=plt, file_name=f'{attr}_byauthor-{self.by_author}')

    def plot_two_vars(self):
        yearattr, yeartitle = self.yeartup
        canonlist = [self.canontup]
        if self.by_author:
            canonlist.extend([self.canonmin_tup, self.canonmax_tup])

        for canonattr, canontitle in canonlist:
            self.make_plot(xattr=yearattr, yattr=canonattr, xaxis_title=yeartitle, yaxis_title=canontitle)
            self.make_plot(yattr=yearattr, xattr=canonattr, yaxis_title=yeartitle, xaxis_title=canontitle)



    # def make_plot(self, xattr, yattr, xaxis_title, yaxis_title):
    #     mh = MetadataHandler(self.language, by_author=self.by_author)
    #     metadf = mh.get_metadata(add_color=False)
    #     print(metadf.shape)
    #     df = metadf.sort_values(by=xattr, ascending=True)

    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     ax.scatter(df[xattr], df[yattr], s=3)

    #     ax.set_xlabel(xaxis_title)
    #     ax.set_ylabel(yaxis_title)
    #     ax.grid(True, linestyle='--', alpha=0.5)

    #     self.save_data(data=plt, file_name=f'{xattr}-{yattr}_byauthor-{self.by_author}')


    def make_plot(self, xattr, yattr, xaxis_title, yaxis_title):
        mh = MetadataHandler(self.language, by_author=self.by_author)
        metadf = mh.get_metadata(add_color=False)
        print(metadf.shape)
        df = metadf.sort_values(by=xattr, ascending=True)

        # Perform linear regression
        x = df[xattr].values
        y = df[yattr].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        regression_line = slope * x + intercept

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, s=3, label='Data points')
        ax.plot(x, regression_line, color='red', label=f'Regression line: y={slope:.2f}x+{intercept:.2f}')

        ax.set_xlabel(xaxis_title)
        ax.set_ylabel(yaxis_title)
        ax.grid(True, linestyle='--', alpha=0.5)
        # ax.legend()

        # Save the figure
        self.save_data(data=plt, file_name=f'{xattr}-{yattr}_byauthor-{self.by_author}')

        # Print regression details
        print(f'Regression line: y = {slope:.2f}x + {intercept:.2f}')
        print(f'R-squared: {r_value**2:.2f}')
        print(f'P-value: {p_value:.2e}')
        print(f'Standard error: {std_err:.2f}')

















# for language in ['eng', 'ger']:
#     print('Text Statistics:')

#     pyac = PlotYearAndCanon(language)
#     pyac.plot_single_var()
#     pyac.plot_two_vars()
    # mdstats = MetadataStats(language)
    # mdstats.get_stats()
    # ts = TextStatistics(language)
    # ts.get_longest_shortest_text()

    # pc = PlotCanonscoresBarChart(language)
    # pc.plot()

    # pfd = PlotFeatureDist(language)
    # pfd.plot()


    # pcspa = PlotCanonScoresPerAuthor(language)
    # pcspa.make_plot()

    # p = PlotCanonScoresPerAuthorByYearAndGender(language)
    # p.make_plot()






# %%
# task = 'regression-importances'
# label_type = 'canon'
# # Importances are calculated on cacb features
# features = 'book'

# data_dir = '../data'
# canonscores_dir = os.path.join(data_dir, 'canonscores')
# n_outer_folds = 5



# %%
# Nr reviewed texts after contradicting labels were removed: 191
# for language in languages:
#     # Use full features set for classification to avoid error with underrepresented classes
#     sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
#     metadata_dir = os.path.join(data_dir, 'metadata', language)
#     features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
#     gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
#     if not os.path.exists(gridsearch_dir):
#         os.makedirs(gridsearch_dir, exist_ok=True)


#     ## Sentiscores
#     # Review year 
#     if language == 'eng':
#         sentiscores_file = 'ENG_reviews_senti_FINAL.csv'
#     else:
#         sentiscores_file = 'GER_reviews_senti_FINAL.csv'
#     sentiscores_df = pd.read_csv(os.path.join(sentiscores_dir, sentiscores_file), header=0, sep=';')

#     # Nr of different texts that were reviewed
#     nr_reviewed_texts = len(sentiscores_df.file_name.unique())

#     sentiscores_df = sentiscores_df[['file_name', 'textfile', 'pub_year']]
#     # ID des reviewten Texts _ Jahr der Rezension _ Kürzel der Zeitschrift _ Seitenangabe
#     sentiscores_df['review_year'] = sentiscores_df['textfile'].str.split(pat='_', expand=True)[1]
#     #sentiscores_df['review_year'].sort_values().to_csv(f'review-year_{language}')


#     ## Library Scores
#     if language == 'eng':
#         lib_file = 'ENG_texts_circulating-libs.csv'
#     else:
#         lib_file = 'GER_texts_circulating-libs.csv'
#     lib_df = pd.read_csv(os.path.join(metadata_dir, lib_file), header=0, sep=';')[['file_name', 'pub_year']]
#     old_lib_df = pd.read_csv('/home/annina/scripts/great_unread_nlp/data/metadata_old/ger/GER_texts_circulating-libs.csv', header=0, sep=';')[['file_name', 'pub_year']]

#     # publication year
#     lib_df['publication_year'] = lib_df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
#     lib_df = lib_df.loc[lib_df['pub_year'] != lib_df['publication_year']]
#     lib_df = lib_df.drop(columns='publication_year')
#     lib_df = lib_df.sort_values(by='file_name')
#     lib_df.to_csv(f'publication-year_{language}.csv', index=False)


#     old_lib_df['publication_year'] = old_lib_df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
#     old_lib_df = old_lib_df.loc[old_lib_df['pub_year'] != old_lib_df['publication_year']]
#     old_lib_df = old_lib_df.drop(columns='publication_year')
#     old_lib_df = old_lib_df.sort_values(by='file_name')
#     old_lib_df.to_csv(f'publication-year-olddata_{language}.csv', index=False)

#     if language == 'ger':
#         difference = lib_df.merge(old_lib_df, how='outer', on='file_name')
#         difference.to_csv(f'difference-old-new-libdata_{language}.csv', header=True, index=False)

    
#     new_file_names = pd.read_csv('/home/annina/scripts/great_unread_nlp/src/new_file_names.csv', header=0)[['file_name', 'file_name_new']]
#     print(new_file_names)


