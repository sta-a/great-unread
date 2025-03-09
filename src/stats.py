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
from scipy.stats import skew, spearmanr
from scipy import stats
import os
import seaborn as sns
from scipy.stats import ttest_ind
from utils import DataHandler, get_filename_from_path, get_files_in_dir, DataLoader, TextsByAuthor
from cluster.cluster_utils import MetadataHandler
#from hpo_functions import get_data, ColumnTransformer




class YearAndCanonByLanguage(DataHandler):
    '''
    Create boxplots showing the distribution of year and canon by language
    Save the plots only once, in the 'eng' subdir
    '''
    def __init__(self, by_author):
        super().__init__(language='eng', output_dir='text_statistics', data_type='png', by_author=by_author)
        self.fontsize = 30


    def make_language_boxplots(self):
        mh = MetadataHandler('eng', by_author=self.by_author)
        df_eng = mh.get_metadata(add_color=False)
        mh = MetadataHandler('ger', by_author=self.by_author)
        df_ger = mh.get_metadata(add_color=False)

        # Add a 'language' column to each dataframe
        df_eng['language'] = 'English'
        df_ger['language'] = 'German'

        df = pd.concat([df_eng, df_ger])

        for attrtup in [('year', 'Year'), ('canon', 'Canon Score')]:
            attr, label = attrtup

            eng_attrs = df[df['language'] == 'English'][attr]
            ger_attrs = df[df['language'] == 'German'][attr]
            print(f'Average eng {attr}: ', eng_attrs.mean())
            print(f'Average ger {attr}: ', ger_attrs.mean())


            # Create the boxplot
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='language', y=attr, data=df)

            # Adding title and labels
            plt.xlabel('Language', fontsize=self.fontsize)
            plt.ylabel(label, fontsize=self.fontsize)
            # Changing the font size of the tick labels
            plt.xticks(fontsize=self.fontsize - 5)
            plt.yticks(fontsize=self.fontsize - 5)
            self.save_data(data=plt, file_name=f'{attr}-by-language_byauthor-{self.by_author}.png')




class YearAndCanonByGender(DataHandler):
    '''
    Create boxplots showing the distribution of year and canon by gender
    '''
    def __init__(self, language, by_author):
        super().__init__(language, output_dir='text_statistics', data_type='png', by_author=by_author)
        self.fontsize = 30

    def canon_year_correlation_by_gender(self):
        print('------------------------')
        print(language, by_author)
        print('------------------------')

        # Initialize a dictionary to store Spearman correlations for each gender and language
        spearman_data = {}

        mh = MetadataHandler(language, by_author=self.by_author)
        df = mh.get_metadata(add_color=False)
        print('Overall correlation:', spearmanr(df['canon'], df['year']))

        # Replace numeric gender codes with strings
        gender_map = {0: 'Male', 1: 'Female', 2: 'Anonymous', 3: 'Both'}
        df['gender'] = df['gender'].replace(gender_map)

        spearman_data = {}

        # Calculate Spearman's rank correlation for each gender
        for gender in ['Male', 'Female', 'Anonymous']:
            gender_df = df[df['gender'] == gender]
            corr, _ = spearmanr(gender_df['canon'], gender_df['year'])
            spearman_data[gender] = corr

        # Print Spearman correlations
        for gender, corr in spearman_data.items():
            print(f'\nSpearman\'s Rank Correlation for {gender}:')
            print(f'  {gender}: {round(corr, 3)}')



    def make_gender_means_table(self):
        print('------------------------')
        print(language, by_author)
        print('------------------------')

        # Initialize a dictionary to store the means for each gender
        means_data = {}

        for attrtup in [('year', 'Year'), ('canon', 'Canon Score')]:
            attr, label = attrtup

            mh = MetadataHandler(self.language, by_author=self.by_author)
            df = mh.get_metadata(add_color=False)

            gender_map = {0: 'Male', 1: 'Female', 2: 'Anonymous', 3: 'Both'}
            df['gender'] = df['gender'].replace(gender_map)

            # Filter out entries with missing values for the current attribute
            df = df.dropna(subset=[attr])

            # Calculate means by gender
            attr_means = df.groupby('gender')[attr].mean()

            # Store the means in the dictionary
            means_data[label] = attr_means

        # Convert the dictionary to a DataFrame
        means_df = pd.DataFrame(means_data)
        means_df['Year'] = means_df['Year'].round(1)
        means_df['Canon Score'] = means_df['Canon Score'].round(3)

        # Print the resulting table
        print('Means of Year and Canon Scores by Gender:')
        print(means_df)
        print(means_df.to_latex(index=True))

        return means_df

    def make_gender_stripplot(self):
        for attrtup in [('year', 'Year'), ('canon', 'Canon Score')]:
            attr, label = attrtup

            mh = MetadataHandler(self.language, by_author=self.by_author)
            df = mh.get_metadata(add_color=False)

            gender_map = {0: 'Male', 1: 'Female', 2: 'Anonymous', 3: 'Both'}
            df['gender'] = df['gender'].replace(gender_map)

            order=['Male', 'Female']
            if 'Anonymous' in df['gender'].values:
                order.append('Anonymous')

            # Filter out entries with missing values for the current attribute
            df = df.dropna(subset=[attr])


            # Print means
            attr_means = df.groupby('gender')[attr].mean().reindex(['Male', 'Female', 'Anonymous', 'Both'], fill_value=0)
            print(f'Mean Canon Scores per {attr}:')
            print(attr_means)

            # # Creating the plot for every single value
            # plt.figure(figsize=(12, 8))
            # sns.stripplot(x='gender', y=attr, data=df, order=order, palette='Set2', jitter=True, size=6)

            # # Adding title and labels
            # plt.xlabel('Gender', fontsize=self.fontsize)
            # plt.ylabel(label, fontsize=self.fontsize)
            # # plt.title(f'{label} Distribution by Gender', fontsize=self.fontsize + 2)

            # # Changing the font size of the tick labels
            # plt.xticks(fontsize=self.fontsize - 5)
            # plt.yticks(fontsize=self.fontsize - 5)

            # # plt.show()
            # # Save the plot
            # self.save_data(data=plt, file_name=f'{attr}-by-gender_stripplot_byauthor-{self.by_author}.png')


    def make_gender_boxplots(self):
        for attrtup in [('year', 'Year'), ('canon', 'Canon Score')]:
            attr, label = attrtup

            mh = MetadataHandler(self.language, by_author=self.by_author)
            df = mh.get_metadata(add_color=False)

            gender_map = {0: 'Male', 1: 'Female', 2: 'Anonymous', 3: 'Both'}
            df['gender'] = df['gender'].replace(gender_map)

            male_attrs = df[df['gender'] == 'Male'][attr]
            female_attrs = df[df['gender'] == 'Female'][attr]
            print(f'Average male {attr}: ', male_attrs.mean())
            print(f'Average female {attr}: ', female_attrs.mean())
            # No t-test because assumptions are not met
            # Normality test
            # print(stats.shapiro(male_attrs))
            # print(stats.shapiro(female_attrs))

            # # Variance equality test
            # print(stats.levene(male_attrs, female_attrs))

            order=['Male', 'Female']
            if 'Anonymous' in df['gender'].values:
                order.append('Anonymous')

            # Create the boxplot
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='gender', y=attr, data=df, order=order)

            # Adding title and labels
            plt.xlabel('Gender', fontsize=self.fontsize)
            plt.ylabel(label, fontsize=self.fontsize)
            # Changing the font size of the tick labels
            plt.xticks(fontsize=self.fontsize - 5)
            plt.yticks(fontsize=self.fontsize - 5)

            self.save_data(data=plt, file_name=f'{attr}-by-gender_byauthor-{self.by_author}.png')
        

class ColorBarChart(DataHandler):
    '''
    Create a histogram of the canon scores. 
    Highlight the bars in the corresponding colors with which the values are represented in the network and MDS plots.
    The left and right edge of the buckets are not exaxtly 0, 0.1 ... because min and max values are not 0 and 1.
    '''
    def __init__(self, language, by_author):
        super().__init__(language, output_dir='text_statistics', data_type='png', by_author=by_author)
        self.by_author = by_author

    def make_plots(self):
        mh = MetadataHandler(self.language, by_author=self.by_author)
        df = mh.get_metadata(add_color=False)# [['canon', 'gender', 'year']]

        print('min canon: ', min(df['canon']))
        print('max canon: ', max(df['canon']))

        # Create histogram with 10 bins
        num_bins = 10
        counts, bins = np.histogram(df['canon'], bins=num_bins)
        cmap = plt.get_cmap('seismic')

        # Normalize the bin ranges to use with the colormap
        norm = plt.Normalize(vmin=bins.min(), vmax=bins.max())
        colors = cmap(norm(bins))

        # Plotting the histogram
        fig, ax = plt.subplots()

        for i in range(num_bins):
            ax.bar(bins[i], counts[i], width=(bins[1] - bins[0]), color=colors[i], edgecolor='black', align='edge')

        ax.set_xlabel('Canon Score')
        ax.set_ylabel('Frequency')
        # ax.set_title('Histogram of Canon Scores with Colormap')

        # Adjust x-axis ticks to align with bin edges
        # ax.set_xticks(bins)
        # ax.set_xlim([bins[0], bins[-1]])

        self.save_data(data=plt, file_name=f'histogram-with-colormaps_byauthor-{self.by_author}.png')


class MetadataStats(DataHandler):
    def __init__(self, language, by_author):
        super().__init__(language, output_dir='text_statistics', by_author=by_author)


    def get_stats(self):
        print(f'\n-------Language: {self.language}----------------------------')
        print(f'-------by_author: {self.by_author}----------------------------')
        # df = DataLoader(self.language).prepare_features(scale=True)
        mh = MetadataHandler(self.language, by_author=self.by_author)
        df = mh.get_metadata(add_color=False)

        # Find authors that occur only once
        if not self.by_author:
            author_counts = df['author'].value_counts()
            authors_occurring_once = author_counts[author_counts == 1]
            num_authors_occurring_once = len(authors_occurring_once)
            print(f"Number of authors that occur only once: {num_authors_occurring_once}")

        # Get the counts of gender
        gender_counts = df['gender'].value_counts()
        print('Gender counts:\n', gender_counts)
        print('------------------------\n')

        # Get the number of different authors 
        num_authors = df['author'].nunique()
        print(f'\nNumber of different authors: {num_authors}')
        print('------------------------\n')

        # Group the data by gender and count the different authors
        author_counts = df.groupby('gender')['author'].nunique()
        print('Nr different authors per gender\n', author_counts)
        print('------------------------\n')

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
    '''
    The value of each text is shown, therefore it makes no sense to make the plots for author-based analysis.
    '''
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='png')


    def make_plot(self):
        self.combine_author_work_metadata()
        self.get_x_position()
        self.plot()


    def combine_author_work_metadata(self):
        # df = DataLoader(self.language).prepare_features(scale=True)
        mh = MetadataHandler(self.language)
        df = mh.get_metadata(add_color=True)# [['canon', 'gender', 'year']]
        # author_filename_mapping = TextsByAuthor(self.language, by_author=False).author_filename_mapping
        # author = pd.DataFrame([(author, work) for author, works in author_filename_mapping.items() for work in works],
        #                 columns=['author', 'file_name'])
        # self.df = df.merge(author, left_index=True, right_on='file_name', validate='1:1')
        self.df = df


    def get_sorted_authors(self):
        mean_canon = self.df.groupby('author')['canon'].max()
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
        self.df['min_year'] = self.df.groupby('author')['year'].transform('mean') # Aggregated by mean, variable names still contain min!!
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


        ax.set_xlabel('Year', fontsize=self.fontsize)
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
    def __init__(self, language, by_author=False):
        super().__init__(language, output_dir='text_statistics', data_type='png', by_author=by_author)
        self.yeartup = ('year', 'Publication Year')
        self.canontup = ('canon', 'Canon Score')
        self.canonmin_tup = ('canon-min', 'Canon Score')
        self.canonmax_tup = ('canon-max', 'Canon Score')

    def get_correlations(self):
        mh = MetadataHandler(self.language, by_author=self.by_author)
        df = mh.get_metadata(add_color=False)
        rho, p_value = spearmanr(df['year'], df['canon'])
        print('\nRho', rho, 'pval', p_value, 'by_author', self.by_author, 'language', self.language, '\n\n')

        df_male = df.loc[df['gender'] == 0]
        df_female = df.loc[df['gender'] == 1]
        rho, p_value = spearmanr(df_male['year'], df_male['canon'])
        print('Gendder:m, Rho', rho, 'pval', p_value, 'by_author', self.by_author, 'language', self.language, '\n\n')
        rho, p_value = spearmanr(df_female['year'], df_female['canon'])
        print('Gender:f Rho', rho, 'pval', p_value, 'by_author', self.by_author, 'language', self.language, '\n\n')

    def make_joint_plot(self):
        mh = MetadataHandler(self.language, by_author=self.by_author)
        metadf = mh.get_metadata(add_color=False)
        metadf = metadf.rename(columns={'canon': 'Canon Score', 'year': 'Year'})
        sns.jointplot(x='Year', y='Canon Score', data=metadf, kind='scatter', marginal_kws=dict(bins=20))
        self.save_data(data=plt, file_name=f'joint-canon-year_byauthor-{self.by_author}')



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
        ax.scatter(x, y, s=3, label='Data points',)
        ax.plot(x, regression_line, color='black', label=f'Regression line: y={slope:.2f}x+{intercept:.2f}')

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


class PlotYearAndCanonPresentation(PlotYearAndCanon):
    '''
    This class is only used for thesis presentation.
    '''
    def __init__(self, language, by_author=False):
        super().__init__(language, by_author=by_author)
        self.canontup = ('canon', 'Canonization Score')

    def plot_single_var(self):
        if not self.by_author:
            plotlist = [self.yeartup, self.canontup]
        else:
            plotlist = [self.yeartup, self.canontup]

        for attr, yaxis_title in plotlist:
            mh = MetadataHandler(self.language, by_author=self.by_author)
            metadf = mh.get_metadata(add_color=False)
            print(metadf.shape)
            df = metadf.sort_values(by=attr, ascending=True)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df.index, df[attr], s=3)

            ax.set_xlabel('Number of Texts', fontsize=16)  # Increase x-axis label size
            ax.set_ylabel(yaxis_title, fontsize=16)        # Increase y-axis label size
            ax.grid(True, linestyle='--', alpha=0.5)

            max_text_index = len(df)
            text_ticks = list(range(0, max_text_index + 1, 100))
            ax.set_xticks(text_ticks)
            ax.set_xticklabels([str(t) for t in text_ticks], fontsize=14)  # Increase x-tick label size

            # Set size for y-axis tick labels
            ax.tick_params(axis='y', labelsize=14)  # Increase y-tick label size

            self.save_data(data=plt, file_name=f'{attr}_byauthor-{self.by_author}_presentation')



class CanonVariancePerAuthor(DataHandler):
    def __init__(self, language):
        super().__init__(language, output_dir='text_statistics', data_type='png')
        self.fontsize = 12

    def prepare_data(self):
        mh = MetadataHandler(self.language, by_author=self.by_author)
        df = mh.get_metadata(add_color=False)

        # keep only rows where author occurs more than once
        df = df[df['author'].duplicated(keep=False)]

        # Stevenson has high deviation
        filtered_df = df[df['author'].str.contains('stevenson', case=False, na=False)]
        print('Stevenson', filtered_df[['canon']])

        # Group by 'author' and calculate statistics
        df = df.groupby('author').agg(
            mean = ('canon', 'mean'),
            std_dev=('canon', 'std'),
            min_value=('canon', 'min'),
            max_value=('canon', 'max'),
            diff_max_min=('canon', lambda x: x.max() - x.min()),
            nr_texts=('canon', 'size')
        ).reset_index()

        df = df.sort_values(by=['diff_max_min', 'mean'], ascending=True)

        average_std_dev_per_author = df['std_dev'].mean()
        print('Average standard deviation per author:', average_std_dev_per_author)

        return df

    def make_plot(self):
        df = self.prepare_data()
        bar_width = 0.25
        index = np.arange(len(df))

        fig, ax1 = plt.subplots(figsize=(14, 7))

        bars1 = ax1.bar(index - bar_width, df['mean'], bar_width, color='blue', alpha=0.5, label='Mean')

        # Create a second y-axis for diff_max_min
        ax2 = ax1.twinx()
        bars2 = ax2.bar(index, df['diff_max_min'], bar_width, color='darkgreen', alpha=0.7, label='Diff Max-Min')
        bars3 = ax1.bar(index + bar_width, df['std_dev'], bar_width, color='#FF6F61', label='Std Dev')

        ax1.set_xlabel('Author', fontsize=self.fontsize)
        ax1.set_ylabel('Value', fontsize=self.fontsize)

        # Set x-ticks to match the positions of the bars
        ax1.set_xticks(index)
        ax1.set_xticklabels(df['author'], rotation=90, fontsize=self.fontsize-2)

        # Remove the right y-axis ticks
        ax2.set_yticks([])

        # Add a single legend for both plots
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2

        # Position legends in a single bounding box
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.05, 0.95), ncol=1, frameon=True)

        # Increase the gap between different authors
        # ax1.set_xlim(-bar_width, len(result) - 0.5)
        
        plt.tight_layout()

        path = self.get_file_path(file_name='canon-variance-per-author')
        plt.savefig(path, bbox_inches='tight', dpi=300)




if __name__ == '__main__':
    # for by_author in [False, True]:
    #     yacbl = YearAndCanonByLanguage(by_author=by_author)
    #     yacbl.make_language_boxplots()

    for language in ['eng', 'ger']:


        # cvpa = CanonVariancePerAuthor(language)
        # cvpa.prepare_data()
        # cvpa.make_plot()

        # pcspa = PlotCanonScoresPerAuthor(language)
        # pcspa.make_plot()

        # p = PlotCanonScoresPerAuthorByYearAndGender(language)
        # p.make_plot()

        
        for by_author in [False, True]:

            # mdstats = MetadataStats(language, by_author=by_author)
            # mdstats.get_stats()

            # pyac = PlotYearAndCanon(language, by_author=by_author)
            # pyac.get_correlations()
            # pyac.make_joint_plot()
            # pyac.plot_single_var()
            # pyac.plot_two_vars()

            pyac = PlotYearAndCanonPresentation(language, by_author=by_author)
            pyac.plot_single_var()

            # ybg = YearAndCanonByGender(language, by_author)
            # ybg.make_plots()
            # ybg.make_gender_stripplot()
            # ybg.make_gender_means_table()
            # ybg.canon_year_correlation_by_gender()
        
        # cbc = ColorBarChart(language, by_author)
        # cbc.make_plots()



        # ts = TextStatistics(language)
        # ts.get_longest_shortest_text()


        # pfd = PlotFeatureDist(language)
        # pfd.plot()



