import pandas as pd
import statistics
import os
import numpy as np
import heapq
from collections import Counter
import random
random.seed(3)
from itertools import product
from math import sqrt
from xgboost import XGBClassifier
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, TransformerMixin

# def get_data_subset_cols(X):
#     # X is full dataset
#     n_texts = X.index.to_series().nunique()
#     print('Get fulltext features ', n_texts, X.index.to_series())
#     X_fulltext = X.loc[:, X.apply(lambda col: True if col.nunique() == n_texts else False)]
#     cols_fulltext = set(X_fulltext.columns)
#     cols_chunk = set(X.columns) - cols_fulltext
#     return cols_fulltext, cols_chunk

def get_averaged_chunk_features(X, X_full=None):
    x_shape = X.shape
    # Replace value of chunk features with averages over all chunk features for a text
    print('-------------------------------\nx before averaging chunks: ', X.shape)
    X_chunk = ColumnTransformer(columns_to_drop=['_fulltext']).fit_transform(X_full, None)
    X_chunk = X_chunk.groupby(X_chunk.index).mean()
    X_new = X.merge(X_chunk, how='left', left_index=True, right_index=True, suffixes=('_unchanged', '_average')) ####################3
    X_new = ColumnTransformer(columns_to_drop=['_unchanged']).fit_transform(X_new, None)
    n_texts = X.index.nunique()
    n_unique_in_cols = X_new.apply(lambda col: True if col.nunique() == n_texts else False)
    print('All columns averaged chunk df have 1 value per file name: ', n_unique_in_cols.all())
    print(n_unique_in_cols)
    print(n_unique_in_cols.value_counts())
    print('Cols that have more different values (fulltext features): ', n_unique_in_cols.index[~n_unique_in_cols])
    if not X_new == x_shape:
        print('\n?????????????????\nshape of x before and after averaging chunks not the same')
    return X_new


class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self. columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def _drop_column(column):
            for string in self.columns_to_drop:
                if string in column:
                    return True
            return False

        if self.columns_to_drop != None:
            X_new = X[[column for column in X.columns if not _drop_column(column)]]# .reset_index(drop=True)##########
        # columns_before_drop = set(X.columns)
        # columns_after_drop = set(X_new.columns)
        # print(f'Dropped {len(columns_before_drop - columns_after_drop)} columns.') 
        return X_new

# def custom_pca(X):
#     X = StandardScaler().fit_transform(X)
#     max_components = min(X.shape)
#     for i in range(5, X.shape[1], int((X.shape[1] - 5) / 10)):
#         if i < max_components:
#             pca = PCA(n_components=i)
#             X_trans = pca.fit_transform(X)
#             if pca.explained_variance_ratio_.sum() >= 0.95:
#                 break
#         else:
#             pca = PCA(n_components=max_components)
#             X_trans = pca.fit_transform(X)
#             return X_trans


# from sklearn.impute import KNNImputer
# def _impute(X):
#     imputer = KNNImputer()
#     X = imputer.fit_transform(X)
#     return X

def get_min_fold_size(cv):
    min_fold_size = 100000
    for tup in cv:
        for inner_list in tup:
            if len(inner_list) < min_fold_size:
                min_fold_size = len(inner_list)
    return min_fold_size


def permute_params(model, **kwargs):
    '''
    This function takes a class and named lists of parameters: param_name = [value1, value2].
    It returns an instance of the class for every combination of the parameters of the different lists.
    '''
    param_names = list(kwargs.keys())
    params_values = list(kwargs.values())
    # All combinations of the parameters of the different lists
    combinations = list(product(*params_values))
    # Combine parameter names and parameters and instantiate class
    all_models = []
    for tup in combinations:
        x = dict()
        for i in range(0, len(tup)):
            x.update(({param_names[i]: tup[i]}))
        all_models.append(model(**x))
    return all_models

def _score_regression(estimator, X, y):
    print('scoring--------------------------', 'estimator', estimator)
    y_pred = estimator.predict(X)
    '''
    Calculate task-specific evaluation metrics.
    '''
    try:
        corr, corr_pvalue= pearsonr(y, y_pred)
    except:
        print(f'Correlation coefficient not calculated.') ############################################
        corr = 0.888888
        corr_pvalue = 0.1
    r2 = r2_score(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    return {'corr': corr,
            'corr_pvalue': corr_pvalue,
            'r2': r2,
            'rmse': rmse,
            'mae': mae}

def get_labels(label_type, language, canonscores_dir=None, sentiscores_dir=None, metadata_dir=None):
    if label_type == 'canon':
        canon_file = '210907_regression_predict_02_setp3_FINAL.csv'
        labels = pd.read_csv(os.path.join(canonscores_dir, canon_file), sep=';')[['file_name', 'm3']]
        labels = labels.rename(columns={'m3': 'y'})
        labels = labels.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
        
    elif label_type == 'library':
        if language == 'eng':
            library_file = 'ENG_texts_circulating-libs.csv'
        else:
            library_file = 'GER_texts_circulating-libs.csv'
        labels = pd.read_csv(os.path.join(metadata_dir, library_file), sep=';', header=0)[['file_name', 'sum_libraries']]
        labels = labels.rename(columns={'sum_libraries': 'classified'})
        labels['classified'] = labels['classified'].apply(lambda x: 1 if x!=0 else 0)

    else:
        # Combine all labels and filenames in one file
        # File with sentiment scores for both tools
        if language == 'eng':
            scores_file = 'ENG_reviews_senti_FINAL.csv'
        else:
            scores_file = 'GER_reviews_senti_FINAL.csv'
        score_labels = pd.read_csv(os.path.join(sentiscores_dir, scores_file), sep=';', header=0)

        # File with class labels
        if language == 'eng':
            class_file = 'ENG_reviews_senti_classified.csv'
        else:
            class_file = 'GER_reviews_senti_classified.csv'
        class_labels = pd.read_csv(os.path.join(sentiscores_dir, class_file), sep=';', header=0)[['textfile', 'journal', 'file_name', 'classified']]

        labels = score_labels.merge(right=class_labels, 
                                    on=['textfile', 'journal', 'file_name'], 
                                    how='outer', 
                                    suffixes=(None, '_classlabels'), 
                                    validate='one_to_one')
        labels = labels[['sentiscore_average', 'sentiment_Textblob', 'file_name', 'classified']]

        # Sentiment Scores
        if (label_type == 'textblob') or (label_type == 'sentiart'):
            def _average_scores(group):
                textblob_value = group['sentiment_Textblob'].mean()
                sentiart_value = group['sentiscore_average'].mean()
                group['sentiment_Textblob'] = textblob_value
                group['sentiscore_average'] = sentiart_value
                return group        
            # Get one label per book
            labels = labels.groupby('file_name').apply(_average_scores)

        elif label_type == 'combined':
            def _aggregate_scores(row):
                if row['classified'] == 'positive':
                    score = row['sentiment_Textblob']
                elif row['classified'] == 'negative':
                    score = row['sentiscore_average']
                elif row['classified'] == 'not_classified':
                    score = statistics.mean([row['sentiment_Textblob'], row['sentiscore_average']]) 
                return score
            labels['combined'] = labels.apply(lambda row: _aggregate_scores(row), axis=1)
            #labels['combined'].sort_values().plot(kind='bar', figsize=(10, 5))

        elif (label_type == 'multiclass') or (label_type == 'binary'):
            #Assign one class label per work
            grouped_docs = labels.groupby('file_name')
            single_review = grouped_docs.filter(lambda x: len(x)==1)
            # If work has multiple reviews, keep labels only if reviews are not opposing (positive and negative)
            multiple_reviews = grouped_docs.filter(lambda x: len(x)>1 and not('negative' in x['classified'].values and 'positive' in x['classified'].values))
            #opposed_reviews = grouped_docs.filter(lambda x: len(x)>1 and ('negative' in x['classified'].values and 'positive' in x['classified'].values))  
            def _select_label(group):
                # Keep label with higher count, keep more extreme label if counts are equal
                count = group['classified'].value_counts().reset_index().rename(columns={'index': 'classified', 'classified': 'count'})
                if count.shape[0]>1:
                    if count.iloc[0,1] == count.iloc[1,1]:
                        grouplabel = count['classified'].max()
                    else:
                        grouplabel = count.iloc[0,0]
                    group['classified'] = grouplabel
                return group
            multiple_reviews = multiple_reviews.groupby('file_name').apply(_select_label)
            labels = pd.concat([single_review, multiple_reviews])
            labels['classified'] = labels['classified'].replace(to_replace={'positive': 3, 'not_classified': 2, 'negative': 1})

            if label_type =='binary':
                # Create label reviewed/not reviewed
                labels['classified'] = 1
            
    labels = labels.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
    labels = labels.drop_duplicates(subset='file_name')
    if label_type == 'canon':
        labels = labels.rename(columns={'m3': 'y'})
    if label_type == 'textblob':
        labels = labels[['file_name', 'sentiment_Textblob']].rename(columns={'sentiment_Textblob': 'y'})
    elif label_type == 'sentiart':
        labels = labels[['file_name', 'sentiscore_average']].rename(columns={'sentiscore_average': 'y'})
    elif (label_type == 'multiclass') or (label_type == 'binary'):
        labels = labels[['file_name', 'classified']].rename(columns={'classified': 'y'})
    elif label_type == 'combined':
        labels = labels[['file_name', 'combined']].rename(columns={'combined': 'y'})
    elif label_type == 'library':
        labels = labels[['file_name', 'classified']].rename(columns={'classified': 'y'})
    return labels


def get_data(task_type, language, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir): 
    X = pd.read_csv(os.path.join(features_dir, 'cacb_features.csv'))
    y = get_labels(label_type, language, canonscores_dir, sentiscores_dir, metadata_dir)

    if task_type == 'regression':
        df = X.merge(right=y, on='file_name', how='inner', validate='many_to_one')
    else:
        # Select books written after year of first review
        # Keep only those texts that that were published after the first review had appeared
        df = X.merge(right=y, on='file_name', how='left', validate='many_to_one')
        df['y'] = df['y'].fillna(value=0)
        book_year = df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
        review_year = book_year[df['y'] != 0]
        df = df.loc[book_year>=min(review_year)]

    X = df.drop(labels=['y'], axis=1, inplace=False).set_index('file_name', drop=True)
    y = df[['y']]

    return X, y


class XGBClassifierMulticlassImbalanced(XGBClassifier):
    def __init__(self, objective, random_state, use_label_encoder):
        super().__init__(objective, random_state, use_label_encoder)

    def fit(self, X,y):
        weights = compute_sample_weight('balanced', y=y)
        super().fit(X, y, sample__weight=weights)

        return self


class AuthorCV():
    '''
    Split book names into n folds.
    All works of an author are put into the same fold.
    Adapted from https://www.titanwolf.org/Network/q/b7ee732a-7c92-4416-bc80-a2bd2ed136f1/y
    '''
    def __init__(self, df, n_folds, seed=4, stratified=False, return_indices=True):
        self.df = df
        self.n_folds = n_folds
        self.stratified = stratified
        self.return_indices = return_indices
        self.file_names = df.index.to_series()
        self.author_bookname_mapping, self.works_per_author = self.get_author_books()
        random.seed(seed)

    def get_author_books(self):
        authors = []
        author_bookname_mapping = {}
        #Get books per authors
        for file_name in self.file_names:
            author = '_'.join(file_name.split('_')[:2])
            authors.append(author)
            if author in author_bookname_mapping:
                author_bookname_mapping[author].append(file_name)
            else:
                author_bookname_mapping[author] = []
                author_bookname_mapping[author].append(file_name)
                
        # Aggregate if author has collaborations with others
            agg_dict = {'Hoffmansthal_Hugo': ['Hoffmansthal_Hugo-von'], 
                        'Schlaf_Johannes': ['Holz-Schlaf_Arno-Johannes'],
                         'Arnim_Bettina': ['Arnim-Arnim_Bettina-Gisela'],
                         'Stevenson_Robert-Louis': ['Stevenson-Grift_Robert-Louis-Fanny-van-de', 
                                                   'Stevenson-Osbourne_Robert-Louis-Lloyde']}
            
        for author, aliases in agg_dict.items():
            if author in authors:
                for alias in aliases:
                    if alias in authors:
                        author_bookname_mapping[author].extend(author_bookname_mapping[alias]) 
                        del author_bookname_mapping[alias]
                        authors = [author for author in authors if author != alias]
        
        works_per_author = Counter(authors)
        return author_bookname_mapping, works_per_author
    
    def get_folds(self):
        splits = [[] for _ in range(0,self.n_folds)]

        if self.stratified == True:
            rare_labels = sorted(self.df['y'].unique().tolist())[1:]
            splits_counter = [0 for _ in range(0, self.n_folds)]
            for rare_label in rare_labels:
                # If stratified, first distribute authors that have rarest label over split so that the author is assigned to the split with the smallest number or rarest labels
                counts = [(0,i) for i in range (0, self.n_folds)]
                # heapify based on first element of tuple, inplace
                heapq.heapify(counts)
                for author in list(self.works_per_author.keys()):
                    rare_label_counter = 0
                    for curr_file_name in self.author_bookname_mapping[author]:
                        if self.df.loc[self.df['file_name'] == curr_file_name].squeeze().at['y'] == rare_label:
                            rare_label_counter += 1
                    if rare_label_counter != 0:
                        author_workcount = self.works_per_author.pop(author)
                        count, index = heapq.heappop(counts)
                        splits[index].append(author)
                        splits_counter[index] += author_workcount
                        heapq.heappush(counts, (count + rare_label_counter, index))
            totals = [(splits_counter[i],i) for i in range(0,len(splits_counter))]
        else:
            totals = [(0,i) for i in range (0, self.n_folds)]
        # heapify based on first element of tuple, inplace
        heapq.heapify(totals)
        while bool(self.works_per_author):
            author = random.choice(list(self.works_per_author.keys()))
            author_workcount = self.works_per_author.pop(author)
            # find split with smallest number of books
            total, index = heapq.heappop(totals)
            splits[index].append(author)
            heapq.heappush(totals, (total + author_workcount, index))

        if not self.return_indices:
            # Return file_names in splits
            #Map author splits to book names
            map_splits = []
            for split in splits:
                new = []
                for author in split:
                    new.extend(self.author_bookname_mapping[author])
                map_splits.append(new)

            if self.stratified == True:
                for split in map_splits:
                    split_df = self.df[self.df['file_name'].isin(split)]
        else:
            # Return indices of file_names in split
            file_name_idx_mapping = dict((file_name, index) for index, file_name in enumerate(self.file_names))
            map_splits = []
            for split in splits:
                test_split = []
                for author in split:
                    # Get all indices from file_names from the same author
                    test_split.extend([file_name_idx_mapping[file_name] for file_name in  self.author_bookname_mapping[author]])
                # Indices of all file_names that are not in split
                train_split = list(set(file_name_idx_mapping.values()) - set(test_split))
                map_splits.append((train_split, test_split))
        return map_splits
