import pandas as pd
import statistics
import os
import numpy as np
from copy import deepcopy
from decimal import Decimal
import random
random.seed(3)
from itertools import product
from math import sqrt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor, XGBClassifier

def analyze_cv(X, cv):
    X_fulltext = ColumnTransformer(columns_to_drop='_chunk').fit_transform(X, None)
    for train_idxs, test_idxs in cv:
        groups = get_author_groups(X_fulltext)
        X_train = X_fulltext.iloc[train_idxs]
        groups_train = groups.iloc[train_idxs]
        X_test = X_fulltext.iloc[test_idxs]
        groups_test = groups.iloc[test_idxs]
        dfs = {'X_train': X_train, 'X_test': X_test} #'groups_train': groups_train, 'groups_test': groups_test
        for name, df in dfs.items():
            print(f'Shape of {name} before removing duplicate rows: {df.shape}')
            dfs[name] = df = df[~df.index.duplicated(keep="first")]
            print(f'Shape of {name} after removing duplicate rows: {df.shape}')


def get_document_features(X):
    print('X before taking docu features,', X.shape)
    X_new = ColumnTransformer(columns_to_drop=['_chunk']).fit_transform(X, None).drop_duplicates()
    print('X after taking docu features,', X.shape)
    return X_new


def average_chunk_features(X):
    '''
    Average over chunk features for each document. Result is one value per feature and document.
    # All columns averaged chunk df have 1 value per file name:  False
    # Check if several features have value 0
    '''
    X_av = X.groupby(X.index).mean()
    X_new = X.merge(X_av, how='left', left_index=True, right_index=True, suffixes=('_unchanged', '_average'), validate='many_to_one')
    X_new = ColumnTransformer(columns_to_drop=['_unchanged']).fit_transform(X_new, None).drop_duplicates()
    n_texts = X.index.nunique()
    n_unique_in_cols = X_new.apply(lambda col: True if col.nunique() == n_texts else False)
    print('All columns averaged chunk df have 1 value per file name: ', n_unique_in_cols.all())
    print('Cols that have more different values (fulltext features): ', n_unique_in_cols.index[~n_unique_in_cols])
    return X_new


def refit_regression(cv_results):
    # refit using a callable
    # find highest correlation coefficiant that has a significant harmonic p-value
    df = pd.DataFrame(cv_results)
    significance_threshold = 0.1
    df['harmonic_pvalue'] = df.apply(apply_harmonic_pvalue, axis=1)
    nr_max_corr = (df['mean_test_corr'].values == df['mean_test_corr'].max()).sum()
    print('nr max corr: ', nr_max_corr)
    df = df.sort_values(by='mean_test_corr', axis=0, ascending=False)
    print(df.index)

    best_corr = 0
    for index, row in df.iterrows():
        if row['harmonic_pvalue'] < significance_threshold:
            best_corr = row['mean_test_corr']
            best_idxs = df.index[(df['mean_test_corr'] == best_corr) & (df['harmonic_pvalue']<significance_threshold)].tolist()
            if len(best_idxs) > 1:
                print(f'More than 1 model has the highest correlation coefficient and a significant harmonic p-value. Only one model is returned')
            else:
                print('Only one model has highest correlation coefficient and a significant harmonic p-value.')
            best_index = best_idxs[0]
            break
    else:
    # Run when no break occurs
        print(f'No model has a hamonic p value below {significance_threshold}. Model with the highest correlation coefficient is returned.')
        best_index = df['mean_test_corr'].idxmax()

    print(f'Best score:{best_corr}')
    print(f'Best index:{best_index}')

    return best_index
    

def apply_harmonic_pvalue(row):
    # Harmonic mean p-value
    # Takes row from GridSearchCV.cv_results_ as input
    # Match columns that contain test pvalues for each split
    pvalues = row[row.index.str.contains('split._test_corr_pvalue', regex=True)]
    try:
        denominator = sum([Decimal(1)/Decimal(x) for x in pvalues])
        harmonic_pval = len(pvalues)/denominator
    except ZeroDivisionError:
        print('Could not calculate harmonic p-value because cv p-values are 0 or approximately 0.')
        harmonic_pval = np.nan
    finally:
        return harmonic_pval


def score_regression(estimator, X, y):
    '''
    Multiple evaluation metrics for regression.
    Callable for the 'scoring' parameter in GridSearchCV. 
    '''

    y_pred = estimator.predict(X)
    corr, corr_pvalue = pearsonr(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    return {'corr': corr,
            'corr_pvalue': corr_pvalue,
            'r2': r2,
            'rmse': rmse,
            'mae': mae}

def score_binary(estimator, X, y):
    '''
    Multiple evaluation metrics for binary classification.
    Callable for the 'scoring' parameter in GridSearchCV. 
    '''
    y_pred = estimator.predict(X)
    acc = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    return {'acc': acc,
            'balanced_acc': balanced_acc}


def score_multiclass(estimator, X, y):
    '''
    Multiple evaluation metrics for multiclass classification.
    Callable for the 'scoring' parameter in GridSearchCV. 
    '''
    y_pred = estimator.predict(X)
    y_prob = estimator.predict_proba(X)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')

    # PR AUC
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    pr_auc = auc(recall, precision)
    return {'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'PR_AUC': pr_auc}


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


def get_data(task, language, label_type, features_dir, canonscores_dir, sentiscores_dir, metadata_dir): 
    '''
    The following file names had inconsistent spelling between versions of the data, check if errors.
    'Hegeler_Wilhelm_Mutter-Bertha_1893': 'Hegelers_Wilhelm_Mutter-Bertha_1893',
    'Hoffmansthal_Hugo_Ein-Brief_1902': 'Hoffmansthal_Hugo-von_Ein-Brief_1902'
        '''
    print('task get data', task)
    X = pd.read_csv(os.path.join(features_dir, 'cacb_features.csv'))
    print(f'Nr chunks for {language}: {X.shape[0]}')
    print(f'Nr texts for {language}: {len(X.file_name.unique())}')
    y = get_labels(label_type, language, canonscores_dir, sentiscores_dir, metadata_dir)
    
    print(y)
    # For english regression, 1 label is duplicated
    print(f'Nr labels for {language} {task}: {y.shape}, {y.y.nunique()}')
    #y_before_merge = set(y['file_name'])

    if task == 'regression':
        df = X.merge(right=y, on='file_name', how='inner', validate='many_to_one')
    else:
        # Select books written after year of first review
        # Keep only those texts that that were published after the first review had appeared
        df = X.merge(right=y, on='file_name', how='left', validate='many_to_one')
        df['y'] = df['y'].fillna(value=0)
        book_year = df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
        review_year = book_year[df['y'] != 0]
        df = df.loc[book_year>=min(review_year)]
        print('Min review year', min(review_year))

    print(f'Nr texts for {language} after combining with labels: {df.file_name.nunique()}')
    print(f'Nr labels for {language} {task} after combining with features: {df.y.shape}')
    # y_after_merge = set(df['file_name'])
    # print('labels difference', list(y_before_merge - y_after_merge))

    X = df.drop(labels=['y'], axis=1, inplace=False).set_index('file_name', drop=True)
    # y = df[['file_name', 'y']].set_index('file_name', drop=True)
    y = df[['y', 'file_name']].set_index('file_name', drop=True)


    print('NaN in X: ', X.isnull().values.any())
    print('NaN in y: ', y.isnull().values.any())
    print('X, y shape', X.shape, y.shape)

    return X, y


class CustomXGBClassifier(XGBClassifier):
    '''
    XGBClassifier that sets sample_weights parameter in fit()
    To be used in pipeline & grid search because fit() has common interface for all estimators
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X,y):
        weights = compute_sample_weight('balanced', y=y)
        super().fit(X, y, sample_weight=weights)

        return self


def get_author_groups(X):
    '''
    Create column with author name so that all texts by one author belong to the same group
    Account for different spelling versions of author name
    '''
    alias_dict = {
    'Hoffmansthal_Hugo': ['Hoffmansthal_Hugo-von'], 
    'Schlaf_Johannes': ['Holz-Schlaf_Arno-Johannes'],
    'Arnim_Bettina': ['Arnim-Arnim_Bettina-Gisela'],
    'Stevenson_Robert-Louis': ['Stevenson-Grift_Robert-Louis-Fanny-van-de', 
                                'Stevenson-Osbourne_Robert-Louis-Lloyde']}

    authors = X.index.to_frame(name='file_name')
    authors['author'] = authors['file_name'].str.split('_').str[:2].str.join('_')

    for author, aliases in alias_dict.items():
        for alias in aliases:
            authors['author'].replace(to_replace=alias, value=author, inplace=True)
    return authors['author']


class CustomGroupKFold():
    '''
    Split book names into n folds.
    All works of an author are put into the same fold.
    If stratified==True, the folds are created so that each fold contains approximately the same number of samples from each class.
    For classification, all labels must be represented in all classes. XGBClassifier throws an error if labels do not go from 0 to num_classes-1.
    '''
    def __init__(self, n_splits, stratified=False):
        self.n_splits = n_splits
        self.stratified = stratified

    def split(self, X, y):
        author_groups = get_author_groups(X)
        print(author_groups)
        indices = []
        if self.stratified:
            cv = StratifiedGroupKFold(n_splits=self.n_splits)
            splits = cv.split(X, y, groups=author_groups)
        else:
            cv = GroupKFold(n_splits=self.n_splits)
            splits = cv.split(X, y, groups=author_groups)
        for train_idxs, test_idxs in splits:
            indices.append((train_idxs, test_idxs))

        return indices

def get_model(task, testing):
    # All paramters
    # Separate grids for conditional parameters
    param_grid_regression = [
        {'clf': (SVR(),),
        'clf__C': [0.1, 1],
        'clf__epsilon': [0.001, 0.01]},
        {'clf': (Lasso(),),
        'clf__alpha': [1,10, 100, 1000],
        'clf__tol': [0.0001], # default
        'clf__max_iter': [1000]}, # default
        {'clf': (XGBRegressor(objective='reg:squarederror', random_state=7),),
        'clf__max_depth': [2,4,6,8, 20],
        'clf__learning_rate': [None, 0.033, 0.1, 1], #0.01 does not produce result
        'clf__colsample_bytree': [0.33, 0.60, 0.75]}, 
        ]

    param_grid_binary = [
        {'clf': (SVC(class_weight='balanced'),),
        'clf__C': [0.1, 1, 10, 100, 1000]},
        {'clf': (CustomXGBClassifier(objective='binary:logistic', random_state=7, use_label_encoder=False),),
        'clf__max_depth': [2,4,6,8, 20],
        'clf__learning_rate': [None, 0.01, 0.033, 0.1],
        'clf__colsample_bytree': [0.33, 0.60, 0.75]}, 
        ]

    param_grid_multiclass= [
        {'clf': (SVC(class_weight='balanced'),),
        'clf__C': [0.1, 1, 10, 100, 1000]},
        {'clf': (CustomXGBClassifier(objective='multi:softmax', random_state=7, use_label_encoder=False),),
        'clf__max_depth': [2,4,6,8, 20],
        'clf__learning_rate': [None, 0.01, 0.033, 0.1],
        'clf__colsample_bytree': [0.33, 0.60, 0.75]}, 
        ]

    if testing:
        print('Using testing param grid.')
        param_grid_regression = [
            {'clf': (SVR(),),
            'clf__C': [0.1],
            'clf__epsilon': [0.001]},
            {'clf': (Lasso(),),
            'clf__alpha': [1],
            'clf__tol': [0.0001], # default
            'clf__max_iter': [1000]}, # default
            {'clf': (XGBRegressor(objective='reg:squarederror', random_state=7),),
            'clf__max_depth': [20],
            'clf__learning_rate': [0.033],
            'clf__colsample_bytree': [0.33]}, 
            ]

        param_grid_binary = [
            {'clf': (SVC(class_weight='balanced'),),
            'clf__C': [1]},
            {'clf': (CustomXGBClassifier(objective='binary:logistic', random_state=666, use_label_encoder=False),),
            'clf__max_depth': [20],
            'clf__learning_rate': [0.1],
            'clf__colsample_bytree': [0.75]}, 
            ]

        param_grid_multiclass= [
            {'clf': (SVC(class_weight='balanced'),),
            'clf__C': [1]},
            {'clf': (CustomXGBClassifier(objective='multi:softmax', random_state=777, use_label_encoder=False),),
            'clf__max_depth': [20],
            'clf__learning_rate': [0.1],
            'clf__colsample_bytree': [0.75]}, 
            ]

    models = {
        'regression': {
            'labels': ['textblob', 'sentiart', 'combined'],
            'scoring': score_regression,
            'refit': refit_regression,
            'stratified': False,
            'param_grid': param_grid_regression,
            'param_data_subset': ['passthrough', ColumnTransformer(columns_to_drop=['_fulltext'])] #Use chunk features, drop fulltext features
            },
        'binary': {
            'labels': ['binary'],
            'refit': 'balanced_acc',
            'scoring': score_binary,
            'stratified': True,
            'param_grid': param_grid_binary,
            'param_data_subset': [] # Don't use chunk features
            },
        'multiclass':{
            'labels': ['multiclass'],
            'refit': 'f1',
            'scoring': score_multiclass,
            'stratified': True,
            'param_grid': param_grid_multiclass,
            'param_data_subset': [] # Don't use chunk features
            }
        }

    models['library'] = deepcopy(models['binary'])
    models['library']['labels'] = ['library']

    # Overwrite for testing 
    if testing:
        #models['regression']['features'] = ['book', 'chunk']
        models['regression']['labels'] = ['sentiart']
        #models['binary']['features'] = ['book']
        models['library']['features'] = ['book']
        models['multiclass']['features'] = ['book']

    return models[task]